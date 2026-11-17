"""Command line entry points for data preparation and model experiments."""

import argparse
import hashlib
import json
import platform
import sys
import tomllib
from collections import Counter
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError

from science_tutor.baseline import BASELINE_VERSION, Mode, compare_prompts
from science_tutor.config import ExperimentConfig, load_config
from science_tutor.data import DatasetValidationError, load_scienceqa
from science_tutor.download import SourceDownloadError, fetch_scienceqa
from science_tutor.duplicates import write_duplicate_report
from science_tutor.eligibility import (
    POLICY_VERSION,
    assess_eligibility,
    write_eligibility_report,
)
from science_tutor.inference import load_generator
from science_tutor.prompts import build_prompt_dataset
from science_tutor.reports import write_json_report
from science_tutor.sampling import build_sample_manifest


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _run_baselines(args: argparse.Namespace, config: ExperimentConfig) -> int:
    try:
        artifact = fetch_scienceqa(config.dataset, args.data_dir)
        if args.output.resolve() in {
            args.config.resolve(),
            artifact.path.resolve(),
            artifact.path.with_name("source.json").resolve(),
        }:
            raise ValueError("Report output must not overwrite configuration or source files.")
        dataset = build_prompt_dataset(load_scienceqa(artifact.path), config)
        if args.shots > len(dataset["train"]):
            raise ValueError("Few-shot count exceeds available training rows.")
        print(f"Loading {config.model.model_id} ({args.device})", file=sys.stderr)
        generate, runtime = load_generator(
            config.model, args.device, args.max_new_tokens, args.model_cache
        )

        def progress(mode: Mode, done: int, total: int) -> None:
            print(f"{mode}: {done}/{total}", file=sys.stderr)

        start = perf_counter()
        result = compare_prompts(
            dataset,
            generate,
            shots=args.shots,
            limit=args.limit,
            batch_size=args.batch_size,
            progress=progress,
        )
        elapsed = round(perf_counter() - start, 3)
        report = {
            "baseline_version": BASELINE_VERSION,
            "prompt_format_version": dataset["format_version"],
            "prompt_dataset_sha256": hashlib.sha256(
                json.dumps(dataset, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
            "source": dataset["source"],
            "sampling_policy": dataset["sampling_policy"],
            "seed": dataset["seed"],
            "quality_exclusions": dataset["quality_exclusions"],
            "model": config.model.model_dump(),
            "generation": {
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                **runtime,
            },
            "inference_seconds": elapsed,
            **result,
        }
        write_json_report(args.output, report)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Cannot run baselines: {exc}", file=sys.stderr)
        return 2
    brief = {
        key: {metric: value for metric, value in row.items() if metric != "predictions"}
        for key, row in (("zero_shot", result["zero_shot"]), ("few_shot", result["few_shot"]))
    }
    print(json.dumps({"output": str(args.output), "inference_seconds": elapsed, **brief}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command and return a conventional process exit code."""
    parser = argparse.ArgumentParser(
        prog="science-tutor",
        description="Validate and inspect educational LoRA experiment plans.",
    )
    parser.add_argument("--version", action="version", version=version("science-tutor-lora"))
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate a TOML experiment configuration.")
    validate.add_argument("config", type=Path)
    commands.add_parser("doctor", help="Print local runtime metadata without loading a model.")
    inspect = commands.add_parser("inspect-data", help="Validate a local ScienceQA JSON file.")
    inspect.add_argument("source", type=Path)
    fetch = commands.add_parser("fetch-data", help="Fetch or verify the pinned ScienceQA JSON.")
    fetch.add_argument("config", type=Path)
    fetch.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    filter_data = commands.add_parser(
        "filter-data", help="Audit eligibility of verified ScienceQA."
    )
    filter_data.add_argument("config", type=Path)
    filter_data.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    filter_data.add_argument("--output", type=Path, default=Path("data/processed/eligibility.json"))
    duplicates = commands.add_parser(
        "audit-duplicates", help="Audit duplicates among eligible records."
    )
    duplicates.add_argument("config", type=Path)
    duplicates.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    duplicates.add_argument("--output", type=Path, default=Path("data/processed/duplicates.json"))
    samples = commands.add_parser("build-samples", help="Quarantine duplicates and sample IDs.")
    samples.add_argument("config", type=Path)
    samples.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    samples.add_argument("--output", type=Path, default=Path("data/processed/samples.json"))
    prompts = commands.add_parser("build-prompts", help="Build the train and validation prompts.")
    prompts.add_argument("config", type=Path)
    prompts.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    prompts.add_argument("--output", type=Path, default=Path("data/processed/prompts.json"))
    baselines = commands.add_parser(
        "run-baselines", help="Compare zero-shot and few-shot accuracy."
    )
    baselines.add_argument("config", type=Path)
    baselines.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    baselines.add_argument("--model-cache", type=Path, default=Path("data/huggingface/hub"))
    baselines.add_argument("--output", type=Path, default=Path("outputs/baselines.json"))
    baselines.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    baselines.add_argument("--shots", type=_positive_int, default=2)
    baselines.add_argument("--batch-size", type=_positive_int, default=8)
    baselines.add_argument("--max-new-tokens", type=_positive_int, default=128)
    baselines.add_argument(
        "--limit", type=_positive_int, help="Score only the first N validation IDs."
    )
    args = parser.parse_args(argv)

    if args.command == "doctor":
        print(
            json.dumps(
                {
                    "package_version": version("science-tutor-lora"),
                    "python": platform.python_version(),
                    "system": platform.system(),
                    "machine": platform.machine(),
                    "executable": sys.executable,
                    "accelerator": "not probed; model dependencies are not installed by default",
                },
                indent=2,
            )
        )
        return 0

    if args.command == "inspect-data":
        try:
            records = load_scienceqa(args.source)
        except (OSError, DatasetValidationError) as exc:
            print(f"Invalid dataset: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "records": len(records),
                    "splits": dict(Counter(record.split for record in records.values())),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    try:
        config = load_config(args.config)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValidationError) as exc:
        print(f"Invalid experiment configuration: {exc}", file=sys.stderr)
        return 2

    if args.command == "run-baselines":
        return _run_baselines(args, config)

    if args.command == "fetch-data":
        try:
            artifact = fetch_scienceqa(config.dataset, args.data_dir)
        except (OSError, SourceDownloadError) as exc:
            print(f"Cannot fetch dataset: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "path": str(artifact.path),
                    "sha256": artifact.sha256,
                    "bytes": artifact.size_bytes,
                    "cached": artifact.cached,
                },
                indent=2,
            )
        )
        return 0

    actions = {
        "filter-data": "filter dataset",
        "audit-duplicates": "audit duplicates",
        "build-samples": "build samples",
        "build-prompts": "build prompts",
    }
    if args.command in actions:
        try:
            artifact = fetch_scienceqa(config.dataset, args.data_dir)
            protected = {
                args.config.resolve(),
                artifact.path.resolve(),
                (artifact.path.parent / "source.json").resolve(),
            }
            if args.output.resolve() in protected:
                raise ValueError("Report output must not overwrite configuration or source files.")
            records = load_scienceqa(artifact.path)
            if args.command == "build-prompts":
                dataset = build_prompt_dataset(records, config)
                write_json_report(args.output, dataset)
                print(
                    json.dumps(
                        {
                            "output": str(args.output),
                            "format_version": dataset["format_version"],
                            "train": len(dataset["train"]),
                            "validation": len(dataset["validation"]),
                        },
                        indent=2,
                    )
                )
                return 0
            if args.command == "build-samples":
                manifest = build_sample_manifest(records, config)
                write_json_report(args.output, manifest)
                print(
                    json.dumps(
                        {
                            "manifest": str(args.output),
                            "seed": manifest["seed"],
                            "by_split": manifest["by_split"],
                        },
                        indent=2,
                    )
                )
                return 0
            if args.command == "audit-duplicates":
                duplicate_summary = write_duplicate_report(args.output, records, config.dataset)
                print(json.dumps({"report": str(args.output), **duplicate_summary}, indent=2))
                return 0
            decisions = assess_eligibility(records, config.dataset.subject)
            summary = write_eligibility_report(args.output, decisions, config.dataset)
        except (OSError, ValueError) as exc:
            print(f"Cannot {actions[args.command]}: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {"report": str(args.output), "policy_version": POLICY_VERSION, **summary}, indent=2
            )
        )
        return 0

    print(config.model_dump_json(indent=2))
    return 0
