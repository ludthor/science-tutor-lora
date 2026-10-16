"""A lightweight command line interface; model libraries are added with training."""

import argparse
import json
import platform
import sys
import tomllib
from collections import Counter
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from pydantic import ValidationError

from science_tutor.config import load_config
from science_tutor.data import DatasetValidationError, load_scienceqa


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
                    "accelerator": "not probed; training dependencies are not installed by default",
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

    print(config.model_dump_json(indent=2))
    return 0
