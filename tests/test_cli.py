"""Exercise user-facing CLI behavior without models, data, or a GPU."""

import hashlib
import json
from pathlib import Path
from urllib.error import URLError

import pytest

from science_tutor import cli, download
from science_tutor.cli import main

SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.toml"


def test_validate_prints_normalized_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(SMOKE_CONFIG)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["name"] == "scienceqa-smoke"
    assert captured.err == ""


@pytest.mark.parametrize("contents", [None, "[broken", 'seed = "incorrect"', "\xff"])
def test_invalid_file_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], contents: str | None
) -> None:
    path = tmp_path / "invalid.toml"
    if contents is not None:
        path.write_bytes(contents.encode("latin-1"))

    assert main(["validate", str(path)]) == 2
    captured = capsys.readouterr()
    assert "Invalid experiment configuration" in captured.err
    assert captured.out == ""


def test_doctor_prints_runtime_metadata(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor"]) == 0
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["python"]
    assert metadata["system"]


def test_inspect_data_prints_counts(
    scienceqa_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["inspect-data", str(scienceqa_file)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"records": 2, "splits": {"train": 1, "val": 1}}
    assert captured.err == ""


@pytest.mark.parametrize("contents", [None, b"{", b"[]", b"\xff"])
def test_inspect_data_reports_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], contents: bytes | None
) -> None:
    path = tmp_path / "invalid.json"
    if contents is not None:
        path.write_bytes(contents)

    assert main(["inspect-data", str(path)]) == 2
    captured = capsys.readouterr()
    assert "Invalid dataset" in captured.err
    assert captured.out == ""


def test_fetch_data_reuses_verified_cache(
    tmp_path: Path, source_payload: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "experiment.toml"
    config.write_text(
        SMOKE_CONFIG.read_text().replace(
            "4d9b598da966d9736dd79e430a97da861a2216aeb7483a5092350e823ab20ce7",
            hashlib.sha256(source_payload).hexdigest(),
        ),
        encoding="utf-8",
    )
    directory = tmp_path / "scienceqa" / "2cbf8318e07b9ece895bb2ae605e71e38d623264"
    directory.mkdir(parents=True)
    source = directory / "problems.json"
    source.write_bytes(source_payload)

    assert main(["fetch-data", str(config), "--data-dir", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["cached"] is True
    assert result["path"] == str(source)
    assert result["bytes"] == len(source_payload)
    assert captured.err == ""


def test_fetch_data_rejects_corrupt_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "scienceqa" / "2cbf8318e07b9ece895bb2ae605e71e38d623264"
    directory.mkdir(parents=True)
    (directory / "problems.json").write_bytes(b"corrupt")

    assert main(["fetch-data", str(SMOKE_CONFIG), "--data-dir", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert "Cannot fetch dataset" in captured.err
    assert "SHA-256 mismatch" in captured.err
    assert captured.out == ""


def test_fetch_data_rejects_invalid_config_before_network(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["fetch-data", str(tmp_path / "missing.toml")]) == 2
    assert "Invalid experiment configuration" in capsys.readouterr().err


def test_fetch_data_reports_network_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_download(*args: object, **kwargs: object) -> None:
        raise URLError("offline")

    monkeypatch.setattr(download, "urlopen", fail_download)

    assert main(["fetch-data", str(SMOKE_CONFIG), "--data-dir", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert "Cannot fetch dataset" in captured.err
    assert "offline" in captured.err
    assert captured.out == ""


def _cache_source(tmp_path: Path, source_payload: bytes) -> tuple[Path, Path]:
    config = tmp_path / "experiment.toml"
    before, _, remainder = SMOKE_CONFIG.read_text().partition("[dataset.quality_exclusions]")
    _, _, after = remainder.partition("[lora]")
    config.write_text(
        (before + "[lora]" + after).replace(
            "4d9b598da966d9736dd79e430a97da861a2216aeb7483a5092350e823ab20ce7",
            hashlib.sha256(source_payload).hexdigest(),
        ),
        encoding="utf-8",
    )
    directory = tmp_path / "scienceqa" / "2cbf8318e07b9ece895bb2ae605e71e38d623264"
    directory.mkdir(parents=True)
    source = directory / "problems.json"
    source.write_bytes(source_payload)
    return config, source


@pytest.fixture
def cached_source(tmp_path: Path, source_payload: bytes) -> tuple[Path, Path]:
    return _cache_source(tmp_path, source_payload)


def test_filter_data_writes_audit_from_verified_source(
    tmp_path: Path, cached_source: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    config, source = cached_source
    before = source.read_bytes()
    output = tmp_path / "processed" / "eligibility.json"

    assert (
        main(["filter-data", str(config), "--data-dir", str(tmp_path), "--output", str(output)])
        == 0
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    report = json.loads(output.read_text())
    assert summary["by_status"] == {"eligible": 0, "excluded": 1, "review": 0}
    assert summary["report"] == str(output)
    assert report["decisions"][0]["reasons"] == ["missing_solution"]
    assert source.read_bytes() == before
    assert captured.err == ""


@pytest.mark.parametrize(
    "command", ["filter-data", "audit-duplicates", "build-samples", "build-prompts"]
)
@pytest.mark.parametrize("target", ["source", "config", "provenance"])
def test_audit_commands_cannot_overwrite_their_inputs(
    tmp_path: Path,
    cached_source: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    target: str,
    command: str,
) -> None:
    config, source = cached_source
    output = {"source": source, "config": config, "provenance": source.parent / "source.json"}[
        target
    ]
    before = source.read_bytes(), config.read_bytes()

    assert main([command, str(config), "--data-dir", str(tmp_path), "--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert "must not overwrite" in captured.err
    assert captured.out == ""
    assert (source.read_bytes(), config.read_bytes()) == before
    assert json.loads((source.parent / "source.json").read_text())["revision"]


@pytest.mark.parametrize(
    "command", ["filter-data", "audit-duplicates", "build-samples", "build-prompts"]
)
def test_audit_commands_reject_corruption_before_writing_report(
    tmp_path: Path,
    cached_source: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    config, source = cached_source
    source.write_bytes(b"corrupt")
    output = tmp_path / "eligibility.json"

    assert main([command, str(config), "--data-dir", str(tmp_path), "--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert "SHA-256 mismatch" in captured.err
    assert captured.out == ""
    assert not output.exists()


def test_audit_duplicates_reports_empty_eligible_pool(
    tmp_path: Path, cached_source: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    config, source = cached_source
    before = source.read_bytes()
    output = tmp_path / "duplicates.json"

    assert (
        main(
            ["audit-duplicates", str(config), "--data-dir", str(tmp_path), "--output", str(output)]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["candidate_records"] == 0
    assert json.loads(output.read_text())["exact_groups"] == []
    assert captured.err == ""
    assert source.read_bytes() == before


def test_build_samples_is_repeatable_and_emits_only_ids(
    tmp_path: Path, scienceqa_record: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    source_rows = {
        split: {
            **scienceqa_record,
            "split": split,
            "question": question,
            "solution": "Water flows.",
        }
        for split, question in {
            "train": "What is in the glass?",
            "val": "Which substance boils?",
            "test": "Which liquid takes the shape of its container?",
        }.items()
    }
    config, source = _cache_source(tmp_path, json.dumps(source_rows).encode())
    before = source.read_bytes()
    output = tmp_path / "processed" / "samples.json"
    args = ["build-samples", str(config), "--data-dir", str(tmp_path), "--output", str(output)]

    assert main(args) == 0
    first = capsys.readouterr()
    manifest_bytes = output.read_bytes()
    assert main(args) == 0
    assert capsys.readouterr() == first
    assert output.read_bytes() == manifest_bytes
    manifest = json.loads(manifest_bytes)
    assert manifest["samples"] == {"train": ["train"], "val": ["val"]}
    assert manifest["reserved_test_ids"] == ["test"]
    assert json.loads(first.out)["by_split"]["test"]["sampled"] == 0
    assert "Water flows." not in manifest_bytes.decode()
    assert "What is in the glass?" not in manifest_bytes.decode()
    assert first.err == ""
    assert source.read_bytes() == before


def test_build_samples_leaves_existing_manifest_when_a_split_is_empty(
    tmp_path: Path, cached_source: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    config, _ = cached_source
    output = tmp_path / "samples.json"
    output.write_text("previous manifest")
    args = ["build-samples", str(config), "--data-dir", str(tmp_path), "--output", str(output)]

    assert main(args) == 2
    assert "No usable train records" in capsys.readouterr().err
    assert output.read_text() == "previous manifest"


def test_build_prompts_writes_split_specific_rows(
    tmp_path: Path, scienceqa_record: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    source_rows = {
        split: {
            **scienceqa_record,
            "split": split,
            "question": question,
            "solution": "Water flows.",
        }
        for split, question in {
            "train": "What is in the glass?",
            "val": "Which substance boils?",
            "test": "Which liquid takes the shape of its container?",
        }.items()
    }
    config, _ = _cache_source(tmp_path, json.dumps(source_rows).encode())
    output = tmp_path / "prompts.json"
    args = ["build-prompts", str(config), "--data-dir", str(tmp_path), "--output", str(output)]

    assert main(args) == 0
    stdout, stderr = capsys.readouterr()
    dataset_text = output.read_text()
    dataset = json.loads(dataset_text)
    assert json.loads(stdout) == {
        "output": str(output),
        "format_version": "1",
        "train": 1,
        "validation": 1,
    }
    assert stderr == ""
    assert "completion" in dataset["train"][0]
    assert "completion" not in dataset["validation"][0]
    assert dataset["validation"][0]["reference"]["answer"] == "A"
    assert "Which liquid takes the shape of its container?" not in dataset_text
    assert main(args) == 0
    assert output.read_text() == dataset_text


def test_run_baselines_writes_scoring_and_reproducibility_metadata(
    tmp_path: Path,
    scienceqa_record: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = {
        split: {
            **scienceqa_record,
            "split": split,
            "question": f"What is in the {split} container?",
            "solution": f"Reference for {split}.",
        }
        for split in ("train", "val", "test")
    }
    config, _ = _cache_source(tmp_path, json.dumps(rows).encode())
    path = tmp_path / "baselines.json"
    monkeypatch.setattr(
        cli,
        "load_generator",
        lambda *_: (
            lambda batch: ['{"answer":"A","explanation":"Water."}'] * len(batch),
            {"device": "mock"},
        ),
    )

    assert (
        main(
            [
                "run-baselines",
                str(config),
                "--data-dir",
                str(tmp_path),
                "--output",
                str(path),
                "--shots",
                "1",
            ]
        )
        == 0
    )
    stdout, stderr = capsys.readouterr()
    report = json.loads(path.read_text())
    assert report["validation_ids"] == ["val"]
    assert report["demonstration_ids"] == ["train"]
    assert report["runtime"]["device"] == "mock"
    assert len(report["prompt_dataset_sha256"]) == 64
    assert report["model"]["revision"]
    assert report["zero_shot"]["predictions"][0]["correct"]
    assert json.loads(stdout)["few_shot"]["accuracy"] == 1.0
    assert "few_shot: 1/1" in stderr
