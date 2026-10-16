"""Exercise user-facing CLI behavior without models, data, or a GPU."""

import json
from pathlib import Path

import pytest

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
