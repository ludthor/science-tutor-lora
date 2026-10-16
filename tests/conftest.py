"""Small invented source examples; tests do not download or redistribute ScienceQA."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def scienceqa_record() -> dict[str, object]:
    """A structurally valid synthetic record with optional reference text empty."""
    return {
        "question": "Which example is a liquid?",
        "choices": ["Water", "A stone"],
        "answer": 0,
        "hint": "",
        "image": None,
        "task": "closed choice",
        "grade": "grade2",
        "subject": "natural science",
        "topic": "matter",
        "category": "States of matter",
        "skill": "Identify liquids",
        "lecture": "",
        "solution": "",
        "split": "train",
    }


@pytest.fixture
def scienceqa_file(tmp_path: Path, scienceqa_record: dict[str, object]) -> Path:
    path = tmp_path / "problems.json"
    path.write_text(
        json.dumps({"001": scienceqa_record, "17": {**scienceqa_record, "split": "val"}}),
        encoding="utf-8",
    )
    return path
