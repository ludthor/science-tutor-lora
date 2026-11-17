"""Protect record identity and reject malformed data before training selection."""

import json
from pathlib import Path

import pytest

from science_tutor.data import DatasetValidationError, load_scienceqa


def test_preserves_ids_order_splits_and_empty_reference_text(scienceqa_file: Path) -> None:
    records = load_scienceqa(scienceqa_file)

    assert list(records) == ["001", "17"]
    assert records["001"].split == "train"
    assert records["17"].split == "val"
    assert records["001"].hint == ""
    assert records["001"].solution == ""
    assert records["001"].image is None


def test_preserves_source_text_and_keeps_image_records(
    tmp_path: Path, scienceqa_record: dict[str, object]
) -> None:
    question = "  Which state is liquid?\n¿Agua?  "
    scienceqa_record.update(question=question, image="image.png", subject="language science")
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"42": scienceqa_record}), encoding="utf-8")

    record = load_scienceqa(path)["42"]

    assert record.question == question
    assert record.image == "image.png"
    assert record.subject == "language science"


def test_allows_omitted_optional_text(tmp_path: Path, scienceqa_record: dict[str, object]) -> None:
    for field in ("hint", "lecture", "solution"):
        del scienceqa_record[field]
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"42": scienceqa_record}), encoding="utf-8")

    record = load_scienceqa(path)["42"]

    assert record.hint == record.lecture == record.solution == ""


@pytest.mark.parametrize("choice_count", [2, 3, 4, 5])
def test_accepts_last_choice_for_different_choice_counts(
    tmp_path: Path, scienceqa_record: dict[str, object], choice_count: int
) -> None:
    scienceqa_record.update(
        choices=[f"Option {index}" for index in range(choice_count)], answer=choice_count - 1
    )
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"42": scienceqa_record}), encoding="utf-8")

    record = load_scienceqa(path)["42"]

    assert record.answer == choice_count - 1
    assert len(record.choices) == choice_count


def test_preserves_test_split(tmp_path: Path, scienceqa_record: dict[str, object]) -> None:
    scienceqa_record["split"] = "test"
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"42": scienceqa_record}), encoding="utf-8")

    assert load_scienceqa(path)["42"].split == "test"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question", "  \n"),
        ("choices", []),
        ("choices", ["Only one"]),
        ("choices", ["Water", " "]),
        ("choices", ["Water", 1]),
        ("choices", "Water, stone"),
        ("answer", -1),
        ("answer", 2),
        ("answer", True),
        ("answer", "0"),
        ("answer", 0.0),
        ("split", "validation"),
        ("subject", "unknown science"),
        ("task", "unknown task"),
        ("image", ""),
        ("image", 1),
        ("hint", None),
        ("solution", None),
        ("unexpected_field", "schema drift"),
    ],
)
def test_rejects_invalid_record_without_silently_dropping_it(
    tmp_path: Path, scienceqa_record: dict[str, object], field: str, value: object
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps({"good": scienceqa_record, "bad": {**scienceqa_record, field: value}}),
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="bad"):
        load_scienceqa(path)


@pytest.mark.parametrize("field", ["question", "choices", "answer", "image", "split"])
def test_rejects_missing_required_fields(
    tmp_path: Path, scienceqa_record: dict[str, object], field: str
) -> None:
    del scienceqa_record[field]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"42": scienceqa_record}), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match=field):
        load_scienceqa(path)


@pytest.mark.parametrize("source", [{}, [], None, {"42": None}, {"42": []}])
def test_rejects_invalid_root_or_record_shapes(tmp_path: Path, source: object) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(DatasetValidationError):
        load_scienceqa(path)


def test_rejects_blank_record_ids(tmp_path: Path, scienceqa_record: dict[str, object]) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({" ": scienceqa_record}), encoding="utf-8")

    with pytest.raises(DatasetValidationError):
        load_scienceqa(path)


@pytest.mark.parametrize(
    "contents",
    [
        b'{"42": {}, "42": {}}',
        b'{"42": {"answer": 0, "answer": 1}}',
    ],
)
def test_rejects_duplicate_ids_and_fields(tmp_path: Path, contents: bytes) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(contents)

    with pytest.raises(DatasetValidationError, match="Duplicate JSON key"):
        load_scienceqa(path)


@pytest.mark.parametrize("contents", [b'{"unfinished":', b"\xff"])
def test_rejects_unreadable_json(tmp_path: Path, contents: bytes) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(contents)

    with pytest.raises(DatasetValidationError, match="Invalid UTF-8 JSON"):
        load_scienceqa(path)


def test_missing_file_remains_a_file_access_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_scienceqa(tmp_path / "missing.json")
