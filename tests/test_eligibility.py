"""Keep training eligibility separate from source validity and split assignment."""

import json
from pathlib import Path

import pytest

from science_tutor.config import DatasetSettings
from science_tutor.data import ScienceQARecord
from science_tutor.eligibility import (
    POLICY_VERSION,
    assess_eligibility,
    summarize_eligibility,
    write_eligibility_report,
)


@pytest.fixture
def eligible_record(scienceqa_record: dict[str, object]) -> ScienceQARecord:
    return ScienceQARecord.model_validate(
        {**scienceqa_record, "solution": "Water is a liquid and takes the shape of its container."}
    )


def test_keeps_source_unchanged_and_preserves_all_ids_and_splits(
    eligible_record: ScienceQARecord,
) -> None:
    records = {
        "001": eligible_record,
        "17": eligible_record.model_copy(update={"split": "val", "image": "image.png"}),
        "008": eligible_record.model_copy(
            update={"split": "test", "question": "What does the picture show?"}
        ),
    }
    before = {key: row.model_dump() for key, row in records.items()}

    decisions = assess_eligibility(records)

    assert [(row.record_id, row.split, row.status) for row in decisions] == [
        ("001", "train", "eligible"),
        ("17", "val", "excluded"),
        ("008", "test", "review"),
    ]
    assert {key: row.model_dump() for key, row in records.items()} == before
    assert decisions[0].reasons == decisions[0].review_flags == ()


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"subject": "social science"}, "other_subject"),
        ({"image": "image.png"}, "image_present"),
        ({"solution": ""}, "missing_solution"),
        ({"solution": " \n\t"}, "missing_solution"),
    ],
)
def test_excludes_clear_eligibility_failures(
    eligible_record: ScienceQARecord, updates: dict[str, object], reason: str
) -> None:
    record = eligible_record.model_copy(update=updates)

    decision = assess_eligibility({"42": record})[0]

    assert decision.status == "excluded"
    assert decision.reasons == (reason,)
    assert decision.review_flags == ()


def test_records_overlapping_exclusion_reasons_before_visual_screening(
    eligible_record: ScienceQARecord,
) -> None:
    record = eligible_record.model_copy(
        update={
            "subject": "social science",
            "image": "image.png",
            "solution": "",
            "question": "Look at the picture.",
        }
    )

    decisions = assess_eligibility({"42": record})
    summary = summarize_eligibility(decisions)

    assert decisions[0].reasons == ("other_subject", "image_present", "missing_solution")
    assert decisions[0].review_flags == ()
    assert summary["total"] == summary["by_status"]["excluded"] == 1
    assert summary["by_reason"] == {
        "other_subject": 1,
        "image_present": 1,
        "missing_solution": 1,
    }


@pytest.mark.parametrize(
    "question",
    [
        "Use the diagram below.",
        "Look at each picture.",
        "What do these images show?",
        "Read the following table.",
        "Use the bar chart.",
        "A graph shows a rising line. What happens?",
        "What is shown above?",
        "Figure 2 contains several cells.",
        "Refer to Fig. 2.",
        "Figure: a leaf. Which part absorbs light?",
        '<IMG src="leaf.png"> Which structure is labeled?',
        "![a leaf](leaf.png) Which structure is labeled?",
    ],
)
def test_visual_references_are_withheld_for_review(
    eligible_record: ScienceQARecord, question: str
) -> None:
    record = eligible_record.model_copy(update={"question": question})

    decision = assess_eligibility({"42": record})[0]

    assert decision.status == "review"
    assert decision.reasons == ("possible_visual_reference",)
    assert decision.review_flags
    assert {flag.field for flag in decision.review_flags} == {"question"}


@pytest.mark.parametrize("field", ["question", "hint", "choices.1", "solution"])
def test_screens_each_input_and_reference_solution(
    eligible_record: ScienceQARecord, field: str
) -> None:
    text = "Look at the picture below."
    updates: dict[str, object] = (
        {"choices": ["Water", text]} if field == "choices.1" else {field: text}
    )

    decision = assess_eligibility({"42": eligible_record.model_copy(update=updates)})[0]

    assert decision.status == "review"
    assert {flag.field for flag in decision.review_flags} == {field}


@pytest.mark.parametrize(
    "question",
    [
        "How do you figure out density?",
        "What is a graph?",
        "Which material is used to make graph paper?",
        "What information can a map represent?",
    ],
)
def test_visual_vocabulary_alone_does_not_require_review(
    eligible_record: ScienceQARecord, question: str
) -> None:
    record = eligible_record.model_copy(update={"question": question})

    assert assess_eligibility({"42": record})[0].status == "eligible"


def test_unused_lecture_and_empty_hint_do_not_exclude_record(
    eligible_record: ScienceQARecord,
) -> None:
    record = eligible_record.model_copy(update={"lecture": "See the diagram.", "hint": ""})

    assert assess_eligibility({"42": record})[0].status == "eligible"


def test_summary_partitions_every_record_and_keeps_zero_counts(
    eligible_record: ScienceQARecord,
) -> None:
    decisions = assess_eligibility(
        {
            "001": eligible_record,
            "17": eligible_record.model_copy(update={"split": "val", "solution": ""}),
            "008": eligible_record.model_copy(
                update={"split": "test", "hint": "Read the chart below."}
            ),
        }
    )

    summary = summarize_eligibility(decisions)

    assert summary["total"] == 3
    assert summary["by_status"] == {"eligible": 1, "excluded": 1, "review": 1}
    assert summary["by_split"] == {
        "train": {"eligible": 1, "excluded": 0, "review": 0},
        "val": {"eligible": 0, "excluded": 1, "review": 0},
        "test": {"eligible": 0, "excluded": 0, "review": 1},
    }
    assert sum(summary["by_status"].values()) == summary["total"]
    assert sum(sum(counts.values()) for counts in summary["by_split"].values()) == summary["total"]


def test_empty_candidate_input_has_zero_summary() -> None:
    summary = summarize_eligibility(assess_eligibility({}))

    assert summary["total"] == 0
    assert summary["by_status"] == {"eligible": 0, "excluded": 0, "review": 0}
    assert summary["by_reason"] == {}


def test_report_is_reproducible_and_contains_provenance_without_source_text(
    tmp_path: Path, eligible_record: ScienceQARecord, source_settings: DatasetSettings
) -> None:
    path = tmp_path / "processed" / "eligibility.json"
    decisions = assess_eligibility({"001": eligible_record})

    summary = write_eligibility_report(path, decisions, source_settings)
    first = path.read_bytes()
    write_eligibility_report(path, decisions, source_settings)
    report = json.loads(first)

    assert path.read_bytes() == first
    assert report["summary"] == summary
    assert report["policy"]["version"] == POLICY_VERSION
    assert report["policy"]["subject"] == "natural science"
    assert report["policy"]["visual_reference_handling"] == "review"
    assert report["policy"]["visual_rules"]
    assert report["source"] == {
        "repository": source_settings.repository,
        "revision": source_settings.revision,
        "sha256": source_settings.sha256,
    }
    assert report["decisions"] == [
        {
            "record_id": "001",
            "split": "train",
            "status": "eligible",
            "reasons": [],
            "review_flags": [],
        }
    ]
    assert eligible_record.question not in first.decode()
    assert eligible_record.solution not in first.decode()
    assert list(path.parent.iterdir()) == [path]


def test_failed_report_write_preserves_existing_report_and_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    eligible_record: ScienceQARecord,
    source_settings: DatasetSettings,
) -> None:
    path = tmp_path / "eligibility.json"
    path.write_bytes(b"previous report")

    def fail_replace(*args: object, **kwargs: object) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="write failure"):
        write_eligibility_report(path, assess_eligibility({"42": eligible_record}), source_settings)

    assert path.read_bytes() == b"previous report"
    assert list(tmp_path.iterdir()) == [path]
