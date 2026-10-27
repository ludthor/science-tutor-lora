"""Guard leakage detection against reordered choices, shared stems, and unstable ordering."""

import json
from pathlib import Path

import pytest

from science_tutor.config import DatasetSettings
from science_tutor.data import ScienceQARecord
from science_tutor.duplicates import audit_duplicates, normalize_text, write_duplicate_report


@pytest.fixture
def candidate(scienceqa_record: dict[str, object]) -> ScienceQARecord:
    return ScienceQARecord.model_validate({**scienceqa_record, "solution": "Water is a liquid."})


def test_normalization_preserves_numbers_operators_and_superscripts() -> None:
    assert normalize_text("  CAFE\u0301\t has\nwater.  ") == "café has water."
    assert normalize_text("2 + 3") != normalize_text("2 - 3")
    assert normalize_text("x²") != normalize_text("x2")
    assert normalize_text("3 cm") != normalize_text("4 cm")


def test_normalized_input_matches_across_splits_and_shuffled_choices(
    candidate: ScienceQARecord,
) -> None:
    other = candidate.model_copy(
        update={
            "question": "  WHICH example is a LIQUID?\n",
            "choices": ["A stone", "Water"],
            "answer": 1,
            "split": "test",
            "solution": "A differently worded explanation.",
        }
    )

    audit = audit_duplicates({"001": candidate, "17": other})

    assert len(audit.exact_groups) == 1
    group = audit.exact_groups[0]
    assert [(member.record_id, member.split) for member in group.members] == [
        ("001", "train"),
        ("17", "test"),
    ]
    assert group.cross_split is True
    assert group.answer_conflict is False
    assert audit.near_pairs == ()


@pytest.mark.parametrize(
    "updates",
    [
        {"hint": "There is ice in the container."},
        {"choices": ["A wooden spoon", "An ice cube"]},
        {"choices": ["Water", "A stone", "A stone"]},
    ],
)
def test_shared_stem_with_different_context_is_not_an_exact_duplicate(
    candidate: ScienceQARecord, updates: dict[str, object]
) -> None:
    audit = audit_duplicates({"a": candidate, "b": candidate.model_copy(update=updates)})

    assert audit.exact_groups == ()
    assert audit.near_pairs == ()


def test_same_split_matches_are_reported_and_conflicting_labels_remain_in_group(
    candidate: ScienceQARecord,
) -> None:
    audit = audit_duplicates({"a": candidate, "b": candidate.model_copy(update={"answer": 1})})

    group = audit.exact_groups[0]
    assert group.cross_split is False
    assert group.answer_conflict is True
    assert audit.summary()["exact_answer_conflict_groups"] == 1


@pytest.mark.parametrize(("word_count", "expected_pairs"), [(19, 1), (18, 0)])
def test_near_threshold_boundary_keeps_operator_difference_for_review(
    candidate: ScienceQARecord, word_count: int, expected_pairs: int
) -> None:
    prefix = " ".join(f"word{index}" for index in range(word_count))
    audit = audit_duplicates(
        {
            "a": candidate.model_copy(update={"question": prefix + " +"}),
            "b": candidate.model_copy(update={"question": prefix + " -", "split": "val"}),
        }
    )

    assert audit.exact_groups == ()
    assert len(audit.near_pairs) == expected_pairs
    if expected_pairs:
        assert audit.near_pairs[0].similarity == 0.9
        assert audit.near_pairs[0].cross_split is True


@pytest.mark.parametrize("context", [{"hint": "A different context."}, {"choices": ["Yes", "No"]}])
def test_near_screening_requires_matching_hint_and_choices(
    candidate: ScienceQARecord, context: dict[str, object]
) -> None:
    prefix = " ".join(f"word{index}" for index in range(25))
    audit = audit_duplicates(
        {
            "a": candidate.model_copy(update={"question": prefix + " +"}),
            "b": candidate.model_copy(update={"question": prefix + " -", **context}),
        }
    )

    assert audit.exact_groups == audit.near_pairs == ()


def test_near_pairs_do_not_transitively_merge_distinct_questions(
    candidate: ScienceQARecord,
) -> None:
    first = [f"word{index}" for index in range(45)]
    second = first.copy()
    second[1] = "changed"
    third = second.copy()
    third[40] = "different"
    audit = audit_duplicates(
        {
            "a": candidate.model_copy(update={"question": " ".join(first)}),
            "b": candidate.model_copy(update={"question": " ".join(second)}),
            "c": candidate.model_copy(update={"question": " ".join(third)}),
        }
    )

    assert {(pair.left.record_id, pair.right.record_id) for pair in audit.near_pairs} == {
        ("a", "b"),
        ("b", "c"),
    }


def test_audit_is_order_independent_and_does_not_mutate_records(
    candidate: ScienceQARecord,
) -> None:
    prefix = " ".join(f"word{index}" for index in range(25))
    records = {
        "008": candidate.model_copy(update={"question": prefix + " +"}),
        "001": candidate.model_copy(update={"question": prefix + " +", "split": "test"}),
        "17": candidate.model_copy(update={"question": prefix + " -", "split": "val"}),
    }
    before = {key: row.model_dump() for key, row in records.items()}

    first = audit_duplicates(records)
    second = audit_duplicates(dict(reversed(list(records.items()))))

    assert first == second
    assert {key: row.model_dump() for key, row in records.items()} == before
    assert first.summary() == {
        "candidate_records": 3,
        "exact_groups": 1,
        "exact_cross_split_groups": 1,
        "exact_records": 2,
        "exact_answer_conflict_groups": 0,
        "near_pairs": 2,
        "near_cross_split_pairs": 2,
    }


def test_empty_pool_and_single_token_questions_are_supported(candidate: ScienceQARecord) -> None:
    assert all(value == 0 for value in audit_duplicates({}).summary().values())
    audit = audit_duplicates(
        {
            "a": candidate.model_copy(update={"question": "Water"}),
            "b": candidate.model_copy(update={"question": "Ice"}),
        }
    )
    assert audit.exact_groups == audit.near_pairs == ()


def test_report_recomputes_eligibility_and_records_no_source_text(
    tmp_path: Path, candidate: ScienceQARecord, source_settings: DatasetSettings
) -> None:
    path = tmp_path / "processed" / "duplicates.json"
    records = {
        "001": candidate,
        "17": candidate.model_copy(update={"split": "test"}),
        "image": candidate.model_copy(update={"image": "source.png"}),
        "review": candidate.model_copy(update={"solution": "Look at the picture."}),
    }

    summary = write_duplicate_report(path, records, source_settings)
    original = path.read_bytes()
    write_duplicate_report(path, dict(reversed(list(records.items()))), source_settings)
    report = json.loads(path.read_text())

    assert path.read_bytes() == original
    assert report["source"]["sha256"] == source_settings.sha256
    assert report["candidate_ids"] == ["001", "17"]
    assert report["summary"] == summary
    assert summary["candidate_records"] == 2
    assert summary["exact_cross_split_groups"] == 1
    assert "audit_only" in report["policy"]["action"]
    assert candidate.question not in path.read_text()
    assert candidate.solution not in path.read_text()
    assert list(path.parent.iterdir()) == [path]
