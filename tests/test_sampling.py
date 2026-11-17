"""Exercise quarantine decisions and reproducible split-preserving samples."""

from pathlib import Path

import pytest

from science_tutor.config import ExperimentConfig, load_config
from science_tutor.data import ScienceQARecord, Split
from science_tutor.duplicates import audit_duplicates
from science_tutor.sampling import build_sample_manifest


@pytest.fixture
def config() -> ExperimentConfig:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.toml")
    return config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"quality_exclusions": {}})}
    )


@pytest.fixture
def records(scienceqa_record: dict[str, object]) -> dict[str, ScienceQARecord]:
    questions: dict[Split, str] = {
        "train": "Which liquid fills the vessel?",
        "val": "What material evaporates in warm conditions?",
        "test": "Which substance flows freely at room temperature?",
    }
    return {
        split: ScienceQARecord.model_validate(
            {
                **scienceqa_record,
                "question": question,
                "split": split,
                "solution": "Water is liquid.",
            }
        )
        for split, question in questions.items()
    }


def test_conflicting_exact_members_are_all_quarantined(
    records: dict[str, ScienceQARecord], config: ExperimentConfig
) -> None:
    ambiguous = records["train"].model_copy(update={"question": "What is in the second container?"})
    records["conflict-train"] = ambiguous
    records["conflict-test"] = ambiguous.model_copy(update={"split": "test", "answer": 1})

    manifest = build_sample_manifest(records, config)

    assert manifest["samples"] == {"train": ["train"], "val": ["val"]}
    assert manifest["reserved_test_ids"] == ["test"]
    assert manifest["quarantine"] == [
        {
            "record_id": record_id,
            "split": records[record_id].split,
            "reasons": ["conflicting_exact_answers"],
            "retained_record_id": None,
        }
        for record_id in ("conflict-test", "conflict-train")
    ]


@pytest.mark.parametrize(("splits", "winner"), [("train,val,test", "test"), ("train,val", "val")])
def test_agreed_exact_priority_protects_held_out_copy_without_moving_records(
    records: dict[str, ScienceQARecord], config: ExperimentConfig, splits: str, winner: str
) -> None:
    repeated = records["train"].model_copy(update={"question": "What is in the first container?"})
    for split in splits.split(","):
        records[f"a-{split}"] = repeated.model_copy(update={"split": split})
    records[f"z-{winner}"] = repeated.model_copy(
        update={"split": winner, "choices": ["A stone", "Water"], "answer": 1}
    )

    manifest = build_sample_manifest(records, config)
    withheld = {entry["record_id"]: entry for entry in manifest["quarantine"]}
    selected = [
        *manifest["samples"]["train"],
        *manifest["samples"]["val"],
        *manifest["reserved_test_ids"],
    ]

    assert set(withheld) == {f"a-{split}" for split in splits.split(",") if split != winner} | {
        f"z-{winner}"
    }
    assert all(entry["retained_record_id"] == f"a-{winner}" for entry in withheld.values())
    assert all(entry["reasons"] == ["redundant_exact"] for entry in withheld.values())
    assert f"a-{winner}" in selected
    assert not audit_duplicates(
        {record_id: records[record_id] for record_id in selected}
    ).exact_groups
    assert all(
        records[record_id].split == split
        for split, ids in manifest["samples"].items()
        for record_id in ids
    )
    assert all(records[record_id].split == "test" for record_id in manifest["reserved_test_ids"])


def test_near_match_withholds_endpoints_and_exact_copies(
    records: dict[str, ScienceQARecord], config: ExperimentConfig
) -> None:
    stem = " ".join(f"token{index}" for index in range(30))
    near = records["train"].model_copy(update={"question": stem + " +"})
    records.update(
        {
            "near-train": near,
            "near-val": near.model_copy(update={"split": "val"}),
            "near-test": near.model_copy(update={"split": "test", "question": stem + " -"}),
        }
    )

    manifest = build_sample_manifest(records, config)

    assert {entry["record_id"] for entry in manifest["quarantine"]} == {
        "near-train",
        "near-val",
        "near-test",
    }
    assert all(entry["reasons"] == ["near_duplicate_review"] for entry in manifest["quarantine"])
    assert all(entry["retained_record_id"] is None for entry in manifest["quarantine"])


def test_seed_changes_samples_without_changing_split_membership_or_reserve(
    records: dict[str, ScienceQARecord], config: ExperimentConfig
) -> None:
    dataset = config.dataset.model_copy(
        update={"max_train_examples": 4, "max_validation_examples": 3}
    )
    config = config.model_copy(update={"dataset": dataset})
    for index in range(15):
        for split in ("train", "val"):
            records[f"{split}-{index:02}"] = records[split].model_copy(
                update={"question": f"Which liquid belongs to {split} experiment {index}?"}
            )

    first = build_sample_manifest(records, config)
    reorder = dict(reversed(list(records.items())))
    second = build_sample_manifest(reorder, config)
    different_seed = build_sample_manifest(records, config.model_copy(update={"seed": 43}))

    assert first == second
    assert first["samples"] != different_seed["samples"]
    assert len(first["samples"]["train"]) == 4
    assert len(first["samples"]["val"]) == 3
    assert first["reserved_test_ids"] == different_seed["reserved_test_ids"] == ["test"]
    assert first["by_split"]["test"]["sampled"] == 0
    assert not (set(first["samples"]["train"]) & set(first["samples"]["val"]))


def test_ineligible_records_never_enter_samples_or_reserve(
    records: dict[str, ScienceQARecord], config: ExperimentConfig
) -> None:
    records["with-image"] = records["train"].model_copy(update={"image": "figure.png"})
    records["needs-review"] = records["test"].model_copy(update={"hint": "See the diagram."})

    manifest = build_sample_manifest(records, config)

    assert manifest["samples"] == {"train": ["train"], "val": ["val"]}
    assert manifest["reserved_test_ids"] == ["test"]
    assert manifest["quarantine"] == []
    assert manifest["by_split"]["train"] == {
        "eligible": 1,
        "quarantined": 0,
        "quality_excluded": 0,
        "available": 1,
        "sampled": 1,
        "reserved": 0,
    }


@pytest.mark.parametrize("split", ["train", "val"])
def test_empty_required_split_fails_instead_of_borrowing_from_another_split(
    records: dict[str, ScienceQARecord], config: ExperimentConfig, split: str
) -> None:
    records[split] = records[split].model_copy(update={"solution": ""})

    with pytest.raises(ValueError, match=f"No usable {split} records"):
        build_sample_manifest(records, config)


def test_documented_exclusion_is_backfilled_without_changing_test_reserve(
    records: dict[str, ScienceQARecord], config: ExperimentConfig
) -> None:
    records["backfill"] = records["train"].model_copy(update={"question": "Why does ice melt?"})
    config = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(
                update={"quality_exclusions": {"train": "Ambiguous"}}
            )
        }
    )

    manifest = build_sample_manifest(records, config)

    assert manifest["samples"]["train"] == ["backfill"]
    assert manifest["by_split"]["train"]["quality_excluded"] == 1
    assert manifest["by_split"]["train"]["quarantined"] == 0
    assert manifest["reserved_test_ids"] == ["test"]
    assert manifest["quality_exclusions"] == {"train": "Ambiguous"}


@pytest.mark.parametrize("record_id", ["missing", "test"])
def test_quality_exclusions_cannot_silently_change_test_or_reference_missing_ids(
    records: dict[str, ScienceQARecord], config: ExperimentConfig, record_id: str
) -> None:
    settings = config.dataset.model_copy(update={"quality_exclusions": {record_id: "Flagged"}})

    with pytest.raises(ValueError, match="usable train/val record"):
        build_sample_manifest(records, config.model_copy(update={"dataset": settings}))
