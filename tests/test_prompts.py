"""Check the prompt/reference boundary and that only train/validation are exported."""

import json
from pathlib import Path

import pytest

from science_tutor.config import ExperimentConfig, load_config
from science_tutor.data import ScienceQARecord
from science_tutor.prompts import build_prompt, build_prompt_dataset, inference_messages
from science_tutor.sampling import build_sample_manifest


@pytest.fixture
def config() -> ExperimentConfig:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.toml")
    return config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"quality_exclusions": {}})}
    )


@pytest.fixture
def records(scienceqa_record: dict[str, object]) -> dict[str, ScienceQARecord]:
    return {
        f"{split}-id": ScienceQARecord.model_validate(
            {
                **scienceqa_record,
                "question": question,
                "split": split,
                "solution": f"Reference for {split} only.",
                "lecture": f"Do not expose the {split} lecture.",
            }
        )
        for split, question in {
            "train": "Which substance fills the container?",
            "val": "Which substance evaporates?",
            "test": "A unique reserved question.",
        }.items()
    }


def test_prompt_preserves_choice_order_and_structures_multiline_text() -> None:
    prompt = build_prompt('Why?\nUse "this".', [" Ice ", "Steam", "Water"], " At 20 °C. ")

    assert [message["role"] for message in prompt] == ["system", "user"]
    assert json.loads(prompt[1]["content"]) == {
        "question": 'Why?\nUse "this".',
        "choices": {"A": "Ice", "B": "Steam", "C": "Water"},
        "hint": "At 20 °C.",
    }
    assert "hint" not in json.loads(build_prompt("Why?", ["Ice", "Water"], " \n")[1]["content"])


@pytest.mark.parametrize("choices", [["Only one"], ["Choice"] * 27])
def test_unrepresentable_choice_counts_fail_clearly(choices: list[str]) -> None:
    with pytest.raises(ValueError, match="2 to 26 choices"):
        build_prompt("Which choice?", choices)


def test_targets_can_change_without_changing_inference_inputs(
    config: ExperimentConfig, records: dict[str, ScienceQARecord]
) -> None:
    original = build_prompt_dataset(records, config)
    train, validation = records["train-id"], records["val-id"]
    records["train-id"] = train.model_copy(update={"answer": 1, "solution": "New train target."})
    records["val-id"] = validation.model_copy(
        update={"answer": 1, "solution": "New validation target.", "lecture": "New lecture."}
    )

    changed = build_prompt_dataset(records, config)

    assert changed["train"][0]["prompt"] == original["train"][0]["prompt"]
    assert changed["validation"][0]["prompt"] == original["validation"][0]["prompt"]
    assert changed["train"][0]["completion"][0]["role"] == "assistant"
    assert json.loads(changed["train"][0]["completion"][0]["content"]) == {
        "answer": "B",
        "explanation": "New train target.",
    }
    assert changed["validation"][0]["reference"] == {
        "answer": "B",
        "explanation": "New validation target.",
    }


def test_dataset_matches_manifest_and_never_exports_test_rows(
    config: ExperimentConfig, records: dict[str, ScienceQARecord]
) -> None:
    manifest = build_sample_manifest(records, config)
    dataset = build_prompt_dataset(records, config)

    assert [row["record_id"] for row in dataset["train"]] == manifest["samples"]["train"]
    assert [row["record_id"] for row in dataset["validation"]] == manifest["samples"]["val"]
    assert dataset["source"] == manifest["source"]
    assert dataset["sampling_policy"] == manifest["policy"]
    assert dataset["seed"] == manifest["seed"]
    serialized = json.dumps(dataset)
    assert "test-id" not in serialized
    assert "A unique reserved question." not in serialized
    assert "Reference for test only." not in serialized
    assert "lecture" not in serialized


def test_few_shot_includes_only_training_targets_and_ends_with_validation_question(
    config: ExperimentConfig, records: dict[str, ScienceQARecord]
) -> None:
    dataset = build_prompt_dataset(records, config)
    query, demonstration = dataset["validation"][0], dataset["train"][0]

    assert inference_messages(query) == query["prompt"]
    messages = inference_messages(query, [demonstration])

    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert messages[1] == demonstration["prompt"][1]
    assert messages[2] == demonstration["completion"][0]
    assert messages[3] == query["prompt"][1]
    assert query["reference"]["explanation"] not in json.dumps(messages)
    messages[-1]["content"] = "A mutation to an assembled request."
    assert messages[-1] != query["prompt"][1]


def test_refuses_query_id_as_demonstration(
    config: ExperimentConfig, records: dict[str, ScienceQARecord]
) -> None:
    dataset = build_prompt_dataset(records, config)
    query = dataset["validation"][0]
    demonstration = dataset["train"][0].copy()
    demonstration["record_id"] = query["record_id"]

    with pytest.raises(ValueError, match="own demonstration"):
        inference_messages(query, [demonstration])
