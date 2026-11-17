"""Verify paired evaluation and strict scoring with no model download."""

import pytest

from science_tutor.baseline import compare_prompts, parse_prediction
from science_tutor.prompts import PromptDataset, PromptMessage, build_prompt


@pytest.mark.parametrize(
    "output",
    [
        '```json\n{"answer":"A","explanation":"Reason"}\n```',
        '{"answer":"C","explanation":"Not an offered choice"}',
        '{"answer":"A","explanation":"   "}',
        '{"answer":"A","answer":"B","explanation":"Duplicate key"}',
        '{"answer":"A","explanation":"Reason","extra":true}',
        '{"answer":"A","explanation":"Reason"} trailing text',
        '{"answer":"a","explanation":"Lowercase"}',
        '{"answer":"A","explanation":42}',
        '[{"answer":"A","explanation":"Reason"}]',
    ],
)
def test_strict_parser_rejects_invalid_outputs(output: str) -> None:
    assert parse_prediction(output, {"A", "B"}) is None


def test_parser_accepts_requested_shape_and_trims_explanation() -> None:
    assert parse_prediction(' \n{"explanation":" Water flows. ","answer":"B"}', {"A", "B"}) == {
        "answer": "B",
        "explanation": "Water flows.",
    }


@pytest.fixture
def dataset() -> PromptDataset:
    return {
        "schema_version": 1,
        "format_version": "1",
        "experiment": "fixture",
        "source": {},
        "sampling_policy": {},
        "seed": 42,
        "limits": {"train": 2, "val": 2},
        "quality_exclusions": {},
        "train": [
            {
                "record_id": f"train-{index}",
                "prompt": build_prompt(f"Training question {index}", ["ice", "stone"]),
                "completion": [{"role": "assistant", "content": "One training answer"}],
            }
            for index in range(2)
        ],
        "validation": [
            {
                "record_id": f"val-{index}",
                "prompt": build_prompt(f"Validation question {index}", ["fog", "wood"]),
                "reference": {"answer": "A", "explanation": "Do not send this to the model"},
            }
            for index in range(2)
        ],
    }


def test_same_queries_and_denominator_for_both_prompting_conditions(dataset: PromptDataset) -> None:
    calls: list[list[list[PromptMessage]]] = []

    def generate(batch: list[list[PromptMessage]]) -> list[str]:
        calls.append(batch)
        return ['{"answer":"A","explanation":"It is fog."}', "bad JSON"]

    result = compare_prompts(dataset, generate, shots=1, batch_size=2)

    assert result["validation_ids"] == ["val-0", "val-1"]
    assert result["demonstration_ids"] == ["train-0"]
    assert [row[-1] for row in calls[0]] == [row[-1] for row in calls[1]]
    assert [msg["role"] for msg in calls[0][0]] == ["system", "user"]
    assert [msg["role"] for msg in calls[1][0]] == ["system", "user", "assistant", "user"]
    assert all(
        "Do not send this" not in message["content"]
        for call in calls
        for row in call
        for message in row
    )
    for mode in ("zero_shot", "few_shot"):
        assert result[mode]["total"] == 2
        assert result[mode]["accuracy"] == result[mode]["invalid_rate"] == 0.5
        assert result[mode]["predictions"][1]["parsed"] is None


def test_rejects_incomplete_generation_batch(dataset: PromptDataset) -> None:
    with pytest.raises(ValueError, match="one response per query"):
        compare_prompts(dataset, lambda _: [], limit=1)
