"""Evaluate paired prompting baselines independently of the model runtime."""

import json
from collections.abc import Callable, Sequence
from typing import Literal, TypedDict

from science_tutor.prompts import PromptDataset, PromptMessage, ReferenceAnswer, inference_messages

BASELINE_VERSION = "1"
Mode = Literal["zero_shot", "few_shot"]
Generate = Callable[[list[list[PromptMessage]]], Sequence[str]]
Progress = Callable[[Mode, int, int], None]


class Prediction(TypedDict):
    record_id: str
    expected_answer: str
    output: str
    parsed: ReferenceAnswer | None
    correct: bool


class ModeResult(TypedDict):
    total: int
    correct: int
    invalid: int
    accuracy: float
    invalid_rate: float
    predictions: list[Prediction]


class Comparison(TypedDict):
    validation_ids: list[str]
    demonstration_ids: list[str]
    zero_shot: ModeResult
    few_shot: ModeResult


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON field.")
    return result


def parse_prediction(output: str, choices: set[str]) -> ReferenceAnswer | None:
    """Accept one JSON object with exactly the requested fields and a valid choice."""
    try:
        result = json.loads(output, object_pairs_hook=_unique_keys)
    except (ValueError, RecursionError):
        return None
    if not isinstance(result, dict) or set(result) != {"answer", "explanation"}:
        return None
    answer, explanation = result["answer"], result["explanation"]
    if not isinstance(answer, str) or answer not in choices:
        return None
    if not isinstance(explanation, str) or not explanation.strip():
        return None
    return {"answer": answer, "explanation": explanation.strip()}


def compare_prompts(
    dataset: PromptDataset,
    generate: Generate,
    *,
    shots: int = 2,
    limit: int | None = None,
    batch_size: int = 8,
    progress: Progress | None = None,
) -> Comparison:
    """Run both conditions on the same queries; demonstrations are first-N train rows."""
    if not 1 <= shots <= len(dataset["train"]):
        raise ValueError("Few-shot count must fit the available training rows and be positive.")
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("Batch size and validation limit must be positive.")
    queries = dataset["validation"][:limit]
    if not queries:
        raise ValueError("No validation examples to score.")
    demonstrations = dataset["train"][:shots]

    def evaluate(mode: Mode) -> ModeResult:
        predictions: list[Prediction] = []
        for start in range(0, len(queries), batch_size):
            batch = queries[start : start + batch_size]
            outputs = generate(
                [
                    inference_messages(query, demonstrations if mode == "few_shot" else ())
                    for query in batch
                ]
            )
            if len(outputs) != len(batch):
                raise ValueError("Generator must return one response per query.")
            for query, output in zip(batch, outputs, strict=True):
                choices = set(json.loads(query["prompt"][-1]["content"])["choices"])
                parsed = parse_prediction(output, choices)
                expected = query["reference"]["answer"]
                predictions.append(
                    {
                        "record_id": query["record_id"],
                        "expected_answer": expected,
                        "output": output,
                        "parsed": parsed,
                        "correct": parsed is not None and parsed["answer"] == expected,
                    }
                )
            if progress is not None:
                progress(mode, len(predictions), len(queries))
        total = len(predictions)
        correct = sum(row["correct"] for row in predictions)
        invalid = sum(row["parsed"] is None for row in predictions)
        return {
            "total": total,
            "correct": correct,
            "invalid": invalid,
            "accuracy": correct / total,
            "invalid_rate": invalid / total,
            "predictions": predictions,
        }

    return {
        "validation_ids": [query["record_id"] for query in queries],
        "demonstration_ids": [example["record_id"] for example in demonstrations],
        "zero_shot": evaluate("zero_shot"),
        "few_shot": evaluate("few_shot"),
    }
