"""Build chat inputs and supervised targets from the selected ScienceQA records."""

import json
from collections.abc import Mapping, Sequence
from string import ascii_uppercase
from typing import Literal, TypedDict

from science_tutor.config import ExperimentConfig
from science_tutor.data import ScienceQARecord
from science_tutor.sampling import build_sample_manifest

PROMPT_FORMAT_VERSION = "1"
_SYSTEM = (
    "You are a school science tutor. Choose one of the listed answer letters. "
    'Return only a JSON object with two string fields: "answer" (the letter) and '
    '"explanation" (a brief explanation of why that choice is correct).'
)


class PromptMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class ReferenceAnswer(TypedDict):
    answer: str
    explanation: str


class TrainingExample(TypedDict):
    record_id: str
    prompt: list[PromptMessage]
    completion: list[PromptMessage]


class ValidationExample(TypedDict):
    record_id: str
    prompt: list[PromptMessage]
    reference: ReferenceAnswer


class PromptDataset(TypedDict):
    schema_version: int
    format_version: str
    experiment: str
    source: dict[str, str]
    sampling_policy: dict[str, str]
    seed: int
    limits: dict[str, int]
    quality_exclusions: dict[str, str]
    train: list[TrainingExample]
    validation: list[ValidationExample]


def build_prompt(question: str, choices: Sequence[str], hint: str = "") -> list[PromptMessage]:
    """Render the allowed input fields; answers and annotations are not arguments."""
    if not 2 <= len(choices) <= len(ascii_uppercase):
        raise ValueError("Prompt format v1 requires 2 to 26 choices.")
    payload: dict[str, str | dict[str, str]] = {
        "question": question.strip(),
        "choices": {ascii_uppercase[index]: choice.strip() for index, choice in enumerate(choices)},
    }
    if hint.strip():
        payload["hint"] = hint.strip()
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _reference(record: ScienceQARecord) -> ReferenceAnswer:
    return {"answer": ascii_uppercase[record.answer], "explanation": record.solution.strip()}


def build_prompt_dataset(
    records: Mapping[str, ScienceQARecord], config: ExperimentConfig
) -> PromptDataset:
    """Use the verified source selection, exporting training and validation only."""
    manifest = build_sample_manifest(records, config)
    train: list[TrainingExample] = []
    for record_id in manifest["samples"]["train"]:
        record = records[record_id]
        train.append(
            {
                "record_id": record_id,
                "prompt": build_prompt(record.question, record.choices, record.hint),
                "completion": [
                    {
                        "role": "assistant",
                        "content": json.dumps(_reference(record), ensure_ascii=False),
                    }
                ],
            }
        )
    validation: list[ValidationExample] = []
    for record_id in manifest["samples"]["val"]:
        record = records[record_id]
        validation.append(
            {
                "record_id": record_id,
                "prompt": build_prompt(record.question, record.choices, record.hint),
                "reference": _reference(record),
            }
        )
    return {
        "schema_version": 1,
        "format_version": PROMPT_FORMAT_VERSION,
        "experiment": manifest["experiment"],
        "source": manifest["source"],
        "sampling_policy": manifest["policy"],
        "seed": manifest["seed"],
        "limits": manifest["limits"],
        "quality_exclusions": manifest["quality_exclusions"],
        "train": train,
        "validation": validation,
    }


def inference_messages(
    query: ValidationExample, demonstrations: Sequence[TrainingExample] = ()
) -> list[PromptMessage]:
    """Render a query with optional training demonstrations and no query reference."""
    system, user = query["prompt"]
    messages = [system.copy()]
    for demonstration in demonstrations:
        if demonstration["record_id"] == query["record_id"]:
            raise ValueError("A query cannot serve as its own demonstration.")
        messages.append(demonstration["prompt"][1].copy())
        messages.extend(message.copy() for message in demonstration["completion"])
    messages.append(user.copy())
    return messages
