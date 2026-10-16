"""Read and validate local ScienceQA source records without selecting training data."""

import json
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

NonBlankText = Annotated[str, Field(pattern=r"\S")]


class ScienceQARecord(BaseModel):
    """The upstream record structure, with optional reference text left unfiltered."""

    model_config = ConfigDict(strict=True, extra="forbid")

    question: NonBlankText
    choices: Annotated[list[NonBlankText], Field(min_length=2)]
    answer: Annotated[int, Field(ge=0)]
    hint: str = ""
    image: NonBlankText | None
    task: Literal["closed choice", "true-or false", "yes or no"]
    grade: NonBlankText
    subject: Literal["natural science", "language science", "social science"]
    topic: NonBlankText
    category: NonBlankText
    skill: NonBlankText
    lecture: str = ""
    solution: str = ""
    split: Literal["train", "val", "test"]

    @model_validator(mode="after")
    def check_answer_index(self) -> Self:
        """A nonnegative label must also point to an existing choice."""
        if self.answer >= len(self.choices):
            raise ValueError("answer must be a zero-based index into choices")
        return self


SourceRecords = Annotated[dict[NonBlankText, ScienceQARecord], Field(min_length=1)]

_SOURCE_ADAPTER: TypeAdapter[dict[str, ScienceQARecord]] = TypeAdapter(
    SourceRecords,
    config=ConfigDict(strict=True),
)


class DatasetValidationError(ValueError):
    """A source file cannot be interpreted as valid ScienceQA records."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate IDs and fields before the JSON parser overwrites them."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetValidationError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_scienceqa(path: Path) -> dict[str, ScienceQARecord]:
    """Load the whole local file, preserving IDs, order, source text, and split names.

    Invalid records fail the load. Empty or omitted hint/lecture/solution fields
    become empty strings; eligibility filtering is a separate operation. File
    access failures propagate as OSError. No files are downloaded or modified.
    """
    try:
        with path.open(encoding="utf-8") as stream:
            source = json.load(stream, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise DatasetValidationError(f"Invalid UTF-8 JSON: {exc}") from exc

    try:
        return _SOURCE_ADAPTER.validate_python(source)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False, include_input=False)
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
            for error in errors[:5]
        )
        raise DatasetValidationError(
            f"Invalid ScienceQA source ({exc.error_count()} errors; showing up to 5): {details}"
        ) from exc
