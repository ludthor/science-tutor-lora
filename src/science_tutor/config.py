"""Validate experiment settings before downloading data or allocating a GPU."""

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

PositiveInt = Annotated[int, Field(gt=0)]
CommitRevision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Settings(BaseModel):
    """Reject unknown settings and implicit type conversions."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelSettings(Settings):
    """Pin the starting checkpoint to an immutable Hugging Face revision."""

    model_id: Annotated[str, Field(min_length=1)]
    revision: CommitRevision


class DatasetSettings(Settings):
    """Identify the original ScienceQA source and bound the smoke-test workload."""

    repository: Literal["lupantech/ScienceQA"]
    revision: CommitRevision
    sha256: Sha256Digest
    subject: Literal["natural science"]
    max_train_examples: PositiveInt
    max_validation_examples: PositiveInt


class LoraSettings(Settings):
    """Describe the adapter independently of the model-loading backend."""

    rank: PositiveInt
    alpha: PositiveInt
    dropout: Annotated[float, Field(ge=0, lt=1)]


class TrainingSettings(Settings):
    """Keep the first run small enough to measure before expanding."""

    max_sequence_length: Annotated[int, Field(ge=64)]
    batch_size: PositiveInt
    gradient_accumulation_steps: PositiveInt
    learning_rate: Annotated[float, Field(gt=0, lt=1)]
    max_steps: PositiveInt


class ExperimentConfig(Settings):
    """A versioned plan shared by the CLI and the notebook."""

    schema_version: Literal[1]
    name: Annotated[str, Field(min_length=1)]
    seed: Annotated[int, Field(ge=0)]
    model: ModelSettings
    dataset: DatasetSettings
    lora: LoraSettings
    training: TrainingSettings


def load_config(path: Path) -> ExperimentConfig:
    """Load TOML without network calls or changes to the filesystem."""
    with path.open("rb") as stream:
        return ExperimentConfig.model_validate(tomllib.load(stream))
