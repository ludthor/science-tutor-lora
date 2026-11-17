"""Catch costly configuration mistakes before a cloud training session starts."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from science_tutor.config import load_config

SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.toml"


def test_smoke_config_pins_model_and_dataset() -> None:
    config = load_config(SMOKE_CONFIG)

    assert config.model.model_id == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    assert len(config.model.revision) == len(config.dataset.revision) == 40
    assert len(config.dataset.sha256) == 64
    assert config.dataset.max_train_examples == 500
    assert config.training.max_steps == 20


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("seed = 42", "seed = -1"),
        ("seed = 42", 'seed = "42"'),
        ("seed = 42", "seed = true"),
        ("schema_version = 1", "schema_version = 2"),
        ("rank = 8", "rank = 0"),
        ("dropout = 0.05", "dropout = 1.0"),
        ("dropout = 0.05", "dropout = -0.1"),
        ("learning_rate = 0.0002", "learning_rate = 0.0"),
        ("batch_size = 1", "batch_size = 0"),
        ("max_steps = 20", "max_steps = -1"),
        ("max_sequence_length = 512", "max_sequence_length = 32"),
        ("max_train_examples = 500", "max_train_examples = 0"),
        ("max_validation_examples = 100", "max_validation_examples = 0"),
        ("rank = 8", "rnak = 8"),
        ('name = "scienceqa-smoke"', 'name = "scienceqa-smoke"\nunknown = 1'),
        ("31b70e2e869a7173562077fd711b654946d38674", "main"),
        ("2cbf8318e07b9ece895bb2ae605e71e38d623264", "main"),
        ("4d9b598da966d9736dd79e430a97da861a2216aeb7483a5092350e823ab20ce7", "invalid"),
        ("4d9b598da966d9736dd79e430a97da861a2216aeb7483a5092350e823ab20ce7", "a" * 63),
    ],
)
def test_rejects_invalid_settings(tmp_path: Path, original: str, replacement: str) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(SMOKE_CONFIG.read_text().replace(original, replacement), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)
