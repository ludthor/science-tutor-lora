# Science Tutor LoRA

A Python learning project exploring whether a LoRA adapter can improve a small
language model's answers and explanations for school science questions.

**Research question:** Can a small LoRA adapter improve science answer accuracy
and explanation quality compared with zero-shot and few-shot prompting, at a
measured training cost?

The first experiment uses SmolLM2-1.7B-Instruct and a text-only, natural-science
subset of ScienceQA. Given a question and its answer choices, the intended model
output is a selected answer and a short explanation.

**Status:** project foundation. Configuration validation, runtime inspection,
tests, CI, and a Colab setup notebook are available. Dataset preparation,
training, inference, and evaluation are planned. No model has been trained and
no performance results are claimed.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/ludthor/science-tutor-lora.git
cd science-tutor-lora
git switch develop
uv sync --locked
uv run science-tutor doctor
uv run science-tutor validate configs/smoke.toml
```

Python 3.12 is the local default; Python 3.11–3.13 are supported. Setup installs
the small application and development tools, without model weights, datasets,
PyTorch, or CUDA packages. `validate` prints the normalized configuration as JSON
and returns exit code 2 on invalid input. It checks the configuration schema;
it does not check remote availability or run training.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv build
```

The CLI is also available as `uv run python -m science_tutor`. Code lives under
`src/`; the notebook calls the same package. Configuration rejects unknown fields,
invalid ranges, and mutable model/data revisions before compute is allocated.
Commit `uv.lock` when dependencies change.

## Cloud and local execution

Use [the setup notebook](notebooks/01_colab_setup.ipynb) in Google Colab to check
the runtime and validate the first experiment plan. It does not start training.
Record the resolved project commit printed by the notebook; use that exact SHA
to repeat a setup. Colab GPU resources and runtime duration are not guaranteed.

The primary training path will use PyTorch, Transformers, and PEFT on a single
NVIDIA GPU. The first run is bounded to 500 training examples, 100 validation
examples, and 20 optimizer steps. These are proposed limits, not measurements.

Local development targets a MacBook Air M4 with 24 GB unified memory. Optional
MLX training is deferred until the CUDA experiment works. Backend-specific model
loading will stay separate from dataset preparation and evaluation.

## Experiment milestones

1. **Foundation:** package, configuration, CLI, tests, CI, Gitflow, setup notebook.
2. **Data audit:** pinned download, text-only filtering, split checks, duplicate
   analysis, and deterministic sampling with an audit report.
3. **Baselines:** zero-shot and few-shot predictions on a fixed validation set.
4. **LoRA smoke test:** train an adapter, save recoverable checkpoints, reload it,
   and measure runtime and memory. Consider QLoRA if memory requires it.
5. **Evaluation:** select settings using validation, run the held-out test once,
   review explanations, and write a results report with failures and limitations.
6. **Demo and release:** a small interface and a documented `v0.1.0` release.

See the [project brief](docs/project-brief.md), [data notes](docs/data.md),
[technical decisions](docs/decisions.md), and [contribution guide](CONTRIBUTING.md).

## Licensing

Original project code is MIT licensed. ScienceQA data has its own
[CC BY-NC-SA 4.0 license](https://github.com/lupantech/ScienceQA/blob/main/LICENSE-DATA).
The code license does not relicense third-party data or model artifacts.
SmolLM2-1.7B-Instruct is distributed under Apache 2.0. Preserve upstream
attributions and record applicable terms before publishing any trained artifacts.
