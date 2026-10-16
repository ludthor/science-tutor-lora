# Contributing

Use small, reviewable changes that can run without a GPU whenever possible.

## Gitflow

- `main`: released work. The initial repository seed precedes the first release.
- `develop`: the integration branch and repository default during development.
- `feature/<description>`: branch from `develop`; open a PR back to `develop`.
- `release/<version>`: branch from `develop` to stabilize a release. Merge into
  `main`, tag `v<version>`, then merge the release back into `develop`.
- `hotfix/<description>`: branch from `main`; merge into `main` and `develop`,
  and tag the patch release.

Create a feature branch:

```bash
git switch develop
git pull --ff-only
git switch -c feature/dataset-audit
```

Prefer merge commits for release integration so both long-lived branches retain
their shared history. Never rewrite shared branches. Gitflow uses ordinary Git;
the `git-flow` extension is optional.

## Local checks

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv build
```

CI runs these checks on Python 3.11, 3.12, and 3.13. Test data transformations,
split isolation, prompt leakage, metrics, and adapter reload behavior as those
features arrive. Keep routine CI offline after dependency installation and free
of GPU requirements. Do not test model quality with unit-test assertions.

Use descriptive commits such as `feat: audit ScienceQA splits` or
`fix: reject answer labels in inference prompts`. Explain the behavior change,
validation, and material limitations in each PR.

## Experiments

Commit configuration and small reports. Keep raw datasets, predictions containing
dataset text, model weights, adapters, caches, and secrets out of Git. Clear
notebook outputs before committing. Dataset/model artifacts retain separate
licensing obligations.

When adding dependencies, use `uv add` (or `uv add --dev`) and commit both
`pyproject.toml` and `uv.lock`. Add training dependencies only with the training
implementation and validate them on the intended GPU runtime.
