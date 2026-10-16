# Technical decisions

## 2026-10-16 — Keep notebooks thin

Put configuration, data preparation, training, and evaluation in a typed Python
package. Use notebooks to install the package and invoke it. This gives local
development, Colab, and CI the same implementation and keeps notebook state from
becoming an implicit dependency.

## 2026-10-16 — Start with CUDA LoRA

Use PyTorch, Transformers, and PEFT for the first training implementation on a
Colab NVIDIA GPU. Begin with regular LoRA on a 1.7B checkpoint. Introduce 4-bit
QLoRA if measured memory use requires it; check the device's precision support
instead of assuming bfloat16 is available.

An MLX path for the local M4 is a later extension. Do not assume PEFT and MLX
adapter files are interchangeable. Share canonical data and metric definitions
while allowing backend-specific loading and serialization.

## 2026-10-16 — Lock environments and source inputs

Use uv, a committed lockfile, TOML configuration, and immutable source revisions.
Python 3.12 is the local default, with 3.11–3.13 covered by CI. The foundation has
only one application dependency, Pydantic, for strict typed validation. Add the
training stack alongside its implementation and verify it in a GPU notebook.

## 2026-10-16 — Use Gitflow with feature PRs

Keep `main` for releases and `develop` for integration. Review features through
PRs to `develop`; stabilize releases separately. Use standard Git commands so
contributors do not need a Gitflow extension.
