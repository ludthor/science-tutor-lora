# Project brief

## User scenario

A learner submits an English school science question and answer choices. The
model selects an answer and provides a brief explanation. Version one handles
questions answerable from text alone.

The project is a learning and portfolio experiment. It does not claim improved
student learning outcomes, reliable grading, or validated classroom readiness.

## Experimental design

Compare the same instruction-tuned checkpoint in three conditions: a fixed
zero-shot prompt, a few-shot prompt using examples from the training split, and
a LoRA adapter trained on that split. Select prompt examples and hyperparameters
using validation only. Keep a final test set untouched until selection is done.

Report generated answer accuracy and invalid-output rate. Do not silently drop
invalid predictions from the denominator. Review a fixed, recorded sample of
explanations for scientific correctness, relevance, and clarity; report who
reviewed it and how it was selected. Wording overlap alone does not establish
explanation correctness.

Measure wall time, peak GPU memory, trainable/total parameters, adapter size, and
observed compute spend. Record hardware, dtype, batch size, sequence length,
dependency versions, Git SHA, source revisions, seed, and exact data manifest.
Seeds aid repeatability; they do not guarantee identical results across hardware.

Public benchmark contamination in pretraining may be unknown. Results on our
filtered subset are not directly comparable with full ScienceQA leaderboards.

## First training checkpoint

After the data audit, run at most 20 optimizer steps with up to 500 training and
100 validation examples, a batch size of 1, and gradient accumulation of 8.
Start with a 512-token sequence limit and measure truncation before accepting it.
Mask prompt tokens from the loss and ensure the target survives tokenization.

A successful smoke test completes training, saves and reloads the adapter, and
produces evaluable predictions with recorded runtime/memory. It does not need
to outperform the baseline. Allocate paid compute only after a measured run
justifies it and a budget has been agreed.

## First release criteria

- A fresh checkout can reproduce data preparation and the documented experiment.
- Automated checks protect split boundaries and keep targets out of inputs.
- Baselines and the adapter are evaluated consistently on identical held-out IDs.
- The report includes actual measurements, representative failures, and limits.
- A simple demo loads the documented model and adapter.
- Code, data, model, and adapter terms are documented separately.

## Scope

Begin with one dataset, one checkpoint, one adapter, and one evaluation report.
RAG, multimodal questions, autonomous agents, distributed training, and extensive
hyperparameter searches are outside the initial release.
