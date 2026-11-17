# Validation prompting baselines

The task is to select an answer letter and explain it. This experiment compares
two prompts on the same validation examples using the unadapted
[SmolLM2-1.7B-Instruct checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct).
No weights are updated and no reserved test records are evaluated.

```bash
uv sync --locked --extra inference
uv run --locked --extra inference science-tutor run-baselines configs/smoke.toml
```

- **Zero-shot:** the system instruction and the validation question.
- **Few-shot:** the same instruction and question, with the first two training
  prompt/completion pairs inserted before the query. These demonstrations
  follow the seed-ranked training order. They were not chosen by model score.

Both conditions use the pinned tokenizer's chat template, greedy decoding,
batch size 8, and at most 128 new tokens per example. The runner decodes only
new tokens. A valid response must consist of one JSON object with exactly a
listed uppercase `answer` and a nonblank `explanation`. Extra text, duplicate
keys, and out-of-range choices are invalid. Accuracy is correct answers divided
by all queries, including invalid responses. Explanation presence is checked
for format; its factual quality is not scored by this metric.

Use `--device mps` on an Apple Silicon Mac, `--device cuda` on an NVIDIA GPU,
or omit the flag for automatic selection. Pass `--limit 8` for a quick smoke
test; the limit takes the same validation prefix in both conditions. `--shots`,
`--batch-size`, and `--max-new-tokens` can be overridden. The model requires
about 3.2 GiB of local space for the checkpoint. GPU inference uses float16;
the CPU fallback uses float32.

The report records model and dataset revisions, the prompt export digest, exact
validation and demonstration IDs, software versions, generation settings,
inference time, and raw responses. It is written atomically to ignored
`outputs/baselines.json` by default. Publish aggregate results rather than the
raw responses, which may reproduce data text. Timings exclude setup and model
loading; they measure both conditions together. Different runtimes or batching
choices may affect exact generated outputs.

## Measured results

Run on Apple M4 with 24 GB unified memory, PyTorch 2.15.0, Transformers 4.57.6,
Python 3.12.8, and float16 on MPS. The two conditions took **246.5 seconds** of
inference time combined. The run used the 100 seed-42 validation IDs after the
documented quality exclusions. Demonstration IDs were `18536` and `5484`.

| Prompt | Correct / total | Accuracy | Invalid responses |
| --- | ---: | ---: | ---: |
| Zero-shot | 57 / 100 | 57% | 0 / 100 |
| Two-shot | 63 / 100 | 63% | 4 / 100 |

Both prompts answered 49 questions correctly. Two-shot gained 14 correct answers
and lost eight relative to zero-shot. Three two-shot responses were invalid JSON,
and one JSON response failed the required output schema. The six-point gain is
specific to this validation sample and these demonstrations. It does not establish
better explanations or performance on the reserved test set.

Prompt dataset SHA-256 (sorted-key JSON as serialized by the runner):
`980791bf8b6cbdec700b195708c01e4134a379242fd7d31564ac677144bacd32`.
The test reserve remains unchanged at 794 IDs. Before comparing a future LoRA
adapter, rerun the relevant unadapted baseline on the same inference runtime and
with the same scoring rules.
