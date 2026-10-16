# Data and model provenance

## ScienceQA

- Original source: <https://github.com/lupantech/ScienceQA>.
- Pinned revision: `2cbf8318e07b9ece895bb2ae605e71e38d623264`.
- Text records: `data/scienceqa/problems.json` in the upstream repository.
- Paper: Lu et al., *Learn to Explain: Multimodal Reasoning via Thought Chains
  for Science Question Answering*, NeurIPS 2022.
- Dataset license: [CC BY-NC-SA 4.0](https://github.com/lupantech/ScienceQA/blob/2cbf8318e07b9ece895bb2ae605e71e38d623264/LICENSE-DATA).

The full dataset contains 21,208 multiple-choice questions across natural,
language, and social science, with answer, lecture, and solution annotations.
The final eligible subset size is unknown until the full audit runs.

## Initial source inspection — 2026-10-16

The pinned `problems.json` file is 31,529,211 bytes. Its observed SHA-256 is
`4d9b598da966d9736dd79e430a97da861a2216aeb7483a5092350e823ab20ce7`.
The JSON root maps string record IDs to question records. Answers are zero-based
indices into `choices`; upstream calls the validation split `val`.

| Split | Full source | Natural science with `image = null` |
| --- | ---: | ---: |
| train | 12,726 | 3,037 |
| val | 4,241 | 1,075 |
| test | 4,241 | 1,043 |
| Total | 21,208 | 5,155 |

Among these 5,155 candidates, 1,098 have empty `solution` strings and 1,753 have
empty `hint` strings. No empty questions or invalid answer indices were found
in this initial inspection. These are descriptive checks, not full schema or
scientific-quality validation.

Empty hints are allowed because context is optional. Records without reference
solutions need an explicit eligibility decision before explanation training.
An absent image alone does not establish that a question is answerable from text:
diagram references, duplicates, and other quality checks remain pending.

The downloaded source and its local `source.json` provenance record are stored
under `data/raw/scienceqa/<revision>/`, which is ignored by Git. The observed
checksum records what was downloaded; automated integrity checks are a later step.

## Planned selection and audit

1. Download from the pinned revision and record a content checksum.
2. Keep natural-science records with no image. Inspect references to absent
   diagrams, malformed choices, and missing explanations; record all exclusions.
3. Preserve upstream record IDs and train/validation/test assignments.
4. Audit exact and near duplicates across splits. Document the detection rule
   and any exclusions before training; never move test examples into training.
5. Sample train/validation deterministically using the configured seed and save
   the selected IDs. Keep the eligible test split isolated.

Inference inputs may contain `question`, `choices`, and audited `hint` text.
The `answer` label and `solution` explanation are targets. `lecture` is an
annotation and will not be provided as inference context in version one.
Use the same input policy for baselines and fine-tuning.

## Model

- Checkpoint: [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct).
- Pinned revision: `31b70e2e869a7173562077fd711b654946d38674`.
- Upstream license: Apache 2.0.

Revisions were resolved on 2026-10-16. The configuration validates SHA syntax;
remote contents and checksums will be verified by the data/model loaders.

## Distribution

Original code uses MIT. Third-party datasets, source text, model weights, and
derived artifacts are not covered by that grant. Retain attribution and record
applicable noncommercial/share-alike requirements when preparing artifacts for
distribution. No raw data or trained artifacts are included in this repository.
