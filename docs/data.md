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
The current policy yields 3,901 eligible candidates before duplicate analysis and
manual quality review; final sample IDs are not selected yet.

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
under `data/raw/scienceqa/<revision>/`, which is ignored by Git. This observed
checksum is now pinned in `configs/smoke.toml` for automated integrity checks.

## Verified download and cache reuse

Run from the repository root:

```bash
uv run science-tutor fetch-data configs/smoke.toml
```

The command uses `dataset.repository`, `dataset.revision`, and `dataset.sha256`
from the validated configuration. It returns JSON containing `path`, `sha256`,
`bytes`, and `cached`. A first successful fetch reports `cached: false`; subsequent
runs rehash the existing file and report `cached: true`, without a network call.
This verifies equality with the bytes observed during the initial source
inspection, not the dataset's scientific correctness or a publisher signature.

Downloads use HTTPS with a 30-second socket timeout, stream in chunks, and stop
after 50 MiB. A temporary file in the destination directory is verified before
atomic replacement of `problems.json`. Ordinary transfer failures, checksum
mismatches, and oversized responses clean up that temporary file. A killed
process may leave an ignored temporary file, but it will not be treated as the
cached source. The cache targets local, single-process use.

An existing cache with the wrong checksum fails explicitly and is left unchanged.
Inspect the reported path and remove that corrupt file before retrying, or use
`--data-dir <another-directory>` for a fresh download. Network and integrity
failures return exit code 2 from the CLI; retry the same command after resolving
the reported problem.

The sibling `source.json` manifest records the repository, revision, URL,
SHA-256, and byte count. Its contents are regenerated atomically from verified
bytes on each successful invocation. Verification uses the configured digest,
not a checksum supplied by that manifest. If writing provenance fails after a
verified download, the valid source remains reusable on the next invocation.

Downloading and schema validation are separate steps. Run `inspect-data` after
`fetch-data`; a matching checksum alone does not validate record structure.

## Local source validation

The `ScienceQARecord` schema and `load_scienceqa(path)` loader live in
`src/science_tutor/data.py`. The loader reads a local UTF-8 JSON object whose
keys are record IDs, and returns a mapping to typed records. It makes no network
calls and does not modify the source file.

The schema requires a nonblank question, at least two nonblank string choices,
and a strictly typed integer answer indexing those choices. Booleans, numeric
strings, and floats are rejected as answer labels. Known split, subject, and task
names are validated; unknown fields are rejected to surface schema changes.
IDs, source text, record order, and upstream split names are preserved.

Hint, lecture, and solution text may be empty or omitted. Omitted reference-text
fields default to empty strings. Explicit null values are rejected for these
string fields; the image field allows null. Records from every subject and those
with images remain valid here, since selection is a subsequent operation.

Malformed records fail the entire load rather than being silently discarded.
Validation messages identify record IDs and show up to five errors without
printing complete source records. Duplicate JSON IDs or fields are rejected
before the JSON parser can overwrite their earlier values.

Run the CLI from the repository root:

```bash
uv run science-tutor inspect-data \
  data/raw/scienceqa/2cbf8318e07b9ece895bb2ae605e71e38d623264/problems.json
```

The pinned file passes this schema: 21,208 records, with 12,726 `train`, 4,241
`val`, and 4,241 `test` records. This is structural validation only. The following
policy decides eligibility for our particular experiment.

## Eligibility policy v1

Run from the repository root:

```bash
uv run science-tutor filter-data configs/smoke.toml
```

The command fetches or verifies the configured source, validates every record,
and writes `data/processed/eligibility.json`. Use `--data-dir` to change the cache
root or `--output` to select a report path. Configuration and source files cannot
be used as the output path. Input, integrity, and report-write failures return
exit code 2. Report replacement is atomic and reruns produce identical bytes
for the same source, policy, and subject setting.

The policy assigns exactly one status to every record, retaining its original ID,
source order, and split. It does not modify source text or move records between
splits.

- **Excluded:** the subject differs from the configured natural science subject,
  an image is present, or the reference solution is empty after whitespace
  stripping. Every applicable exclusion reason is recorded. An empty hint alone
  does not exclude a record.
- **Review:** after passing those base rules, a record contains a possible
  visual reference in its question, hint, choices, or reference solution.
  Review records are withheld from the eligible pool pending human inspection.
- **Eligible:** a record passes the base rules and has no visual-reference flags.
  This does not establish scientific correctness or complete text independence.

Visual screening uses case-insensitive, context-based rules for phrases such as
"the diagram", "each picture", numbered figures, captions, spatial references,
and embedded HTML/Markdown images. Each flag records its field (including a
zero-based choice index when applicable) and rule name. Unused lecture
annotations are not screened. The report stores the actual regex rules and
policy version alongside the source revision and checksum.

The rules were developed using training records and invented test examples,
then applied uniformly to all three splits. They can produce both false
positives and false negatives: a reference to a physical table may be flagged,
while an implicit dependency on an absent illustration may be missed. No
automatic rule promotes review records back into eligibility. Scientific-quality
review of the eligible pool and human handling of the review queue remain pending.

Observed counts on 2026-10-20:

| Split | Eligible candidates | Excluded | Review | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 2,277 | 10,351 | 98 | 12,726 |
| val | 825 | 3,384 | 32 | 4,241 |
| test | 799 | 3,416 | 26 | 4,241 |
| Total | 3,901 | 17,151 | 156 | 21,208 |

Exclusion-reason counts overlap: 9,721 records have another subject, 10,332 have
an image, and 2,006 have an empty solution. Visual screening only runs on the
4,057 records passing those base rules; 156 receive review flags. Status counts
partition the source, whereas reason counts must not be added to obtain a total.

The report records each decision and its reasons without copying question or
reference text. It is generated under the ignored `data/` directory. Sampling,
duplicate-question analysis, prompt construction, and human review are separate
steps and have not run as part of this command.

## Planned selection and audit

1. Download from the pinned revision, verify the configured checksum, and record provenance.
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
