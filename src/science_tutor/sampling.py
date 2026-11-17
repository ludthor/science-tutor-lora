"""Build reproducible train/validation ID samples and reserve upstream test IDs."""

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

from science_tutor.config import ExperimentConfig
from science_tutor.data import ScienceQARecord, Split
from science_tutor.duplicates import DUPLICATE_POLICY_VERSION, DuplicateAudit, audit_duplicates
from science_tutor.eligibility import POLICY_VERSION as ELIGIBILITY_POLICY_VERSION
from science_tutor.eligibility import eligible_records

SAMPLING_POLICY_VERSION = "2"
_SPLITS: tuple[Split, ...] = ("train", "val", "test")
_SPLIT_PRIORITY: dict[Split, int] = {"test": 0, "val": 1, "train": 2}
QuarantineReason = Literal["conflicting_exact_answers", "near_duplicate_review", "redundant_exact"]


class QuarantineEntry(TypedDict):
    record_id: str
    split: Split
    reasons: list[QuarantineReason]
    retained_record_id: str | None


class SplitCounts(TypedDict):
    eligible: int
    quarantined: int
    quality_excluded: int
    available: int
    sampled: int
    reserved: int


class SampleManifest(TypedDict):
    schema_version: int
    experiment: str
    source: dict[str, str]
    policy: dict[str, str]
    seed: int
    limits: dict[str, int]
    by_split: dict[Split, SplitCounts]
    samples: dict[str, list[str]]
    reserved_test_ids: list[str]
    quarantine: list[QuarantineEntry]
    quality_exclusions: dict[str, str]


def _quarantine(
    candidates: Mapping[str, ScienceQARecord], audit: DuplicateAudit
) -> list[QuarantineEntry]:
    reasons: dict[str, set[QuarantineReason]] = defaultdict(set)
    replacements: dict[str, str] = {}
    near_ids = {member.record_id for pair in audit.near_pairs for member in (pair.left, pair.right)}
    for record_id in near_ids:
        reasons[record_id].add("near_duplicate_review")

    for group in audit.exact_groups:
        ids = {member.record_id for member in group.members}
        needs_near_review = bool(ids & near_ids)
        if group.answer_conflict or needs_near_review:
            # An exact copy must not bypass a review flag on another group member.
            for record_id in ids:
                if group.answer_conflict:
                    reasons[record_id].add("conflicting_exact_answers")
                if needs_near_review:
                    reasons[record_id].add("near_duplicate_review")
            continue

        retained = min(
            group.members, key=lambda member: (_SPLIT_PRIORITY[member.split], member.record_id)
        )
        for record_id in ids - {retained.record_id}:
            reasons[record_id].add("redundant_exact")
            replacements[record_id] = retained.record_id

    return [
        {
            "record_id": record_id,
            "split": candidates[record_id].split,
            "reasons": sorted(reasons[record_id]),
            "retained_record_id": replacements.get(record_id),
        }
        for record_id in sorted(reasons)
    ]


def _sample(ids: Sequence[str], split: Split, seed: int, limit: int) -> list[str]:
    def rank(record_id: str) -> tuple[bytes, str]:
        key = f"{seed}:{split}:{record_id}".encode()
        return hashlib.sha256(key).digest(), record_id

    return sorted(ids, key=rank)[:limit]


def build_sample_manifest(
    records: Mapping[str, ScienceQARecord], config: ExperimentConfig
) -> SampleManifest:
    """Recompute the audit, enforce quarantine, and emit split-preserving ID lists."""
    candidates = eligible_records(records, config.dataset.subject)
    quarantine = _quarantine(candidates, audit_duplicates(candidates))
    withheld = {entry["record_id"] for entry in quarantine}
    exclusions = config.dataset.quality_exclusions
    quality_counts: Counter[Split] = Counter()
    for record_id in exclusions:
        record = candidates.get(record_id)
        if record is None or record_id in withheld or record.split == "test":
            raise ValueError(
                f"Quality exclusion {record_id!r} must name a usable train/val record."
            )
        quality_counts[record.split] += 1
    eligible_counts = Counter(record.split for record in candidates.values())
    pools: dict[Split, list[str]] = {split: [] for split in _SPLITS}
    for record_id, record in candidates.items():
        if record_id not in withheld and record_id not in exclusions:
            pools[record.split].append(record_id)

    limits = {
        "train": config.dataset.max_train_examples,
        "val": config.dataset.max_validation_examples,
    }
    samples: dict[str, list[str]] = {}
    for split in ("train", "val"):
        if not pools[split]:
            raise ValueError(f"No usable {split} records remain after eligibility and quarantine.")
        samples[split] = _sample(pools[split], split, config.seed, limits[split])
    reserved = sorted(pools["test"])
    by_split: dict[Split, SplitCounts] = {
        split: {
            "eligible": eligible_counts[split],
            "quarantined": eligible_counts[split] - len(pools[split]) - quality_counts[split],
            "quality_excluded": quality_counts[split],
            "available": len(pools[split]),
            "sampled": len(samples.get(split, [])),
            "reserved": len(reserved) if split == "test" else 0,
        }
        for split in _SPLITS
    }
    return {
        "schema_version": 1,
        "experiment": config.name,
        "source": {
            "repository": config.dataset.repository,
            "revision": config.dataset.revision,
            "sha256": config.dataset.sha256,
            "subject": config.dataset.subject,
        },
        "policy": {
            "version": SAMPLING_POLICY_VERSION,
            "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
            "duplicate_policy_version": DUPLICATE_POLICY_VERSION,
            "exact_conflicts": "quarantine every member",
            "near_matches": "quarantine both endpoints and their exact copies",
            "agreed_exact_tie_break": "test > val > train, then lexicographic record ID",
            "sampling_rank": "SHA-256 of UTF-8 seed:split:record_id, then record ID",
            "quality_exclusions": "apply documented train/val exclusions before sampling",
        },
        "seed": config.seed,
        "limits": limits,
        "by_split": by_split,
        "samples": samples,
        "reserved_test_ids": reserved,
        "quarantine": quarantine,
        "quality_exclusions": dict(sorted(exclusions.items())),
    }
