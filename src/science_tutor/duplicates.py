"""Audit repeated model inputs and possible near duplicates without deleting records."""

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from science_tutor.config import DatasetSettings
from science_tutor.data import ScienceQARecord, Split
from science_tutor.eligibility import POLICY_VERSION as ELIGIBILITY_POLICY_VERSION
from science_tutor.eligibility import assess_eligibility
from science_tutor.reports import write_json_report

DUPLICATE_POLICY_VERSION = "1"
NEAR_THRESHOLD = 0.9
_TOKEN = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class RecordReference:
    record_id: str
    split: Split


@dataclass(frozen=True)
class DuplicateGroup:
    fingerprint: str
    members: tuple[RecordReference, ...]
    cross_split: bool
    answer_conflict: bool


@dataclass(frozen=True)
class NearDuplicatePair:
    left: RecordReference
    right: RecordReference
    similarity: float
    cross_split: bool


@dataclass(frozen=True)
class DuplicateAudit:
    candidate_count: int
    exact_groups: tuple[DuplicateGroup, ...]
    near_pairs: tuple[NearDuplicatePair, ...]

    def summary(self) -> dict[str, int]:
        return {
            "candidate_records": self.candidate_count,
            "exact_groups": len(self.exact_groups),
            "exact_cross_split_groups": sum(group.cross_split for group in self.exact_groups),
            "exact_records": sum(len(group.members) for group in self.exact_groups),
            "exact_answer_conflict_groups": sum(
                group.answer_conflict for group in self.exact_groups
            ),
            "near_pairs": len(self.near_pairs),
            "near_cross_split_pairs": sum(pair.cross_split for pair in self.near_pairs),
        }


def normalize_text(text: str) -> str:
    """Fold case and whitespace using Unicode NFC; retain punctuation and numbers."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _fingerprint(question: str, hint: str, choices: tuple[str, ...]) -> str:
    encoded = json.dumps([question, hint, choices], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _question_bigrams(question: str) -> frozenset[tuple[str, ...]]:
    tokens = _TOKEN.findall(question)
    if len(tokens) < 2:
        return frozenset({tuple(tokens)})
    return frozenset(pairwise(tokens))


def audit_duplicates(records: Mapping[str, ScienceQARecord]) -> DuplicateAudit:
    """Audit the supplied candidate pool, independently of source dictionary order.

    Normalized question/hint/choice multisets define exact groups. Answer text
    only diagnoses conflicting annotations after grouping. Near-pair screening
    uses question-token-bigram Jaccard similarity within matching hint/choice
    contexts. It is a lexical review heuristic, not semantic equivalence.
    """
    exact: dict[str, list[str]] = defaultdict(list)
    contexts: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    fingerprints: dict[str, str] = {}
    shingles: dict[str, frozenset[tuple[str, ...]]] = {}
    references = {
        record_id: RecordReference(record_id, records[record_id].split)
        for record_id in sorted(records)
    }
    for record_id in references:
        record = records[record_id]
        question = normalize_text(record.question)
        hint = normalize_text(record.hint)
        choices = tuple(sorted(normalize_text(choice) for choice in record.choices))
        fingerprint = _fingerprint(question, hint, choices)
        fingerprints[record_id] = fingerprint
        exact[fingerprint].append(record_id)
        contexts[hint, choices].append(record_id)
        shingles[record_id] = _question_bigrams(question)

    groups = []
    for fingerprint, ids in sorted(exact.items()):
        if len(ids) < 2:
            continue
        answers = {normalize_text(records[key].choices[records[key].answer]) for key in ids}
        groups.append(
            DuplicateGroup(
                fingerprint,
                tuple(references[key] for key in ids),
                len({records[key].split for key in ids}) > 1,
                len(answers) > 1,
            )
        )

    pairs = []
    for ids in contexts.values():
        # The index avoids comparing questions with no shared bigram.
        index: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for record_id in ids:
            current = shingles[record_id]
            candidates: set[str] = set()
            for token_pair in current:
                candidates.update(index[token_pair])
            for other_id in sorted(candidates):
                if fingerprints[record_id] == fingerprints[other_id]:
                    continue
                other = shingles[other_id]
                if min(len(current), len(other)) / max(len(current), len(other)) < NEAR_THRESHOLD:
                    continue
                similarity = len(current & other) / len(current | other)
                if similarity >= NEAR_THRESHOLD:
                    pairs.append(
                        NearDuplicatePair(
                            references[other_id],
                            references[record_id],
                            similarity,
                            records[other_id].split != records[record_id].split,
                        )
                    )
            for token_pair in current:
                index[token_pair].add(record_id)
    return DuplicateAudit(
        len(records),
        tuple(groups),
        tuple(sorted(pairs, key=lambda pair: (pair.left.record_id, pair.right.record_id))),
    )


def write_duplicate_report(
    path: Path, records: Mapping[str, ScienceQARecord], settings: DatasetSettings
) -> dict[str, int]:
    """Recompute eligibility, audit candidates, and write a provenance-backed report."""
    decisions = assess_eligibility(records, settings.subject)
    candidates = {
        decision.record_id: records[decision.record_id]
        for decision in decisions
        if decision.status == "eligible"
    }
    audit = audit_duplicates(candidates)
    summary = audit.summary()
    report = {
        "schema_version": 1,
        "policy": {
            "version": DUPLICATE_POLICY_VERSION,
            "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
            "subject": settings.subject,
            "scope": "eligible_candidates",
            "normalization": (
                "NFC, casefold, collapsed whitespace; punctuation and numbers retained"
            ),
            "exact_fields": ["question", "hint", "sorted_choices_with_multiplicity"],
            "answer_use": "annotation_conflict_check_only",
            "near_context": "matching normalized hint and choice multiset",
            "near_metric": "Jaccard on question token bigram sets",
            "near_token_pattern": _TOKEN.pattern,
            "near_threshold": NEAR_THRESHOLD,
            "action": "audit_only; no records removed or splits changed",
        },
        "source": {
            "repository": settings.repository,
            "revision": settings.revision,
            "sha256": settings.sha256,
        },
        "summary": summary,
        "candidate_ids": sorted(candidates),
        "exact_groups": [asdict(group) for group in audit.exact_groups],
        "near_pairs": [asdict(pair) for pair in audit.near_pairs],
    }
    write_json_report(path, report)
    return summary
