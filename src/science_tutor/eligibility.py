"""Explain source eligibility without modifying records or choosing training samples."""

import json
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypedDict

from science_tutor.config import DatasetSettings
from science_tutor.data import ScienceQARecord, Split

POLICY_VERSION = "1"
EligibilityStatus = Literal["eligible", "excluded", "review"]
EligibilityReason = Literal[
    "other_subject", "image_present", "missing_solution", "possible_visual_reference"
]
_STATUSES: tuple[EligibilityStatus, ...] = ("eligible", "excluded", "review")
_SPLITS: tuple[Split, ...] = ("train", "val", "test")
_VISUAL = r"(?:diagram|figure|image|picture|photo(?:graph)?|illustration|graph|chart|table|map)s?"
_VISUAL_RULES = {
    "visual_reference": re.compile(
        rf"\b(?:this|these|the|each|following|provided|accompanying)\s+"
        rf"(?:(?:bar|line|pie|scatter|data|labeled|labelled|following)\s+)?{_VISUAL}\b",
        re.IGNORECASE,
    ),
    "visual_location": re.compile(
        rf"\b{_VISUAL}\s+(?:above|below|shows|illustrates|depicts|compares|represents)\b",
        re.IGNORECASE,
    ),
    "spatial_reference": re.compile(
        r"\b(?:shown|pictured|illustrated|depicted)\s+(?:above|below|here)\b", re.IGNORECASE
    ),
    "visual_marker": re.compile(rf"\b(?:fig\.?|figure)\s*\d+\b|\b{_VISUAL}\s*:", re.IGNORECASE),
    "embedded_image": re.compile(r"<img\b|!\[[^\]]*\]\([^)]*\)", re.IGNORECASE),
}


@dataclass(frozen=True)
class ReviewFlag:
    """Name the field and heuristic so a reviewer can inspect the original text."""

    field: str
    rule: str


@dataclass(frozen=True)
class EligibilityDecision:
    """One auditable decision per source record; no source text is copied."""

    record_id: str
    split: Split
    status: EligibilityStatus
    reasons: tuple[EligibilityReason, ...]
    review_flags: tuple[ReviewFlag, ...] = ()


class EligibilitySummary(TypedDict):
    total: int
    by_status: dict[str, int]
    by_split: dict[str, dict[str, int]]
    by_reason: dict[str, int]


def assess_eligibility(
    records: Mapping[str, ScienceQARecord], subject: str = "natural science"
) -> tuple[EligibilityDecision, ...]:
    """Apply policy v1, keeping the source order and upstream split assignments.

    Clear eligibility failures are excluded with every applicable reason. Only
    records passing those rules are scanned for visual references. Matches in
    inputs or reference solutions require review and are withheld from the
    eligible pool. Unused lecture annotations are not scanned.
    """
    decisions = []
    for record_id, record in records.items():
        reasons: list[EligibilityReason] = []
        if record.subject != subject:
            reasons.append("other_subject")
        if record.image is not None:
            reasons.append("image_present")
        if not record.solution.strip():
            reasons.append("missing_solution")
        if reasons:
            decisions.append(
                EligibilityDecision(record_id, record.split, "excluded", tuple(reasons))
            )
            continue

        fields = [
            ("question", record.question),
            ("hint", record.hint),
            *[(f"choices.{index}", choice) for index, choice in enumerate(record.choices)],
            ("solution", record.solution),
        ]
        flags = tuple(
            ReviewFlag(field, name)
            for field, text in fields
            for name, pattern in _VISUAL_RULES.items()
            if pattern.search(text)
        )
        decisions.append(
            EligibilityDecision(
                record_id,
                record.split,
                "review" if flags else "eligible",
                ("possible_visual_reference",) if flags else (),
                flags,
            )
        )
    return tuple(decisions)


def summarize_eligibility(decisions: Sequence[EligibilityDecision]) -> EligibilitySummary:
    """Partition counts by status; reason counts may overlap for excluded records."""
    by_status: dict[str, int] = {status: 0 for status in _STATUSES}
    by_split: dict[str, dict[str, int]] = {
        split: {status: 0 for status in _STATUSES} for split in _SPLITS
    }
    by_reason: Counter[str] = Counter()
    for decision in decisions:
        by_status[decision.status] += 1
        by_split[decision.split][decision.status] += 1
        by_reason.update(decision.reasons)
    return {
        "total": len(decisions),
        "by_status": by_status,
        "by_split": by_split,
        "by_reason": dict(sorted(by_reason.items())),
    }


def write_eligibility_report(
    path: Path, decisions: Sequence[EligibilityDecision], settings: DatasetSettings
) -> EligibilitySummary:
    """Atomically write deterministic provenance, counts, and per-record decisions."""
    summary = summarize_eligibility(decisions)
    report = {
        "schema_version": 1,
        "policy": {
            "version": POLICY_VERSION,
            "subject": settings.subject,
            "require_no_image": True,
            "require_nonblank_solution": True,
            "visual_reference_handling": "review",
            "visual_fields": ["question", "hint", "choices", "solution"],
            "visual_rules": {name: pattern.pattern for name, pattern in _VISUAL_RULES.items()},
            "visual_rules_ignore_case": True,
        },
        "source": {
            "repository": settings.repository,
            "revision": settings.revision,
            "sha256": settings.sha256,
        },
        "summary": summary,
        "decisions": [asdict(decision) for decision in decisions],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".eligibility-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return summary
