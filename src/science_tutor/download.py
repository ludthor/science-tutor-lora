"""Fetch the pinned ScienceQA source and verify it before publishing a cache entry."""

import hashlib
import json
import tempfile
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from urllib.request import urlopen

from science_tutor.config import DatasetSettings

_CHUNK_BYTES = 1024 * 1024
_MAX_SOURCE_BYTES = 50 * 1024 * 1024


class SourceDownloadError(ValueError):
    """A source transfer failed or exceeded its size limit."""


class SourceIntegrityError(SourceDownloadError):
    """The source bytes differ from the checksum pinned in the configuration."""


@dataclass(frozen=True)
class SourceArtifact:
    """A verified local source, ready for structural validation."""

    path: Path
    sha256: str
    size_bytes: int
    cached: bool


def _verify(path: Path, expected_sha256: str) -> tuple[str, int]:
    with path.open("rb") as stream:
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        raise SourceIntegrityError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}."
        )
    return actual_sha256, path.stat().st_size


def _download(url: str, destination: Path, expected_sha256: str) -> tuple[str, int]:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".problems-", suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            with urlopen(url, timeout=30) as response:
                size = 0
                while chunk := response.read(_CHUNK_BYTES):
                    size += len(chunk)
                    if size > _MAX_SOURCE_BYTES:
                        raise SourceDownloadError(
                            f"Source exceeds the {_MAX_SOURCE_BYTES}-byte download limit."
                        )
                    output.write(chunk)
        verified = _verify(temporary, expected_sha256)
        temporary.replace(destination)
        return verified
    except HTTPException as exc:
        raise SourceDownloadError(f"Source transfer failed: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_provenance(path: Path, metadata: dict[str, str | int]) -> None:
    """Replace the manifest atomically; regenerate it from verified bytes on reuse."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".source-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_scienceqa(settings: DatasetSettings, data_dir: Path) -> SourceArtifact:
    """Reuse a verified cache or download the configured source over HTTPS.

    Existing corrupt files are left untouched and rejected. Fresh downloads are
    size-limited and verified before atomic replacement. The checksum comes from
    the experiment configuration, never from the local provenance manifest.
    """
    directory = data_dir / "scienceqa" / settings.revision
    path = directory / "problems.json"
    url = (
        f"https://raw.githubusercontent.com/{settings.repository}/"
        f"{settings.revision}/data/scienceqa/problems.json"
    )
    cached = path.exists()
    if cached:
        digest, size = _verify(path, settings.sha256)
    else:
        directory.mkdir(parents=True, exist_ok=True)
        digest, size = _download(url, path, settings.sha256)
    _write_provenance(
        directory / "source.json",
        {
            "repository": settings.repository,
            "revision": settings.revision,
            "url": url,
            "sha256": digest,
            "bytes": size,
        },
    )
    return SourceArtifact(path=path, sha256=digest, size_bytes=size, cached=cached)
