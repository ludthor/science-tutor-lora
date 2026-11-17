"""Exercise transfers and cache integrity using invented bytes and no network."""

import io
import json
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import URLError

import pytest

from science_tutor import download
from science_tutor.config import DatasetSettings
from science_tutor.download import SourceDownloadError, SourceIntegrityError, fetch_scienceqa


def test_downloads_pinned_url_and_writes_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_settings: DatasetSettings,
    source_payload: bytes,
) -> None:
    requests = []

    def open_source(url: str, timeout: int) -> io.BytesIO:
        requests.append((url, timeout))
        return io.BytesIO(source_payload)

    monkeypatch.setattr(download, "urlopen", open_source)
    # Small chunks exercise streaming across multiple reads.
    monkeypatch.setattr(download, "_CHUNK_BYTES", 7)

    result = fetch_scienceqa(source_settings, tmp_path)

    expected_url = (
        f"https://raw.githubusercontent.com/lupantech/ScienceQA/"
        f"{source_settings.revision}/data/scienceqa/problems.json"
    )
    assert requests == [(expected_url, 30)]
    assert result.cached is False
    assert result.path == tmp_path / "scienceqa" / source_settings.revision / "problems.json"
    assert result.path.read_bytes() == source_payload
    assert result.sha256 == source_settings.sha256
    assert result.size_bytes == len(source_payload)
    assert json.loads((result.path.parent / "source.json").read_text()) == {
        "repository": "lupantech/ScienceQA",
        "revision": source_settings.revision,
        "url": expected_url,
        "sha256": source_settings.sha256,
        "bytes": len(source_payload),
    }
    assert sorted(path.name for path in result.path.parent.iterdir()) == [
        "problems.json",
        "source.json",
    ]


def test_rechecks_cache_without_network_and_rebuilds_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_settings: DatasetSettings,
    source_payload: bytes,
) -> None:
    directory = tmp_path / "scienceqa" / source_settings.revision
    directory.mkdir(parents=True)
    path = directory / "problems.json"
    path.write_bytes(source_payload)
    (directory / "source.json").write_text("corrupt metadata", encoding="utf-8")

    def reject_network(*args: object, **kwargs: object) -> None:
        pytest.fail("A verified cache must not need the network.")

    monkeypatch.setattr(download, "urlopen", reject_network)

    first = fetch_scienceqa(source_settings, tmp_path)
    manifest = (directory / "source.json").read_bytes()
    second = fetch_scienceqa(source_settings, tmp_path)

    assert first.cached is second.cached is True
    assert path.read_bytes() == source_payload
    assert json.loads(manifest)["sha256"] == source_settings.sha256
    assert (directory / "source.json").read_bytes() == manifest


def test_rejects_corrupt_cache_even_when_manifest_claims_correct_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_settings: DatasetSettings
) -> None:
    directory = tmp_path / "scienceqa" / source_settings.revision
    directory.mkdir(parents=True)
    path = directory / "problems.json"
    path.write_bytes(b"corrupted")
    manifest = directory / "source.json"
    manifest.write_text(json.dumps({"sha256": source_settings.sha256}), encoding="utf-8")
    before = manifest.read_bytes()

    def reject_network(*args: object, **kwargs: object) -> None:
        pytest.fail("A corrupt cache must fail explicitly instead of being silently replaced.")

    monkeypatch.setattr(download, "urlopen", reject_network)

    with pytest.raises(SourceIntegrityError, match="SHA-256 mismatch"):
        fetch_scienceqa(source_settings, tmp_path)

    assert path.read_bytes() == b"corrupted"
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("payload", [b"wrong bytes", b""])
def test_rejects_unexpected_download_and_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_settings: DatasetSettings,
    payload: bytes,
) -> None:
    monkeypatch.setattr(download, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))

    with pytest.raises(SourceIntegrityError):
        fetch_scienceqa(source_settings, tmp_path)

    directory = tmp_path / "scienceqa" / source_settings.revision
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize("failure", [URLError("offline"), TimeoutError("timed out")])
def test_network_failure_leaves_no_partial_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_settings: DatasetSettings,
    failure: OSError,
) -> None:
    def fail_download(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(download, "urlopen", fail_download)

    with pytest.raises(OSError):
        fetch_scienceqa(source_settings, tmp_path)

    assert list((tmp_path / "scienceqa" / source_settings.revision).iterdir()) == []


def test_interrupted_transfer_cleans_partial_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_settings: DatasetSettings
) -> None:
    class InterruptedResponse(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            if self.tell():
                raise IncompleteRead(b"partial")
            return super().read(size)

    monkeypatch.setattr(download, "_CHUNK_BYTES", 3)
    monkeypatch.setattr(
        download, "urlopen", lambda *args, **kwargs: InterruptedResponse(b"partial source")
    )

    with pytest.raises(SourceDownloadError, match="transfer failed"):
        fetch_scienceqa(source_settings, tmp_path)

    assert list((tmp_path / "scienceqa" / source_settings.revision).iterdir()) == []


def test_download_limit_cleans_partial_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_settings: DatasetSettings,
    source_payload: bytes,
) -> None:
    monkeypatch.setattr(download, "_MAX_SOURCE_BYTES", len(source_payload) - 1)
    monkeypatch.setattr(download, "_CHUNK_BYTES", 3)
    monkeypatch.setattr(download, "urlopen", lambda *args, **kwargs: io.BytesIO(source_payload))

    with pytest.raises(SourceDownloadError, match="download limit"):
        fetch_scienceqa(source_settings, tmp_path)

    assert list((tmp_path / "scienceqa" / source_settings.revision).iterdir()) == []


def test_provenance_write_failure_preserves_verified_source_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_settings: DatasetSettings,
    source_payload: bytes,
) -> None:
    original_replace = Path.replace

    def fail_manifest_write(path: Path, target: Path) -> Path:
        if target.name == "source.json":
            raise OSError("simulated manifest write failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_write)
    monkeypatch.setattr(download, "urlopen", lambda *args, **kwargs: io.BytesIO(source_payload))

    with pytest.raises(OSError, match="manifest write failure"):
        fetch_scienceqa(source_settings, tmp_path)

    directory = tmp_path / "scienceqa" / source_settings.revision
    assert [path.name for path in directory.iterdir()] == ["problems.json"]
    assert (directory / "problems.json").read_bytes() == source_payload

    def reject_network(*args: object, **kwargs: object) -> None:
        pytest.fail("The completed source must be reusable after a provenance failure.")

    monkeypatch.setattr(Path, "replace", original_replace)
    monkeypatch.setattr(download, "urlopen", reject_network)

    result = fetch_scienceqa(source_settings, tmp_path)

    assert result.cached is True
    assert json.loads((directory / "source.json").read_text())["sha256"] == source_settings.sha256
