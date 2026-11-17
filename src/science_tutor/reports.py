"""Write deterministic JSON reports without exposing a partial replacement."""

import json
import tempfile
from pathlib import Path


def write_json_report(path: Path, report: object) -> None:
    """Replace a generated report atomically and clean up ordinary write failures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
