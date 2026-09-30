"""Serialize and deserialize the graph file, `.graph/wiring.json`."""

import json
import os
from pathlib import Path

from bytely.graph.types import GraphV1

# We use orjson if available for speed, fallback to stdlib json
try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

# Where the graph lives inside an output directory.
GRAPH_FILE = Path(".graph") / "wiring.json"
# Where builds before the rename wrote it; removed on the next write.
LEGACY_GRAPH_FILE = Path("graph.json")


def graph_path(context_dir: str | Path) -> Path:
    """The graph file inside an output directory."""
    return Path(context_dir) / GRAPH_FILE


def write_graph(graph: GraphV1, context_dir: str) -> None:
    """Serialize a graph into its context directory."""
    path = graph_path(context_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    (Path(context_dir) / LEGACY_GRAPH_FILE).unlink(missing_ok=True)

    data = graph.to_dict()
    payload = (
        orjson.dumps(data, option=orjson.OPT_INDENT_2)
        if HAS_ORJSON
        else json.dumps(data, indent=2).encode("utf-8")
    )
    # Atomic: a build killed mid-write (a `--deep` checkpoint, Ctrl-C)
    # must never leave a truncated graph behind.
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_bytes(payload)
        os.replace(temp, path)
    except OSError:
        temp.unlink(missing_ok=True)
        raise


def read_graph(context_dir: str) -> GraphV1 | None:
    """Load a graph; None when the file is missing or unreadable.

    An unreadable graph (truncated, hand-edited, an older schema) is
    treated as absent, so the next build replaces it instead of failing.
    """
    path = graph_path(context_dir)
    try:
        raw = path.read_bytes()
        data = orjson.loads(raw) if HAS_ORJSON else json.loads(raw)
        return GraphV1.from_dict(data)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
