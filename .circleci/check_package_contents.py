"""Validate public distribution contents before uploading artifacts."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path


def main() -> None:
    """Reject files outside the distribution allowlists."""
    sources = sorted(Path("dist").glob("*.tar.gz"))
    wheels = sorted(Path("dist").glob("*.whl"))
    if not sources or not wheels:
        raise SystemExit("Both source and wheel distributions are required")
    documents = {
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "PKG-INFO",
        ".gitignore",
    }
    for source in sources:
        with tarfile.open(source) as archive:
            for member in archive.getmembers():
                if member.isdir():
                    continue
                relative = member.name.partition("/")[2]
                if not member.isfile() or not (
                    relative in documents or relative.startswith("src/bytely/")
                ):
                    raise SystemExit(f"Unexpected source entry: {member.name}")
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                root = name.partition("/")[0]
                if not (
                    name.startswith("bytely/")
                    or (
                        root.startswith("bytely-")
                        and root.endswith(".dist-info")
                    )
                ):
                    raise SystemExit(f"Unexpected wheel entry: {name}")


if __name__ == "__main__":
    main()
