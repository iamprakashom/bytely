"""Exercise the installed distribution outside its source checkout."""

# ruff: noqa: S603 -- standalone packaging subprocess checks

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    """Build, query, and check all three gate languages."""
    with TemporaryDirectory(prefix="bytely-package-") as directory:
        root = Path(directory)
        sources = {
            "example.py": 'def hello():\n    return "hello"\n',
            "example.rs": "fn hello() -> u8 { 1 }\n",
            "example.js": 'export function hello() { return "hello"; }\n',
        }
        for name, content in sources.items():
            (root / name).write_text(content, encoding="utf-8")
        commands = [
            ["bytely", "--version"],
            [sys.executable, "-m", "bytely", "--help"],
            ["bytely", "build"],
            ["bytely", "check", "--no-cache"],
            ["bytely", "map"],
        ]
        commands.extend(["bytely", "skeleton", name] for name in sources)
        for command in commands:
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
