"""Bytely — Context graph engine for codebases.

Build a repo's context graph as a folder of linked markdown files: a local,
regenerable cache that every query keeps in sync with your code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# A `.devN` version until a release: a release commit sets the published
# version (and the CHANGELOG heading) before tagging it `v<version>`.
__version__ = "0.1.0.dev0"

__all__ = ["Bytely", "__version__"]

if TYPE_CHECKING:
    from bytely.engine import Bytely


def __getattr__(name: str) -> Any:
    # Importing `Bytely` pulls in the whole build pipeline; loading it on
    # first use keeps `import bytely.<module>` (and CLI startup) cheap.
    if name == "Bytely":
        from bytely.engine import Bytely

        return Bytely
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
