"""Bytely — Context graph engine for codebases.

Build a repo's context graph as a folder of linked markdown files: a local,
regenerable cache that every query keeps in sync with your code.
"""

__version__ = "0.21.0a1"

from bytely.engine import Bytely

__all__ = ["Bytely", "__version__"]
