"""The Context Graph Engine."""

from collections.abc import Iterable

from bytely.graph.build import GraphBuildResult, build_graph
from bytely.graph.check import check_graph


class Bytely:
    """The Bytely engine."""

    def build(
        self,
        directory: str,
        context_dir: str | None = None,
        *,
        include_patterns: Iterable[str] = (),
        exclude_patterns: Iterable[str] = (),
    ) -> GraphBuildResult:
        """Build the deterministic Tier-1 graph for a repository."""
        return build_graph(
            directory,
            context_dir,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

    def init(self, directory: str) -> None:
        """Initialize the graph directory for a repository."""
        pass

    def check(
        self,
        directory: str,
        *,
        include_patterns: Iterable[str] = (),
        exclude_patterns: Iterable[str] = (),
    ) -> None:
        """Raise ValueError if the saved graph is stale."""
        fresh, message = check_graph(
            directory,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        if not fresh:
            raise ValueError(message)
