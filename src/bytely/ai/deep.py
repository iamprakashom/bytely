"""`bytely build --deep`: the concept pass, then the graph with meaning.

Concepts come first so the graph's cards can link up to them. Both passes
resume from what earlier runs cached: file summaries by content hash,
synthesis batches by their files, and symbol summaries by body hash in the
graph itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from bytely.ai.concepts import ConceptStats, build_concepts
from bytely.ai.crux import CruxSummarizer
from bytely.ai.providers import create_chat_model
from bytely.ai.summarize import FileSummarizer
from bytely.ai.synthesize import Synthesizer
from bytely.graph.build import (
    DeepOptions,
    GraphBuildResult,
    build_graph,
    list_source_files,
)
from bytely.graph.refresh import build_lock, load_build_state

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from bytely.ai.llm.types import ChatModel
    from bytely.ai.providers import LLMConfig


@dataclass
class DeepResult:
    """Both passes' outcomes."""

    concepts: ConceptStats
    graph: GraphBuildResult

    @property
    def degraded(self) -> bool:
        """Whether some LLM work failed (the meaning layer is incomplete)."""
        meaning = self.graph.meaning
        return bool(
            self.concepts.fatal
            or self.concepts.errors
            or (meaning and (meaning.fatal or meaning.failed_files))
        )


def run_deep(
    root: str,
    context_dir: str | None,
    config: LLMConfig,
    *,
    include_patterns: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = None,
    concurrency: int = 5,
    progress: Callable[[str, int, int, str], None] | None = None,
    model: ChatModel | None = None,
) -> DeepResult:
    """Run both passes; `model` overrides the configured one (tests)."""
    root_path = Path(root).resolve()
    ctx_dir = Path(context_dir) if context_dir else root_path / "bytely"
    state = load_build_state(str(ctx_dir))
    includes = list(
        state.include_patterns if include_patterns is None else include_patterns
    )
    excludes = list(
        state.exclude_patterns if exclude_patterns is None else exclude_patterns
    )
    chat = model or create_chat_model(config)
    with build_lock(ctx_dir):
        concepts = build_concepts(
            root_path,
            ctx_dir,
            list_source_files(root_path, includes, excludes),
            FileSummarizer(chat),
            Synthesizer(chat),
            model=config.model,
            concurrency=max(1, concurrency),
            progress=progress,
        )
        graph = build_graph(
            str(root_path),
            str(ctx_dir),
            include_patterns=includes,
            exclude_patterns=excludes,
            deep=DeepOptions(
                CruxSummarizer(chat),
                concurrency=max(1, concurrency),
                progress=(
                    (lambda i, n, path: progress("enrich", i, n, path))
                    if progress
                    else None
                ),
            ),
        )
    return DeepResult(concepts, graph)
