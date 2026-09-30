"""A plain-prose summary of one source file, for the concept pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bytely.ai.llm.types import ChatRequest, Message

if TYPE_CHECKING:
    from bytely.ai.llm.types import ChatModel

SYSTEM_PROMPT = """\
You document source code for a team knowledge base. Given one source file, write a compact plain-English summary covering:
1. The purpose of the file — what it exists to do.
2. The key exported functions/classes/types and what each is for.
3. Important dependencies: internal modules it builds on, external libraries or services it talks to.
4. Notable design decisions, constraints, or gotchas evident in the code.

Write 3-8 sentences of flowing prose. Name concrete identifiers (modules, classes, services) so they can become graph entities. No code blocks, no line-by-line narration, no filler."""  # noqa: E501

MAX_CODE_CHARS = 24_000


class FileSummarizer:
    """Summarizes a file with one plain-text call."""

    def __init__(self, model: ChatModel) -> None:
        """Use `model` for every call."""
        self.model = model

    def summarize(self, path: str, code: str) -> str:
        """The file's summary (empty if the model returned nothing)."""
        if len(code) > MAX_CODE_CHARS:
            code = (
                code[:MAX_CODE_CHARS]
                + f"\n… (truncated at {MAX_CODE_CHARS} characters)"
            )
        response = self.model.create(
            ChatRequest(
                messages=[
                    Message("system", SYSTEM_PROMPT),
                    Message("user", f"File: {path}\n\n{code}"),
                ],
                max_tokens=2048,
            )
        )
        return response.text.strip()
