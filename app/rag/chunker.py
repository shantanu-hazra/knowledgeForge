# chunker.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rag.parser import Block, BlockType, ParsedDocument


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Chunker:
    """
    Splits a cleaned ParsedDocument into overlapping chunks.

    Structure-aware, matching what parser.py hands us:
      - TABLE blocks are emitted as their own atomic chunk -- never
        split, never merged with surrounding text. Splitting a table
        mid-row would destroy the " | " structure the loaders built.
      - HEADING blocks aren't chunked on their own; instead the most
        recent heading text is carried in every chunk's metadata as
        `section`, so retrieval/citation can say which section a chunk
        came from even after the heading block itself isn't a full chunk.
    """

    _BREAK_PATTERNS = [
        re.compile(r'\n\s*\n'),        # paragraph break
        re.compile(r'(?<=[.!?])\s+'),  # sentence break
        re.compile(r'\s+'),            # word break
    ]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, strategy: str = "recursive"):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if strategy not in ("recursive", "fixed"):
            raise ValueError(f"unknown strategy: {strategy}")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def chunk(self, parsed_document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer = ""
        current_section: str | None = None

        def flush_buffer(final: bool = False):
            nonlocal buffer
            while len(buffer) >= self.chunk_size or (final and buffer.strip()):
                if len(buffer) < self.chunk_size:
                    # final flush, whatever's left
                    text = buffer.strip()
                    buffer = ""
                else:
                    piece = buffer[: self.chunk_size]
                    split_point = self._find_split_point(piece)
                    text = buffer[:split_point].strip()
                    advance = max(split_point - self.chunk_overlap, 1)
                    buffer = buffer[advance:]

                if text:
                    chunks.append(Chunk(
                        text=text,
                        metadata={
                            **parsed_document.metadata,
                            "chunk_index": len(chunks),
                            "section": current_section,
                        },
                    ))

                if not final and len(buffer) < self.chunk_size:
                    break

        for block in parsed_document.blocks:
            if block.type == BlockType.HEADING:
                # Flush whatever's pending so text before the heading
                # doesn't get attributed to the new section, then update
                # the running section label. Heading text itself isn't
                # emitted as a standalone chunk.
                flush_buffer(final=True)
                current_section = block.text
                continue

            if block.type == BlockType.TABLE:
                flush_buffer(final=True)
                chunks.append(Chunk(
                    text=block.text,
                    metadata={
                        **parsed_document.metadata,
                        **block.metadata,  # e.g. num_rows
                        "chunk_index": len(chunks),
                        "section": current_section,
                        "block_type": "table",
                    },
                ))
                continue

            # PARAGRAPH / LIST_ITEM: accumulate into buffer
            separator = "\n\n" if block.type == BlockType.PARAGRAPH else "\n"
            buffer += (separator if buffer else "") + block.text
            flush_buffer(final=False)

        flush_buffer(final=True)
        return chunks

    def _find_split_point(self, piece: str) -> int:
        if self.strategy == "fixed":
            return len(piece)

        for pattern in self._BREAK_PATTERNS:
            matches = list(pattern.finditer(piece))
            if matches:
                split_at = matches[-1].end()
                if split_at > self.chunk_size * 0.5:
                    return split_at

        return len(piece)