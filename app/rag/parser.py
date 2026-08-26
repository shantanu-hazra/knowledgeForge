from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rag.loaders.base import RawDocument


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"


@dataclass
class Block:
    type: BlockType
    text: str
    level: int | None = None          # heading level (1-6), else None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    blocks: list[Block]
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser:
    """
    Converts raw extracted text into structured blocks (headings, paragraphs,
    list items, tables) so downstream chunking can make structure-aware
    decisions (e.g. keep a heading with its following paragraph, chunk
    tables as atomic units).
    """

    # Markdown-style heading: "# Title", "## Subtitle", etc.
    MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

    # Table rows produced by our loaders use " | " as a cell separator
    # (see DocxLoader._extract_table_text). A block with 2+ such lines
    # is treated as a table.
    TABLE_ROW_RE = re.compile(r".+\|.+")

    # Bullet or numbered list items: "- item", "* item", "1. item", "1) item"
    LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$")

    # A short line, all-caps or title-cased, with no trailing punctuation,
    # standing alone between blank lines — treated as an unlabeled heading
    # for sources (like docx/pdf) that don't carry markdown syntax.
    BARE_HEADING_RE = re.compile(r"^[A-Z][A-Za-z0-9 ,'&/-]{0,80}$")

    def parse(self, raw_document: RawDocument) -> ParsedDocument:
        chunks = self._split_into_chunks(raw_document.text)
        blocks = [block for chunk in chunks for block in self._classify_chunk(chunk)]
        blocks = self._merge_adjacent_paragraphs(blocks)

        return ParsedDocument(blocks=blocks, metadata=raw_document.metadata)

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split on blank lines into candidate blocks."""
        raw_chunks = re.split(r"\n\s*\n", text.strip())
        return [c.strip() for c in raw_chunks if c.strip()]

    def _classify_chunk(self, chunk: str) -> list[Block]:
        lines = chunk.splitlines()

        # Markdown heading
        md_match = self.MD_HEADING_RE.match(lines[0]) if lines else None
        if md_match and len(lines) == 1:
            level = len(md_match.group(1))
            return [Block(type=BlockType.HEADING, text=md_match.group(2).strip(), level=level)]

        # Table: multiple pipe-delimited rows
        table_lines = [ln for ln in lines if self.TABLE_ROW_RE.match(ln)]
        if len(table_lines) >= 2 and len(table_lines) == len(lines):
            return [Block(type=BlockType.TABLE, text=chunk, metadata={"num_rows": len(lines)})]

        # List: majority of lines match list-item pattern
        list_matches = [self.LIST_ITEM_RE.match(ln) for ln in lines]
        if lines and sum(1 for m in list_matches if m) >= max(1, len(lines) - 1):
            return [
                Block(type=BlockType.LIST_ITEM, text=(m.group(1) if m else ln).strip())
                for ln, m in zip(lines, list_matches)
            ]

        # Bare heading heuristic: single short line, no terminal punctuation,
        # title/upper case, not followed by more text in this chunk
        if len(lines) == 1 and self._looks_like_bare_heading(lines[0]):
            return [Block(type=BlockType.HEADING, text=lines[0].strip(), level=2)]

        # Default: paragraph
        return [Block(type=BlockType.PARAGRAPH, text=chunk)]

    def _looks_like_bare_heading(self, line: str) -> bool:
        line = line.strip()
        if not line or line[-1] in ".,;:":
            return False
        if len(line) > 80:
            return False
        return bool(self.BARE_HEADING_RE.match(line))

    def _merge_adjacent_paragraphs(self, blocks: list[Block]) -> list[Block]:
        """Merge consecutive PARAGRAPH blocks that were split only by a
        stray blank line, avoiding over-fragmentation before chunking."""
        merged: list[Block] = []
        for block in blocks:
            if (
                merged
                and merged[-1].type == BlockType.PARAGRAPH
                and block.type == BlockType.PARAGRAPH
            ):
                merged[-1] = Block(
                    type=BlockType.PARAGRAPH,
                    text=merged[-1].text + "\n\n" + block.text,
                )
            else:
                merged.append(block)
        return merged