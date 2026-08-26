# cleaner.py
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from rag.parser import Block, BlockType, ParsedDocument


class TextCleaner:
    """
    Cleans text extracted from parsed documents before chunking/embedding.

    Structure-aware:
      - TABLE blocks are cleaned conservatively (no whitespace collapsing
        that could disturb the " | " cell alignment loaders produce).
      - Boilerplate detection operates on whole blocks, not raw lines,
        since the parser already splits on blank lines -- a repeating
        header/footer becomes its own standalone block on every page.
      - TABLE blocks are never dropped as boilerplate.
    """

    _PAGE_NUM_PATTERNS = [
        re.compile(r'^\s*page\s+\d+\s*(of\s+\d+)?\s*$', re.IGNORECASE),
        re.compile(r'^\s*\d+\s*/\s*\d+\s*$'),
        re.compile(r'^\s*-\s*\d+\s*-\s*$'),
        re.compile(r'^\s*\d+\s*$'),
    ]

    _ENCODING_FIXES = {
        'â€™': "'", 'â€˜': "'", 'â€œ': '"', 'â€': '"',
        'â€"': '–', 'â€"': '—', 'â€¦': '…',
        'Â ': ' ', 'Ã©': 'é', 'Ã¨': 'è', 'Ã ': 'à',
        'ï¬': 'fi', 'ï¬‚': 'fl',
        '\ufeff': '',
        '\u00a0': ' ',
    }

    _BOILERPLATE_MAX_LEN = 80
    _BOILERPLATE_MIN_OCCURRENCES = 3

    def clean(self, parsed_document: ParsedDocument) -> ParsedDocument:
        boilerplate_texts = self._find_repeated_blocks(parsed_document.blocks)

        cleaned_blocks: list[Block] = []
        for block in parsed_document.blocks:
            block.text = self.remove_control_chars(block.text)
            block.text = self.fix_encoding_artifacts(block.text)
            block.text = self.remove_extra_whitespace(block.text, block.type)

            if self._is_boilerplate(block, boilerplate_texts):
                continue

            cleaned_blocks.append(block)

        parsed_document.blocks = self.drop_empty(cleaned_blocks)
        return parsed_document

    # ------------------------------------------------------------------
    @staticmethod
    def remove_extra_whitespace(text: str, block_type: BlockType = BlockType.PARAGRAPH) -> str:
        if not text:
            return text

        if block_type == BlockType.TABLE:
            # Only strip trailing whitespace per row -- collapsing internal
            # spaces could shift the " | " cell boundaries.
            return '\n'.join(line.rstrip() for line in text.split('\n')).strip()

        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text.strip()

    # ------------------------------------------------------------------
    @staticmethod
    def remove_control_chars(text: str) -> str:
        if not text:
            return text
        return ''.join(
            ch for ch in text
            if ch in ('\n', '\t') or unicodedata.category(ch) != 'Cc'
        )

    # ------------------------------------------------------------------
    @classmethod
    def fix_encoding_artifacts(cls, text: str) -> str:
        if not text:
            return text
        if 'Ã' in text or 'â€' in text:
            try:
                text = text.encode('latin-1').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        for bad, good in cls._ENCODING_FIXES.items():
            if bad in text:
                text = text.replace(bad, good)
        return unicodedata.normalize('NFKC', text)

    # ------------------------------------------------------------------
    @classmethod
    def _is_boilerplate(cls, block: Block, repeated_texts: set) -> bool:
        if block.type == BlockType.TABLE:
            return False
        stripped = block.text.strip()
        if not stripped:
            return False
        if '\n' not in stripped and cls._is_page_number(stripped):
            return True
        return stripped in repeated_texts

    @classmethod
    def _is_page_number(cls, text: str) -> bool:
        return any(p.match(text) for p in cls._PAGE_NUM_PATTERNS)

    @classmethod
    def _find_repeated_blocks(cls, blocks: list[Block]) -> set:
        counter = Counter()
        for block in blocks:
            if block.type == BlockType.TABLE:
                continue
            stripped = block.text.strip()
            if stripped and len(stripped) <= cls._BOILERPLATE_MAX_LEN:
                counter[stripped] += 1
        return {text for text, n in counter.items() if n >= cls._BOILERPLATE_MIN_OCCURRENCES}

    # ------------------------------------------------------------------
    @staticmethod
    def drop_empty(blocks: list[Block]) -> list[Block]:
        return [b for b in blocks if b.text and b.text.strip()]