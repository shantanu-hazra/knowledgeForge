from __future__ import annotations

from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag.loaders.base import RawDocument, DocumentLoadError, BaseLoader

class PDFLoader(BaseLoader):
    """Loads .pdf files into a RawDocument, extracting per-page text."""

    SUPPORTED_EXTENSIONS = (".pdf",)

    def load(self, file_path: str | Path) -> RawDocument:
        path = Path(file_path)

        if not path.exists():
            raise DocumentLoadError(f"File not found: {path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(f"Unsupported file type for PDFLoader: {path.suffix}")

        try:
            reader = PdfReader(str(path))
        except (PdfReadError, Exception) as e:
            raise DocumentLoadError(f"Failed to open PDF file {path}: {e}") from e

        if reader.is_encrypted:
            try:
                # Empty-password decrypt handles PDFs that are "encrypted"
                # with no real password (common from some export tools).
                reader.decrypt("")
            except Exception as e:
                raise DocumentLoadError(f"PDF is encrypted and could not be decrypted: {path}") from e

        text_pages, failed_pages = self._extract_pages(reader, path)

        full_text = "\n\n".join(text_pages)
        if not full_text.strip():
            raise DocumentLoadError(f"No extractable text found in {path} (may be scanned/image-based)")

        metadata = self._build_metadata(path, reader, failed_pages)

        return RawDocument(text=full_text, metadata=metadata)

    def _extract_pages(self, reader: PdfReader, path: Path) -> tuple[list[str], list[int]]:
        """
        Extracts text page by page. A single bad page shouldn't kill the
        whole document — collect failures instead of raising immediately.
        """
        text_pages: list[str] = []
        failed_pages: list[int] = []

        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if page_text:
                    text_pages.append(page_text)
            except Exception:
                failed_pages.append(i + 1)  # 1-indexed for readability

        return text_pages, failed_pages

    def _build_metadata(
        self, path: Path, reader: PdfReader, failed_pages: list[int]
    ) -> dict[str, Any]:
        doc_info = reader.metadata or {}
        return {
            "source": str(path),
            "file_type": "pdf",
            "file_name": path.name,
            "num_pages": len(reader.pages),
            "title": doc_info.get("/Title") or None,
            "author": doc_info.get("/Author") or None,
            "failed_pages": failed_pages or None,
        }