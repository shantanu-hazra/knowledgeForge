from __future__ import annotations

from pathlib import Path

from rag.loaders.base import RawDocument, DocumentLoadError, BaseLoader


class TxtLoader(BaseLoader):
    """Loads plain text / markdown files into a RawDocument."""

    SUPPORTED_EXTENSIONS = (".txt", ".md")

    # Tried in order; utf-8 covers the vast majority of files, the rest
    # are common fallbacks for files saved by older/Windows tools.
    ENCODING_FALLBACKS = ("utf-8", "utf-8-sig", "latin-1")

    def load(self, file_path: str | Path) -> RawDocument:
        path = Path(file_path)
        self._validate_path(path)

        text, used_encoding = self._read_with_fallback(path)

        if not text.strip():
            raise DocumentLoadError(f"File is empty: {path}")

        metadata = self._build_metadata(path, used_encoding)

        return RawDocument(text=text, metadata=metadata)

    def _read_with_fallback(self, path: Path) -> tuple[str, str]:
        last_error: Exception | None = None

        for encoding in self.ENCODING_FALLBACKS:
            try:
                text = path.read_text(encoding=encoding)
                return text, encoding
            except (UnicodeDecodeError, UnicodeError) as e:
                last_error = e
                continue
            except Exception as e:
                raise DocumentLoadError(f"Failed to read file {path}: {e}") from e

        raise DocumentLoadError(
            f"Could not decode {path} with any of {self.ENCODING_FALLBACKS}: {last_error}"
        )

    def _build_metadata(self, path: Path, encoding: str) -> dict[str, str]:
        return {
            "source": str(path),
            "file_type": "md" if path.suffix.lower() == ".md" else "txt",
            "file_name": path.name,
            "encoding_used": encoding,
        }