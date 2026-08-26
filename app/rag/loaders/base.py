from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RawDocument:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded or parsed."""


class BaseLoader(ABC):
    SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")

    @abstractmethod
    def load(self, file_path: str | Path) -> RawDocument:
        ...

    def _validate_path(self, path: Path) -> None:
        if not path.exists():
            raise DocumentLoadError(f"File not found: {path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(
                f"Unsupported file type for {self.__class__.__name__}: {path.suffix}"
            )