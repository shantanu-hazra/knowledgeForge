# loaders/registry.py
from __future__ import annotations

from pathlib import Path

from rag.loaders.base import BaseLoader, DocumentLoadError


class LoaderRegistry:
    """Maps file extensions to the loader that handles them."""

    def __init__(self, loaders: list[BaseLoader]):
        self._by_ext: dict[str, BaseLoader] = {}
        for loader in loaders:
            for ext in loader.SUPPORTED_EXTENSIONS:
                self._by_ext[ext.lower()] = loader

    def get_loader(self, file_path: str | Path) -> BaseLoader:
        path = Path(file_path)
        loader = self._by_ext.get(path.suffix.lower())
        if loader is None:
            raise DocumentLoadError(f"No loader registered for extension: {path.suffix}")
        return loader