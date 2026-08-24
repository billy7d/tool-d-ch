"""Lớp tương thích cho các import cũ của bộ chia đoạn."""

from app.services.translation.adaptive_chunker import (
    AdaptiveSemanticChunker,
    SemanticChunk,
    SemanticChunker,
)

__all__ = ["AdaptiveSemanticChunker", "SemanticChunk", "SemanticChunker"]
