from typing import List, Dict, Any
from app.models.canonical import DocumentNode, NodeType


class SemanticChunk:
    def __init__(self, chunk_id: str, nodes: List[DocumentNode], estimated_tokens: int = 0):
        self.chunk_id = chunk_id
        self.nodes = nodes
        self.estimated_tokens = estimated_tokens


class SemanticChunker:
    @staticmethod
    def estimate_tokens(text: str) -> int:
        # Rough heuristic: ~1.3 tokens per English word + punctuation
        words = text.split()
        return max(1, int(len(words) * 1.3))

    @classmethod
    def chunk_nodes(cls, nodes: List[DocumentNode], target_tokens: int = 750, max_tokens: int = 1200) -> List[SemanticChunk]:
        """
        Groups semantic document nodes into coherent batches without splitting paragraphs.
        Optimized for high throughput and context cohesion.
        """

        chunks: List[SemanticChunk] = []
        current_batch: List[DocumentNode] = []
        current_tokens = 0
        chunk_idx = 1

        for node in nodes:
            # Skip non-translatable nodes like images, equations, horizontal rules
            if node.type in [NodeType.IMAGE, NodeType.HORIZONTAL_RULE, NodeType.PAGE_BREAK_HINT]:
                continue

            node_tokens = cls.estimate_tokens(node.content)

            # If node is a heading and we already have some nodes, start a new chunk for clean section boundary
            if node.type == NodeType.HEADING and current_batch and current_tokens >= (target_tokens * 0.5):
                chunks.append(SemanticChunk(
                    chunk_id=f"chunk_{chunk_idx:05d}",
                    nodes=current_batch,
                    estimated_tokens=current_tokens
                ))
                chunk_idx += 1
                current_batch = [node]
                current_tokens = node_tokens
                continue

            # If adding this node exceeds target token budget, flush current batch
            if current_batch and (current_tokens + node_tokens) > target_tokens:
                chunks.append(SemanticChunk(
                    chunk_id=f"chunk_{chunk_idx:05d}",
                    nodes=current_batch,
                    estimated_tokens=current_tokens
                ))
                chunk_idx += 1
                current_batch = [node]
                current_tokens = node_tokens
            else:
                current_batch.append(node)
                current_tokens += node_tokens

        if current_batch:
            chunks.append(SemanticChunk(
                chunk_id=f"chunk_{chunk_idx:05d}",
                nodes=current_batch,
                estimated_tokens=current_tokens
            ))

        return chunks
