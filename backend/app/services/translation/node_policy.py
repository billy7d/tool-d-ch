from typing import Any, Iterable, Set

from app.models.canonical import NodeType


TRANSLATABLE_NODE_TYPES: Set[NodeType] = {
    NodeType.HEADING,
    NodeType.PARAGRAPH,
    NodeType.QUOTE,
    NodeType.LIST,
    NodeType.LIST_ITEM,
    NodeType.CAPTION,
    NodeType.TABLE,
    NodeType.FOOTNOTE,
    NodeType.FOOTNOTE_REFERENCE,
}

NON_TRANSLATABLE_NODE_TYPES: Set[NodeType] = set(NodeType) - TRANSLATABLE_NODE_TYPES


def normalize_node_type(value: Any) -> NodeType:
    if isinstance(value, NodeType):
        return value
    try:
        return NodeType(str(value or "paragraph").lower())
    except ValueError:
        return NodeType.PARAGRAPH


def is_translatable_node_type(value: Any) -> bool:
    return normalize_node_type(value) in TRANSLATABLE_NODE_TYPES


def translatable_values() -> Iterable[str]:
    return tuple(node_type.value for node_type in TRANSLATABLE_NODE_TYPES)
