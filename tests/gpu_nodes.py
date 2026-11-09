"""Parse ``pytest --collect-only -q`` output into node ids."""

from __future__ import annotations


def node_ids_from_collect_output(text: str) -> list[str]:
    nodes: list[str] = []
    for raw in text.splitlines():
        node = raw.split()[0] if raw.split() else ""
        if node.startswith("tests/") and "::" in node:
            nodes.append(node)
    return nodes
