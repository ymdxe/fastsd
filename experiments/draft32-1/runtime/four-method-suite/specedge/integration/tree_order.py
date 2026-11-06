"""Ordering helpers for the SpecEdge integration tree adapter."""

from __future__ import annotations

from typing import Any


def order_tree_compaction_indices(src_indices: Any) -> Any:
    """Keep Tree.gather's selected nodes in their append/topological order.

    ``torch.topk(..., sorted=False)`` selects the right nodes but is allowed to
    return them in arbitrary flat-index order.  ``Tree.gather`` remaps parent
    indices according to that order, so compacting a child before its parent
    creates forward parent references and makes later path extraction emit tree
    storage order.  Tree.add appends children after their parents; ascending
    the original indices therefore preserves the existing tree topology without
    changing which candidates were selected.
    """

    ordered_indices, _ = src_indices.sort()
    return ordered_indices
