"""Absolute-position probability windows shared by target and draft caches."""

from __future__ import annotations

import torch


def append_rows(previous, previous_start: int, rows, start: int, keep_from: int):
    """Append contiguous rows, copying only the live interval.

    Clone sliced views: retaining a one-row view of prompt logits would still
    keep the entire prompt/batch allocation alive.
    """
    start, keep_from = int(start), max(0, int(keep_from))
    end = start + rows.shape[1]
    first_new = max(start, keep_from)
    fresh = rows[:, max(0, first_new - start):, :]
    if previous is not None and keep_from < start:
        previous_end = previous_start + previous.shape[1]
        if previous_end != start or keep_from < previous_start:
            raise ValueError("probability window is missing a required boundary")
        old = previous[:, keep_from - previous_start:, :]
        return torch.cat((old, fresh), dim=1), keep_from
    return fresh.clone(), min(first_new, end)


def select_rows(rows, offset: int, start: int, end: int):
    if rows is None or start < offset or end > offset + rows.shape[1] or end < start:
        raise ValueError(f"probability rows [{start}, {end}) are outside the retained window")
    return rows[:, start - offset:end - offset, :]


def crop_rows(rows, offset: int, end: int, *, boundary_only: bool = False):
    if rows is None:
        return None, int(end)
    end = min(int(end), offset + rows.shape[1])
    start = max(offset, end - 1) if boundary_only else offset
    if end <= start:
        return None, int(end)
    return rows[:, start - offset:end - offset, :].clone(), start
