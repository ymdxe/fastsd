"""Auditable token-level parity reports for the unified workload."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .run_artifacts import require_unique_sample_ids, write_json_once, write_text_once


FOUR_METHODS = ("fastsd", "specedge_cpu_adapted", "standard_sd", "draft_only")
ORACLE_METHOD = "target_only"
EXACT_GATE_METHODS = ("fastsd", "specedge_cpu_adapted", "standard_sd")


def _token_ids(record: dict[str, Any]) -> list[int]:
    values = record.get("output_token_ids", record.get("token_ids", []))
    return [int(value) for value in (values or [])]


def _records_by_sample(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    items = list(records)
    result = {str(record["sample_id"]): record for record in items}
    if len(result) != len(items):
        raise ValueError("duplicate sample_id in parity records")
    require_unique_sample_ids(result.values())
    return result


def build_token_parity_rows(
    records_by_method: dict[str, Iterable[dict[str, Any]]],
    *,
    reference_method: str = ORACLE_METHOD,
) -> list[dict[str, Any]]:
    """Build one JSON object per sample and token position.

    A missing position is represented by ``null``.  This makes early EOS or a
    stop-policy mismatch visible instead of silently comparing only the common
    prefix.
    """

    if reference_method not in records_by_method:
        raise ValueError(f"reference method is missing: {reference_method}")
    indexed = {
        method: _records_by_sample(list(records))
        for method, records in records_by_method.items()
    }
    reference_ids = list(indexed[reference_method])
    rows: list[dict[str, Any]] = []
    for sample_id in reference_ids:
        token_lists = {
            method: _token_ids(indexed[method].get(sample_id, {}))
            for method in indexed
        }
        max_len = max((len(values) for values in token_lists.values()), default=0)
        workload_hashes = {
            str(indexed[method][sample_id].get("workload_hash"))
            for method in indexed
            if sample_id in indexed[method]
            and indexed[method][sample_id].get("workload_hash") is not None
        }
        workload_hash = next(iter(workload_hashes)) if len(workload_hashes) == 1 else None
        for position in range(max_len):
            token_ids = {
                method: values[position] if position < len(values) else None
                for method, values in token_lists.items()
            }
            reference_token = token_ids[reference_method]
            rows.append(
                {
                    "sample_id": sample_id,
                    "global_index": indexed[reference_method][sample_id].get("global_index"),
                    "position": position,
                    "reference_method": reference_method,
                    "reference_token_id": reference_token,
                    "token_ids": token_ids,
                    "equal_to_reference": {
                        method: token == reference_token and token is not None
                        for method, token in token_ids.items()
                        if method != reference_method
                    },
                    "workload_hash": workload_hash,
                }
            )
    return rows


def summarize_token_parity(
    rows: Iterable[dict[str, Any]],
    *,
    methods: Iterable[str] | None = None,
    reference_method: str = ORACLE_METHOD,
) -> dict[str, Any]:
    rows_list = list(rows)
    method_names = list(methods or (next(iter(rows_list), {}).get("token_ids", {}).keys()))
    result: dict[str, Any] = {
        "reference_method": reference_method,
        "total_token_positions": len(rows_list),
        "methods": {},
    }
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_list:
        by_sample[str(row["sample_id"])].append(row)
    for method in method_names:
        if method == reference_method:
            continue
        matches = sum(bool(row.get("equal_to_reference", {}).get(method)) for row in rows_list)
        comparable = sum(
            row.get("token_ids", {}).get(method) is not None
            and row.get("reference_token_id") is not None
            for row in rows_list
        )
        exact_sequences = 0
        first_mismatches: list[int] = []
        for sample_rows in by_sample.values():
            mismatch = next(
                (
                    int(row["position"])
                    for row in sample_rows
                    if not row.get("equal_to_reference", {}).get(method, False)
                ),
                None,
            )
            if mismatch is None:
                exact_sequences += 1
            else:
                first_mismatches.append(mismatch)
        result["methods"][method] = {
            "matching_positions": matches,
            "comparable_positions": comparable,
            "token_parity_rate": matches / comparable if comparable else None,
            "exact_sequence_count": exact_sequences,
            "sample_count": len(by_sample),
            "first_mismatch_position_avg": (
                sum(first_mismatches) / len(first_mismatches) if first_mismatches else None
            ),
        }
    return result


def exact_gate_summary(
    rows: Iterable[dict[str, Any]],
    *,
    methods: Iterable[str] = EXACT_GATE_METHODS,
    reference_method: str = ORACLE_METHOD,
    expected_sample_ids: Iterable[str] | None = None,
    sample_ids_by_method: dict[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Return an auditable exact-token gate for target-based methods.

    ``draft_only`` may be present in the report, but it is intentionally not
    included in the enforced method list.  Missing tails, missing samples, and
    reference tails are all failures for an enforced method.
    """

    rows_list = list(rows)
    method_names = list(methods)
    expected_ids = {str(sample_id) for sample_id in (expected_sample_ids or ())}
    observed_methods = set()
    for row in rows_list:
        observed_methods.update(str(key) for key in row.get("token_ids", {}))

    results: dict[str, Any] = {}
    for method in method_names:
        missing = 0
        mismatch = 0
        reference_missing = 0
        mismatching_samples: set[str] = set()
        available_ids = {
            str(sample_id)
            for sample_id in (sample_ids_by_method or {}).get(method, ())
        }
        if expected_ids and available_ids:
            missing_samples = expected_ids - available_ids
            extra_samples = available_ids - expected_ids
        elif expected_ids and method not in (sample_ids_by_method or {}):
            missing_samples = set(expected_ids)
            extra_samples = set()
        else:
            missing_samples = set()
            extra_samples = set()
        mismatching_samples.update(missing_samples | extra_samples)
        for row in rows_list:
            token_ids = row.get("token_ids", {})
            token = token_ids.get(method)
            reference_token = row.get("reference_token_id")
            if reference_token is None:
                reference_missing += 1
                if token is not None:
                    mismatch += 1
                    mismatching_samples.add(str(row.get("sample_id")))
            elif token is None:
                missing += 1
                mismatching_samples.add(str(row.get("sample_id")))
            elif token != reference_token:
                mismatch += 1
                mismatching_samples.add(str(row.get("sample_id")))
        method_missing = method not in observed_methods
        if method_missing and rows_list:
            missing += len(rows_list)
        gate_pass = not (
            method_missing
            or missing
            or mismatch
            or reference_missing
            or missing_samples
            or extra_samples
        )
        results[method] = {
            "gate_enforced": True,
            "gate_pass": gate_pass,
            "comparison_count": len(rows_list),
            "missing": missing,
            "missing_samples": sorted(missing_samples),
            "extra_samples": sorted(extra_samples),
            "mismatch": mismatch,
            "reference_missing": reference_missing,
            "mismatching_samples": sorted(mismatching_samples),
        }

    return {
        "reference_method": reference_method,
        "gate_methods": method_names,
        "report_only_methods": [
            method for method in FOUR_METHODS if method not in method_names
        ],
        "total_comparison_rows": len(rows_list),
        "methods": results,
        "gate_pass": all(item["gate_pass"] for item in results.values()),
    }


def write_token_parity_report(
    rows: Iterable[dict[str, Any]],
    path: str | Path,
    *,
    summary_path: str | Path | None = None,
    methods: Iterable[str] | None = None,
    reference_method: str = ORACLE_METHOD,
) -> tuple[Path, Path | None]:
    rows_list = list(rows)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows_list)
    output = write_text_once(path, content)
    summary_output = None
    if summary_path is not None:
        summary_output = write_json_once(
            summary_path,
            summarize_token_parity(rows_list, methods=methods, reference_method=reference_method),
        )
    return output, summary_output
