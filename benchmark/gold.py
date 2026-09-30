#!/usr/bin/env python3
"""
Gold derivation for the hierarchical-labeling benchmark.

CSEG-Bench records carry `type` and `target_label`. Optional `actual_label` /
`actual_layer_verdicts` are present when a record was re-scored by the six-layer
pipeline. This module turns either source into a canonical (gold_label, gold_path)
pair, where a *path* is the ordered list of (layer_id, verdict) the pipeline
walks before stopping.

The `type -> path` table mirrors the routing in `data_filter/prompt_config.json`.
"""

from __future__ import annotations

from typing import Optional

# Canonical walk order of the pipeline. L5_consensus and L5_conflict are mutually
# exclusive (chosen by L4's verdict), so filtering this list against a verdict
# dict reconstructs the actual walk order.
LAYER_ORDER: list[str] = [
    "L1",
    "L2",
    "L3",
    "L4",
    "L5_consensus",
    "L5_conflict",
    "L6",
]

# Nodes that are scored individually at the local-node level.
NODES: list[str] = LAYER_ORDER

# Verdict vocabulary per node (for reference / validation).
NODE_VERDICTS: dict[str, list[str]] = {
    "L1": ["pass", "fail"],
    "L2": ["pass", "fail"],
    "L3": ["pass", "fail"],
    "L4": ["consistent", "conflict_identified", "conflict_missed"],
    "L5_consensus": ["pass", "fail"],
    "L5_conflict": ["pass", "fail"],
    "L6": ["pass", "fail"],
}

# A verdict is "advancing" if the pipeline continues to the next layer afterwards.
# Used both for path reconstruction sanity and for the Brier-score outcome.
ADVANCING_VERDICTS: set[str] = {"pass", "consistent", "conflict_identified"}

# Sentinel verdict for "the walk never reached this node".
UNREACHED = "—"

# type -> canonical gold path (list of (layer_id, verdict)).
GOLD_PATHS: dict[str, list[tuple[str, str]]] = {
    "L1_fail": [("L1", "fail")],
    "L2_fail": [("L1", "pass"), ("L2", "fail")],
    "L3_fail": [("L1", "pass"), ("L2", "pass"), ("L3", "fail")],
    "L4_conflict_missed": [
        ("L1", "pass"), ("L2", "pass"), ("L3", "pass"), ("L4", "conflict_missed"),
    ],
    "L5_consensus_fail": [
        ("L1", "pass"), ("L2", "pass"), ("L3", "pass"),
        ("L4", "consistent"), ("L5_consensus", "fail"),
    ],
    "L5_conflict_fail": [
        ("L1", "pass"), ("L2", "pass"), ("L3", "pass"),
        ("L4", "conflict_identified"), ("L5_conflict", "fail"),
    ],
    "L6_fail_consensus": [
        ("L1", "pass"), ("L2", "pass"), ("L3", "pass"),
        ("L4", "consistent"), ("L5_consensus", "pass"), ("L6", "fail"),
    ],
    "L6_fail_conflict": [
        ("L1", "pass"), ("L2", "pass"), ("L3", "pass"),
        ("L4", "conflict_identified"), ("L5_conflict", "pass"), ("L6", "fail"),
    ],
    "L7_pass_consensus": [
        ("L1", "pass"), ("L2", "pass"), ("L3", "pass"),
        ("L4", "consistent"), ("L5_consensus", "pass"), ("L6", "pass"),
    ],
    "L7_pass_conflict": [
        ("L1", "pass"), ("L2", "pass"), ("L3", "pass"),
        ("L4", "conflict_identified"), ("L5_conflict", "pass"), ("L6", "pass"),
    ],
}


def path_from_verdicts(verdicts: dict[str, str]) -> list[tuple[str, str]]:
    """Reconstruct an ordered path from a flat {layer_id: verdict} dict."""
    return [(lid, verdicts[lid]) for lid in LAYER_ORDER if lid in verdicts]


def path_to_tokens(path: list[tuple[str, str]]) -> list[str]:
    """Render a path as a token list, e.g. ['L1:pass', 'L2:fail']."""
    return [f"{lid}:{verdict}" for lid, verdict in path]


def verdict_at(path: list[tuple[str, str]], node: str) -> str:
    """Return the verdict at `node`, or the UNREACHED sentinel if not in path."""
    for lid, verdict in path:
        if lid == node:
            return verdict
    return UNREACHED


def gold_from_record(
    record: dict,
    gold_source: str,
    label_mapping: Optional[dict] = None,
) -> tuple[Optional[int], list[tuple[str, str]]]:
    """
    Derive (gold_label, gold_path) for one eval record.

    gold_source:
      'target' -> use record['target_label'] and GOLD_PATHS[record['type']]
      'actual' -> use record['actual_label'] and record['actual_layer_verdicts']
    """
    if gold_source == "target":
        gtype = record.get("type")
        if gtype not in GOLD_PATHS:
            raise ValueError(f"Unknown record type '{gtype}' (no canonical gold path).")
        gold_path = GOLD_PATHS[gtype]
        gold_label = record.get("target_label")
        return gold_label, gold_path

    if gold_source == "actual":
        verdicts = record.get("actual_layer_verdicts") or {}
        if not verdicts:
            raise ValueError(
                "gold_source='actual' but record has no 'actual_layer_verdicts' "
                "(record was not re-scored by the six-layer pipeline)."
            )
        gold_path = path_from_verdicts(verdicts)
        gold_label = record.get("actual_label")
        return gold_label, gold_path

    raise ValueError(f"Unknown gold_source: {gold_source!r} (expected 'target' or 'actual').")
