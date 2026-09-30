#!/usr/bin/env python3
"""
Metric implementations for the hierarchical-labeling benchmark (README §5).

Three levels:
  a. Final label   : Exact Accuracy, Macro Recall, MAE, Quadratic Weighted Kappa (QWK)
  b. Local node    : per-node + conditional precision/recall/F1/accuracy
  c. Path          : Levenshtein path edit distance (+ optional Brier score)

Pure Python + numpy, no scikit-learn dependency.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from gold import (
    ADVANCING_VERDICTS,
    NODES,
    UNREACHED,
    verdict_at,
)

LABEL_MIN = 1
LABEL_MAX = 7


# ─────────────────────────────────────────────────────────────────────────────
# a. Final-label level
# ─────────────────────────────────────────────────────────────────────────────
def exact_accuracy(gold: list[int], pred: list[int]) -> float:
    if not gold:
        return float("nan")
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return correct / len(gold)


def mae(gold: list[int], pred: list[int]) -> float:
    if not gold:
        return float("nan")
    return float(np.mean([abs(g - p) for g, p in zip(gold, pred)]))


def quadratic_weighted_kappa(
    gold: list[int],
    pred: list[int],
    min_rating: int = LABEL_MIN,
    max_rating: int = LABEL_MAX,
) -> float:
    """
    QWK over a fixed rating range. Quadratic weights w_ij = (i-j)^2 / (N-1)^2.
    Returns NaN when undefined (no samples, or expected agreement == observed).
    """
    if not gold:
        return float("nan")

    n_ratings = max_rating - min_rating + 1
    idx = {r: i for i, r in enumerate(range(min_rating, max_rating + 1))}

    O = np.zeros((n_ratings, n_ratings), dtype=float)
    for g, p in zip(gold, pred):
        # Clamp out-of-range predictions into the valid window.
        gi = idx.get(g, max(0, min(n_ratings - 1, g - min_rating)))
        pi = idx.get(p, max(0, min(n_ratings - 1, p - min_rating)))
        O[gi, pi] += 1

    W = np.zeros((n_ratings, n_ratings), dtype=float)
    denom = (n_ratings - 1) ** 2 if n_ratings > 1 else 1
    for i in range(n_ratings):
        for j in range(n_ratings):
            W[i, j] = ((i - j) ** 2) / denom

    gold_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(gold_hist, pred_hist) / O.sum()

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return float("nan")
    return float(1.0 - num / den)


def macro_recall(gold: list[int], pred: list[int]) -> float:
    """
    Macro-averaged recall over the label classes present in gold (1–7).

    For a multiclass final label, micro-recall would equal exact accuracy, so
    the informative "overall recall" is the unweighted mean of per-class recall
    (each class contributes equally, regardless of support).
    """
    if not gold:
        return float("nan")
    recalls = []
    for c in sorted(set(gold)):
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        support = sum(1 for g in gold if g == c)
        recalls.append(tp / support if support else 0.0)
    return float(np.mean(recalls)) if recalls else float("nan")


def label_metrics(gold: list[int], pred: list[int]) -> dict:
    return {
        "n": len(gold),
        "exact_accuracy": exact_accuracy(gold, pred),
        "macro_recall": macro_recall(gold, pred),
        "mae": mae(gold, pred),
        "qwk": quadratic_weighted_kappa(gold, pred),
    }


# ─────────────────────────────────────────────────────────────────────────────
# b. Local-node level
# ─────────────────────────────────────────────────────────────────────────────
def classification_metrics(
    gold: list[str],
    pred: list[str],
    positive_class: Optional[str] = None,
) -> dict:
    """
    Per-class precision/recall/F1 + accuracy + macro averages.
    If `positive_class` is given, that class's metrics are surfaced as `positive_*`.
    """
    n = len(gold)
    out: dict = {"n": n, "per_class": {}}
    if n == 0:
        out.update({"accuracy": float("nan"), "macro_f1": float("nan")})
        return out

    out["accuracy"] = sum(1 for g, p in zip(gold, pred) if g == p) / n

    classes = sorted(set(gold) | set(pred))
    f1s, precisions, recalls = [], [], []
    for c in classes:
        tp = sum(1 for g, p in zip(gold, pred) if p == c and g == c)
        fp = sum(1 for g, p in zip(gold, pred) if p == c and g != c)
        fn = sum(1 for g, p in zip(gold, pred) if p != c and g == c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        support = sum(1 for g in gold if g == c)
        out["per_class"][c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        # Macro averages only over real (gold) classes, excluding the UNREACHED sentinel.
        if c != UNREACHED:
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)

    out["macro_precision"] = float(np.mean(precisions)) if precisions else float("nan")
    out["macro_recall"] = float(np.mean(recalls)) if recalls else float("nan")
    out["macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")

    if positive_class is not None:
        pc = out["per_class"].get(
            positive_class,
            {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
        )
        out["positive_class"] = positive_class
        out["positive_precision"] = pc["precision"]
        out["positive_recall"] = pc["recall"]
        out["positive_f1"] = pc["f1"]

    return out


def _node_pairs(
    gold_paths: list[list[tuple[str, str]]],
    pred_paths: list[list[tuple[str, str]]],
    node: str,
    conditional: bool,
) -> tuple[list[str], list[str]]:
    """
    Build (gold_verdict, pred_verdict) lists for one node.

    conditional=False : population is all samples (UNREACHED used on either side).
    conditional=True  : population is samples whose GOLD path reaches the node.
    """
    g_list, p_list = [], []
    for gp, pp in zip(gold_paths, pred_paths):
        gv = verdict_at(gp, node)
        if conditional and gv == UNREACHED:
            continue
        pv = verdict_at(pp, node)
        g_list.append(gv)
        p_list.append(pv)
    return g_list, p_list


def node_metrics(
    gold_paths: list[list[tuple[str, str]]],
    pred_paths: list[list[tuple[str, str]]],
) -> dict:
    """Per-node metrics, both unconditional and conditional (gold-reached)."""
    result: dict = {}
    for node in NODES:
        # Binary nodes: 'fail' is the positive (error-detected) class. L4: macro only.
        positive = "fail" if node != "L4" else None

        g_u, p_u = _node_pairs(gold_paths, pred_paths, node, conditional=False)
        g_c, p_c = _node_pairs(gold_paths, pred_paths, node, conditional=True)

        result[node] = {
            "unconditional": classification_metrics(g_u, p_u, positive),
            "conditional": classification_metrics(g_c, p_c, positive),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# c. Path level
# ─────────────────────────────────────────────────────────────────────────────
def levenshtein(a: list[str], b: list[str]) -> int:
    """Edit distance (insert/delete/substitute, unit cost) between two token lists."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = cur
    return prev[lb]


def path_metrics(
    gold_token_paths: list[list[str]],
    pred_token_paths: list[list[str]],
) -> dict:
    if not gold_token_paths:
        return {"n": 0, "mean_edit_distance": float("nan"), "mean_normalized": float("nan")}
    dists, norms = [], []
    for g, p in zip(gold_token_paths, pred_token_paths):
        d = levenshtein(g, p)
        dists.append(d)
        denom = max(len(g), len(p), 1)
        norms.append(d / denom)
    return {
        "n": len(dists),
        "mean_edit_distance": float(np.mean(dists)),
        "mean_normalized": float(np.mean(norms)),
        "exact_path_match_rate": float(np.mean([1.0 if d == 0 else 0.0 for d in dists])),
    }


def _advanced(verdict: str) -> int:
    """Binary outcome: did the pipeline advance past this layer? (1=yes, 0=stop)."""
    return 1 if verdict in ADVANCING_VERDICTS else 0


def brier_metrics(
    gold_paths: list[list[tuple[str, str]]],
    pred_probs: list[dict[str, float]],
) -> dict:
    """
    Per-layer Brier score over samples where GOLD reached the layer and a
    predicted pass/advance probability is available.

      outcome  = 1 if gold advanced past the layer else 0
      brier    = mean((prob - outcome)^2)
    """
    per_layer_sq: dict[str, list[float]] = defaultdict(list)
    for gp, probs in zip(gold_paths, pred_probs or []):
        if not probs:
            continue
        for node, verdict in gp:
            p = probs.get(node)
            if p is None:
                continue
            outcome = _advanced(verdict)
            per_layer_sq[node].append((float(p) - outcome) ** 2)

    per_layer = {
        node: {"brier": float(np.mean(sqs)), "n": len(sqs)}
        for node, sqs in per_layer_sq.items()
    }
    all_sqs = [s for sqs in per_layer_sq.values() for s in sqs]
    return {
        "overall_brier": float(np.mean(all_sqs)) if all_sqs else float("nan"),
        "n": len(all_sqs),
        "per_layer": per_layer,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate
# ─────────────────────────────────────────────────────────────────────────────
def compute_all(
    gold_labels: list[int],
    pred_labels: list[int],
    gold_paths: list[list[tuple[str, str]]],
    pred_paths: list[list[tuple[str, str]]],
    gold_token_paths: list[list[str]],
    pred_token_paths: list[list[str]],
    types: Optional[list[str]] = None,
    pred_probs: Optional[list[dict[str, float]]] = None,
) -> dict:
    out: dict = {
        "label": label_metrics(gold_labels, pred_labels),
        "node": node_metrics(gold_paths, pred_paths),
        "path": path_metrics(gold_token_paths, pred_token_paths),
    }

    if pred_probs and any(pred_probs):
        out["brier"] = brier_metrics(gold_paths, pred_probs)

    # Per-type label slices.
    if types:
        by_type: dict[str, dict] = {}
        groups: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(types):
            groups[t].append(i)
        for t, idxs in sorted(groups.items()):
            g = [gold_labels[i] for i in idxs]
            p = [pred_labels[i] for i in idxs]
            by_type[t] = label_metrics(g, p)
        out["by_type"] = by_type

    return out
