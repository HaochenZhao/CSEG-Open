#!/usr/bin/env python3
"""
Evaluate a predictions.jsonl file against gold and emit metrics + a report.

Reads the predictions produced by predict.py (gold is carried through there), so
this step never queries the model and can be re-run freely.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

from metrics import compute_all


def tokens_to_path(tokens: list[str]) -> list[tuple[str, str]]:
    path: list[tuple[str, str]] = []
    for tok in tokens:
        lid, _, verdict = tok.partition(":")
        path.append((lid, verdict))
    return path


def load_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# Types excluded from the headline metrics by default.
#
# `L6_fail_consensus` is ill-defined: only about half of its samples re-score to
# the intended label, so its gold is mostly noise. CSEG-RM training drops it, and
# the benchmark uses the same exclusion so train and eval stay aligned.
DEFAULT_EXCLUDE_TYPES: tuple[str, ...] = ("L6_fail_consensus",)


def contested_breakdown(predictions: list[dict]) -> Optional[dict]:
    """Accuracy split by whether the re-scoring judge agreed with the construction target.

    Gold is `target_label` regardless (see `--gold-source`), so samples the strong judge
    disagreed with are scored against the generator's intent. That is a defensible
    choice — the judge is fallible too — but it has to be reported rather than hidden,
    and it is the one place where the benchmark and the CSEG-RM training set
    diverge: the GRPO split drops these samples entirely.
    """
    tagged = [p for p in predictions if p.get("match") is not None]
    if not tagged:
        return None
    out: dict = {"n_tagged": len(tagged)}
    for key, subset in (("agreed", [p for p in tagged if p["match"]]),
                        ("contested", [p for p in tagged if not p["match"]])):
        if not subset:
            out[key] = {"n": 0}
            continue
        correct = sum(1 for p in subset if p["gold_label"] == p["pred_label"])
        out[key] = {
            "n": len(subset),
            "exact_accuracy": correct / len(subset),
            "share": len(subset) / len(tagged),
        }
    return out


def evaluate(predictions: list[dict],
             exclude_types: Optional[list[str]] = None) -> dict:
    excluded = list(DEFAULT_EXCLUDE_TYPES if exclude_types is None else exclude_types)
    kept = [p for p in predictions if p.get("type") not in excluded]
    dropped = len(predictions) - len(kept)
    if not kept:
        raise ValueError(
            f"every prediction was excluded by --exclude-types {excluded}; nothing to score"
        )

    gold_labels = [p["gold_label"] for p in kept]
    pred_labels = [p["pred_label"] for p in kept]
    gold_token_paths = [p["gold_path"] for p in kept]
    pred_token_paths = [p["pred_path"] for p in kept]
    gold_paths = [tokens_to_path(p["gold_path"]) for p in kept]
    pred_paths = [tokens_to_path(p["pred_path"]) for p in kept]
    types = [p.get("type") for p in kept]
    pred_probs = [p.get("pred_probs") or {} for p in kept]

    metrics = compute_all(
        gold_labels=gold_labels,
        pred_labels=pred_labels,
        gold_paths=gold_paths,
        pred_paths=pred_paths,
        gold_token_paths=gold_token_paths,
        pred_token_paths=pred_token_paths,
        types=types,
        pred_probs=pred_probs,
    )
    metrics["scoring_scope"] = {
        "excluded_types": excluded,
        "n_predictions": len(predictions),
        "n_scored": len(kept),
        "n_excluded": dropped,
    }
    contested = contested_breakdown(kept)
    if contested is not None:
        metrics["contested"] = contested
    metrics["usage"] = usage_metrics(kept)
    return metrics


def _usage_present(usage: object) -> bool:
    if not isinstance(usage, dict):
        return False
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
        try:
            if int(usage.get(key) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def usage_metrics(predictions: list[dict]) -> dict:
    """Average per-sample token use. Each sample sums every layer judge call.

    ``completion_tokens`` includes thinking. ``reasoning_tokens`` is the
    thinking slice when the server reports it (else 0).
    """
    prompt, completion, reasoning, total = [], [], [], []
    for row in predictions:
        usage = row.get("usage")
        if not _usage_present(usage):
            continue
        prompt.append(int(usage.get("prompt_tokens") or 0))
        completion.append(int(usage.get("completion_tokens") or 0))
        reasoning.append(int(usage.get("reasoning_tokens") or 0))
        total.append(int(usage.get("total_tokens") or 0) or (prompt[-1] + completion[-1]))
    n = len(completion)
    if n == 0:
        nan = float("nan")
        return {
            "n_with_usage": 0,
            "mean_prompt_tokens": nan,
            "mean_completion_tokens": nan,
            "mean_reasoning_tokens": nan,
            "mean_total_tokens": nan,
        }
    return {
        "n_with_usage": n,
        "mean_prompt_tokens": float(sum(prompt) / n),
        "mean_completion_tokens": float(sum(completion) / n),
        "mean_reasoning_tokens": float(sum(reasoning) / n),
        "mean_total_tokens": float(sum(total) / n),
    }


def _fmt(x) -> str:
    if isinstance(x, float):
        return "nan" if x != x else f"{x:.4f}"
    return str(x)


def format_report(metrics: dict, model: Optional[str] = None) -> str:
    L = metrics["label"]
    P = metrics["path"]
    scope = metrics.get("scoring_scope") or {}
    lines = [
        "=" * 70,
        f"  BENCHMARK REPORT{('  —  ' + model) if model else ''}",
        "=" * 70,
        f"  Samples: {L['n']}",
    ]
    if scope.get("n_excluded"):
        lines.append(
            f"  Excluded: {scope['n_excluded']} sample(s) of type "
            f"{', '.join(scope['excluded_types'])} (same exclusion as CSEG-RM training)"
        )
    lines += [
        "",
        "  a. Final-label level",
        f"     Exact Accuracy : {_fmt(L['exact_accuracy'])}",
        f"     Macro Recall   : {_fmt(L.get('macro_recall'))}",
        f"     MAE            : {_fmt(L['mae'])}",
        f"     QWK            : {_fmt(L['qwk'])}",
        "",
        "  c. Path level",
        f"     Mean Edit Dist : {_fmt(P['mean_edit_distance'])}",
        f"     Normalized     : {_fmt(P['mean_normalized'])}",
        f"     Exact Path     : {_fmt(P['exact_path_match_rate'])}",
        "",
        "  b. Local-node level (conditional: gold reaches node)",
        f"     {'node':<14}{'acc':>8}{'macroF1':>10}{'pos.F1(fail)':>14}{'n':>6}",
        "     " + "-" * 52,
    ]
    for node, nm in metrics["node"].items():
        c = nm["conditional"]
        pos_f1 = c.get("positive_f1")
        pos_str = _fmt(pos_f1) if pos_f1 is not None else "  —  "
        lines.append(
            f"     {node:<14}{_fmt(c['accuracy']):>8}{_fmt(c['macro_f1']):>10}"
            f"{pos_str:>14}{c['n']:>6}"
        )

    if "brier" in metrics:
        B = metrics["brier"]
        lines += ["", "  c.ii Brier score (probabilistic)",
                  f"     Overall Brier  : {_fmt(B['overall_brier'])}  (n={B['n']})"]
        for node, bm in B["per_layer"].items():
            lines.append(f"       {node:<14} {_fmt(bm['brier'])}  (n={bm['n']})")

    if "by_type" in metrics:
        lines += ["", "  Per-type (final label)",
                  f"     {'type':<22}{'exactAcc':>10}{'MAE':>8}{'n':>5}"]
        for t, tm in metrics["by_type"].items():
            lines.append(
                f"     {t:<22}{_fmt(tm['exact_accuracy']):>10}{_fmt(tm['mae']):>8}{tm['n']:>5}"
            )

    C = metrics.get("contested")
    if C and C.get("contested", {}).get("n"):
        a, c = C["agreed"], C["contested"]
        lines += [
            "",
            "  Gold provenance (gold is target_label for both rows)",
            f"     {'subset':<34}{'exactAcc':>10}{'n':>6}{'share':>8}",
            f"     {'re-scorer agreed with target':<34}"
            f"{_fmt(a.get('exact_accuracy')):>10}{a['n']:>6}{_fmt(a.get('share')):>8}",
            f"     {'re-scorer DISAGREED (contested)':<34}"
            f"{_fmt(c.get('exact_accuracy')):>10}{c['n']:>6}{_fmt(c.get('share')):>8}",
            "     note: CSEG-RM GRPO training drops contested rows, so the model is",
            "           graded on some samples it never saw the gold convention for.",
            "           Report both numbers.",
        ]

    U = metrics.get("usage") or {}
    if U.get("n_with_usage"):
        lines += [
            "",
            "  Tokens / sample (sum of every layer call; thinking included)",
            f"     Completion     : {_fmt(U.get('mean_completion_tokens'))}"
            f"  (reasoning {_fmt(U.get('mean_reasoning_tokens'))})",
            f"     Prompt         : {_fmt(U.get('mean_prompt_tokens'))}",
            f"     Total          : {_fmt(U.get('mean_total_tokens'))}"
            f"  (n={U['n_with_usage']})",
        ]

    lines.append("=" * 70)
    return "\n".join(lines)


LEADERBOARD_FIELDS = [
    "model",
    "n",
    "exact_accuracy",
    "macro_recall",
    "mae",
    "qwk",
    "mean_path_edit_distance",
    "mean_completion_tokens",
    "mean_reasoning_tokens",
]


def _lb_num(value, nd: int = 4):
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if x != x:
        return ""
    return round(x, nd)


def update_leaderboard(leaderboard_path: Path, model: str, metrics: dict) -> None:
    L = metrics["label"]
    P = metrics["path"]
    U = metrics.get("usage") or {}
    row = {
        "model": model,
        "n": L["n"],
        "exact_accuracy": _lb_num(L.get("exact_accuracy")),
        "macro_recall": _lb_num(L.get("macro_recall")),
        "mae": _lb_num(L.get("mae")),
        "qwk": _lb_num(L.get("qwk")),
        "mean_path_edit_distance": _lb_num(P.get("mean_edit_distance")),
        "mean_completion_tokens": _lb_num(U.get("mean_completion_tokens"), 1),
        "mean_reasoning_tokens": _lb_num(U.get("mean_reasoning_tokens"), 1),
    }
    rows: list[dict] = []
    extra_fields: list[str] = []
    if leaderboard_path.exists():
        with open(leaderboard_path, "r", encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
        if existing:
            extra_fields = [k for k in existing[0].keys() if k not in LEADERBOARD_FIELDS]
        rows = [r for r in existing if r.get("model") != model]
    rows.append(row)
    fieldnames = list(LEADERBOARD_FIELDS) + extra_fields
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    with open(leaderboard_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions against gold.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, default=None,
                        help="Where to write metrics.json (default: alongside predictions).")
    parser.add_argument("--leaderboard", type=Path, default=None,
                        help="Optional leaderboard.csv to update.")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (for the report header / leaderboard row).")
    parser.add_argument("--report", action="store_true", help="Print a human-readable report.")
    parser.add_argument("--exclude-types", nargs="*", default=list(DEFAULT_EXCLUDE_TYPES),
                        help="Record types to leave out of the metrics. Default: "
                             f"{' '.join(DEFAULT_EXCLUDE_TYPES)} (same exclusion as "
                             "CSEG-RM training). Pass with no values to score every type. "
                             "Predictions are never filtered on disk, so this can be "
                             "changed and re-run freely.")
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    metrics = evaluate(predictions, exclude_types=args.exclude_types)

    metrics_out = args.metrics_out or (args.predictions.parent / "metrics.json")
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    model = args.model or args.predictions.parent.name
    if args.report:
        print(format_report(metrics, model))

    if args.leaderboard:
        update_leaderboard(args.leaderboard, model, metrics)

    print(f"\nMetrics written to {metrics_out}")


if __name__ == "__main__":
    main()
