#!/usr/bin/env python3
"""Verifiable rewards for CSEG-RM GRPO.

verl 0.8.0 loads a custom reward through ``reward.custom_reward_function.path`` +
``reward.custom_reward_function.name`` and calls it — via both the classic and the
experimental (``verl.experimental.reward_loop``) ``naive`` reward managers — with the
signature::

    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float | dict

where

* ``data_source``  : the record's ``data_source`` field. In CSEG-RM this is the
  **layer id** (``L1`` … ``L6`` / ``L5_consensus`` / ``L5_conflict``).
* ``solution_str`` : the decoded model completion (should be one JSON object).
* ``ground_truth`` : ``reward_model.ground_truth`` from the record — the **gold verdict**
  for that layer.
* ``extra_info``   : the record's ``extra_info`` dict (``layer_id`` / ``type`` / …).

Returning a ``dict`` lets us surface extra scalars (``format_valid`` /
``verdict_correct``) that verl logs as ``reward/…`` curves.

The JSON parser is the *same* one the trajectory scorer
(``data_filter/score_trajectory.py``) and the benchmark (``benchmark/predict.py``)
use, so the reward never drifts from how
predictions are parsed at eval time.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Reuse the canonical JSON parser from data_filter/score_trajectory.py.
#
# score_trajectory.py imports ``from openai import OpenAI`` at module top only to
# build a client at *inference* time; parsing needs no network. In the verl
# training venv ``openai`` is installed so this shim is a no-op; in a bare offline
# env (e.g. running the unit test in this file) it lets us import the parser
# without pulling in openai. This is the ONLY reason for the shim.
# ─────────────────────────────────────────────────────────────────────────────
_DATA_FILTER = Path(__file__).resolve().parents[2] / "data_filter"
if str(_DATA_FILTER) not in sys.path:
    sys.path.insert(0, str(_DATA_FILTER))

try:  # pragma: no cover - exercised implicitly
    import openai  # noqa: F401
except ImportError:  # pragma: no cover
    _stub = types.ModuleType("openai")
    _stub.OpenAI = object  # type: ignore[attr-defined]
    sys.modules["openai"] = _stub

import score_trajectory as st  # noqa: E402  (path injected above)

# Legal verdict vocabulary per layer (mirrors benchmark/gold.py:NODE_VERDICTS and the
# ``routing`` blocks of data_filter/prompt_config*.json). A verdict outside this set is
# a malformed judgement and is penalised like a parse failure.
LAYER_VERDICTS: dict[str, set[str]] = {
    "L1": {"pass", "fail"},
    "L2": {"pass", "fail"},
    "L3": {"pass", "fail"},
    "L4": {"consistent", "conflict_identified", "conflict_missed"},
    "L5_consensus": {"pass", "fail"},
    "L5_conflict": {"pass", "fail"},
    "L6": {"pass", "fail"},
}


def _parse_verdict(solution_str: str) -> Optional[str]:
    """Extract a normalised ``verdict`` from a model completion, or ``None``.

    Robust to ```json fenced blocks / trailing prose, matching the production scorer.
    """
    try:
        parsed = st._parse_json_from_text(solution_str)
    except Exception:  # noqa: BLE001 - any parse failure is a format failure
        return None
    if not isinstance(parsed, dict):
        return None
    verdict = parsed.get("verdict")
    if not isinstance(verdict, str):
        return None
    return verdict.strip().lower()


def compute_score(
    data_source: Any = None,
    solution_str: str = "",
    ground_truth: Any = "",
    extra_info: Optional[dict] = None,
    *,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
    format_penalty: float = -1.0,
    check_verdict_vocab: bool = True,
    **_ignored: Any,
) -> dict:
    """Layer-level verifiable reward (the main CSEG-RM GRPO experiment).

    reward = ``format_penalty``  if the completion has no parseable / legal verdict,
             ``correct_reward``  if the verdict equals the gold verdict,
             ``incorrect_reward`` otherwise.

    ``correct_reward`` / ``incorrect_reward`` / ``format_penalty`` / ``check_verdict_vocab``
    are overridable through ``reward.custom_reward_function.reward_kwargs`` in the verl
    config (they are merged into the call by ``verl.trainer.ppo.reward.get_custom_reward_fn``).

    Returns a dict so verl records ``reward/format_valid`` and ``reward/verdict_correct``
    alongside the scalar ``score``.
    """
    extra_info = extra_info or {}
    layer_id = extra_info.get("layer_id") or (data_source if isinstance(data_source, str) else None)
    gold = str(ground_truth).strip().lower()

    verdict = _parse_verdict(solution_str)

    if verdict is None:
        return {"score": format_penalty, "format_valid": 0.0, "verdict_correct": 0.0}

    if check_verdict_vocab and layer_id in LAYER_VERDICTS and verdict not in LAYER_VERDICTS[layer_id]:
        # A syntactically-valid JSON that nonetheless emits an out-of-vocabulary verdict
        # (e.g. L4 answering "pass") is treated as a format failure.
        return {"score": format_penalty, "format_valid": 0.0, "verdict_correct": 0.0}

    correct = verdict == gold
    return {
        "score": correct_reward if correct else incorrect_reward,
        "format_valid": 1.0,
        "verdict_correct": 1.0 if correct else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory-level reward (optional ablation; not used in the paper tables)
#
# Directly optimises the final label + path instead of a single layer. This requires
# a rollout that walks the whole six-layer pipeline in one completion; the expected
# completion contract is a single JSON object:
#     {"label": <int 1-7>, "path": ["L1:pass", "L2:pass", ...]}
# and ``ground_truth`` a JSON string / dict with {"gold_label": int, "gold_path": [...]}.
# ─────────────────────────────────────────────────────────────────────────────
def path_edit_distance(pred_path: list[str], gold_path: list[str]) -> int:
    """Levenshtein edit distance over path token lists (matches benchmark/metrics.py)."""
    la, lb = len(pred_path), len(gold_path)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if pred_path[i - 1] == gold_path[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def _coerce_ground_truth(ground_truth: Any) -> dict:
    import json

    if isinstance(ground_truth, dict):
        return ground_truth
    try:
        obj = json.loads(ground_truth)
        return obj if isinstance(obj, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def compute_score_trajectory(
    data_source: Any = None,
    solution_str: str = "",
    ground_truth: Any = "",
    extra_info: Optional[dict] = None,
    *,
    label_reward: float = 1.0,
    path_penalty: float = 0.2,
    format_penalty: float = -1.0,
    **_ignored: Any,
) -> dict:
    """reward = 1{label==gold} − path_penalty · edit_distance(pred_path, gold_path).

    Used only for the optional trajectory-level GRPO ablation. Point the verl
    config at this function via ``reward.custom_reward_function.name=compute_score_trajectory``.
    """
    gt = _coerce_ground_truth(ground_truth)
    gold_label = gt.get("gold_label")
    gold_path = list(gt.get("gold_path", []))

    try:
        parsed = st._parse_json_from_text(solution_str)
    except Exception:  # noqa: BLE001
        parsed = None
    if not isinstance(parsed, dict) or "label" not in parsed:
        return {"score": format_penalty, "format_valid": 0.0, "label_correct": 0.0, "path_edit_distance": float(len(gold_path))}

    try:
        pred_label = int(parsed.get("label"))
    except (TypeError, ValueError):
        return {"score": format_penalty, "format_valid": 0.0, "label_correct": 0.0, "path_edit_distance": float(len(gold_path))}

    pred_path = [str(t) for t in parsed.get("path", [])]
    dist = path_edit_distance(pred_path, gold_path)
    label_correct = 1.0 if pred_label == gold_label else 0.0
    score = label_reward * label_correct - path_penalty * dist
    return {
        "score": float(score),
        "format_valid": 1.0,
        "label_correct": label_correct,
        "path_edit_distance": float(dist),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test:  python rewards/layer_reward.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ok = 0
    total = 0

    def check(name: str, got: float, want: float) -> None:
        global ok, total
        total += 1
        status = "ok" if abs(got - want) < 1e-9 else "FAIL"
        if status == "ok":
            ok += 1
        print(f"[{status}] {name}: got={got} want={want}")

    # 1) correct verdict inside a fenced JSON block
    r = compute_score("L1", '```json\n{"verdict": "fail", "reason": "no citation"}\n```', "fail",
                      {"layer_id": "L1"})
    check("L1 correct fail", r["score"], 1.0)
    check("L1 format_valid", r["format_valid"], 1.0)

    # 2) wrong verdict
    r = compute_score("L1", '{"verdict": "pass"}', "fail", {"layer_id": "L1"})
    check("L1 wrong verdict", r["score"], 0.0)

    # 3) unparseable → format penalty
    r = compute_score("L4", "the answer clearly identifies the conflict", "conflict_identified",
                      {"layer_id": "L4"})
    check("L4 unparseable", r["score"], -1.0)

    # 4) L4 three-class correct, with trailing prose after the JSON
    r = compute_score("L4", '{"verdict": "conflict_identified", "reason": "x"} done.',
                      "conflict_identified", {"layer_id": "L4"})
    check("L4 conflict_identified", r["score"], 1.0)

    # 5) out-of-vocabulary verdict for the layer → format penalty
    r = compute_score("L4", '{"verdict": "pass"}', "conflict_identified", {"layer_id": "L4"})
    check("L4 illegal verdict pass", r["score"], -1.0)

    # 6) reward_kwargs override
    r = compute_score("L2", '{"verdict": "pass"}', "pass", {"layer_id": "L2"},
                      correct_reward=2.0)
    check("L2 correct reward override", r["score"], 2.0)

    # 7) trajectory-level reward
    r = compute_score_trajectory(
        "traj",
        '{"label": 7, "path": ["L1:pass","L2:pass","L3:pass","L4:conflict_identified","L5_conflict:pass","L6:pass"]}',
        {"gold_label": 7,
         "gold_path": ["L1:pass","L2:pass","L3:pass","L4:conflict_identified","L5_conflict:pass","L6:pass"]},
    )
    check("traj exact match", r["score"], 1.0)

    # 8) trajectory-level: right label, one wrong node (edit distance 1)
    r = compute_score_trajectory(
        "traj",
        '{"label": 7, "path": ["L1:pass","L2:pass","L3:pass","L4:consistent","L5_consensus:pass","L6:pass"]}',
        {"gold_label": 7,
         "gold_path": ["L1:pass","L2:pass","L3:pass","L4:conflict_identified","L5_conflict:pass","L6:pass"]},
    )
    # label correct (1.0) minus 0.2 * edit_distance(2)  [L4 verdict + L5 node both differ]
    check("traj label ok path off", round(r["score"], 4), round(1.0 - 0.2 * 2, 4))

    print(f"\n{ok}/{total} checks passed")
    sys.exit(0 if ok == total else 1)
