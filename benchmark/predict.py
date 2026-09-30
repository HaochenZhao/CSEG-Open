#!/usr/bin/env python3
"""
Run a candidate (small open-source) model as the six-layer judge and collect
predictions for the benchmark.

Reuses the canonical scoring pipeline from `data_filter/score_trajectory.py`, so
the benchmark never drifts from the trajectory scorer. Gold is read from the eval
record but never sent to the model.

Output: predictions.jsonl (one line per eval sample).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Make data_filter importable.
_THIS = Path(__file__).resolve().parent
_DATA_FILTER = _THIS.parent / "data_filter"
sys.path.insert(0, str(_DATA_FILTER))

import score_trajectory as st  # noqa: E402  (path injected above)

from gold import gold_from_record, path_to_tokens  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(_THIS / ".env")
except ImportError:
    pass


# Appended to a layer prompt to elicit a calibrated pass/advance probability.
_PROB_SUFFIX = (
    "\n\nAdditionally, include a field `pass_probability`: a float in [0,1] giving "
    "your probability that this layer's verdict is PASS. For the conflict-detection "
    "layer (L4), report the probability that the answer ADVANCES (i.e. verdict is "
    "NOT 'conflict_missed'). Respond as a single JSON object including both "
    "`verdict` and `pass_probability`."
)


def load_records(path: Path, limit: Optional[int] = None) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def _template_vars(record: dict) -> dict:
    """Rebuild the same template variables score_record feeds to each layer."""
    user_question, sources, raw_sources_text, final_answer = st.parse_trajectory(record)
    return {
        "user_question": user_question,
        "retrieved_sources": raw_sources_text,
        "final_answer": final_answer,
        "agent_search_queries": ", ".join({s.query for s in sources if s.query}),
    }


def elicit_probabilities(
    record: dict,
    reached_layers: list[str],
    config: dict,
    client,
    model: str,
    model_opts: Optional[dict] = None,
) -> tuple[dict[str, float], dict]:
    """For each layer the walk reached, ask the model for a pass/advance probability."""
    layers_by_id = {layer["id"]: layer for layer in config["layers"]}
    tvars = _template_vars(record)
    probs: dict[str, float] = {}
    usage_acc = st.add_usage()
    for lid in reached_layers:
        layer = layers_by_id.get(lid)
        if layer is None:
            continue
        prompt = layer["prompt_template"].format(**tvars) + _PROB_SUFFIX
        try:
            parsed, _, extra_usage = st.call_judge(
                client, prompt, model, model_opts=model_opts
            )
            usage_acc = st.add_usage(usage_acc, extra_usage)
            p = parsed.get("pass_probability")
            if p is not None:
                probs[lid] = max(0.0, min(1.0, float(p)))
        except Exception as exc:  # noqa: BLE001
            st.log.warning("Probability elicitation failed at %s: %s", lid, exc)
    return probs, usage_acc


def predict(
    records: list[dict],
    config: dict,
    client,
    model: str,
    gold_source: str,
    probs: bool = False,
    verbose: bool = False,
    concurrency: int = 1,
    model_opts: Optional[dict] = None,
) -> list[dict]:
    """Score every record with the six-layer judge and collect predictions.

    Records are independent and the scorer holds no shared mutable state, so with
    ``concurrency > 1`` they are scored via a thread pool (the OpenAI client is
    I/O-bound and thread-safe). Results are written into an index-keyed slot, so
    the returned order is identical to the serial path regardless of completion
    order; worker exceptions propagate, matching the serial crash semantics.
    """
    label_mapping = config["pipeline"]["label_mapping"]
    n = len(records)

    def _predict_one(i: int, record: dict) -> dict:
        gold_label, gold_path = gold_from_record(record, gold_source, label_mapping)

        result = st.score_record(
            record, config, client, model, verbose=verbose,
            model_opts=model_opts,
        )
        pred_path = [(lr.layer_id, lr.verdict) for lr in result.pipeline_trace]

        pred_probs: dict[str, float] = {}
        extra_usage = st.add_usage()
        if probs:
            reached = [lid for lid, _ in pred_path]
            pred_probs, extra_usage = elicit_probabilities(
                record, reached, config, client, model,
                model_opts=model_opts,
            )

        pred = {
            "index": i,
            "type": record.get("type"),
            "gold_label": gold_label,
            "gold_path": path_to_tokens(gold_path),
            # Whether a re-scoring judge agreed with the construction target.
            # Gold stays `target` either way (--gold-source); the flag lets
            # evaluate.py report the contested subset separately.
            "match": record.get("match"),
            "pred_label": result.label,
            "pred_path": path_to_tokens(pred_path),
            "pred_probs": pred_probs,
            "pipeline_trace": st.result_to_dict(result)["pipeline_trace"],
            # Per-sample sum across every layer (and optional --probs) call.
            # completion_tokens includes thinking; reasoning_tokens is the
            # thinking slice when the server reports it.
            "usage": st.add_usage(result.usage, extra_usage),
        }
        st.log.info(
            "[%d/%d] type=%s gold=%s pred=%s",
            i + 1, n, record.get("type"), gold_label, result.label,
        )
        return pred

    concurrency = max(1, int(concurrency))
    if concurrency > 1 and n > 1:
        predictions: list[Optional[dict]] = [None] * n
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(_predict_one, i, rec): i for i, rec in enumerate(records)}
            for fut in as_completed(futs):
                i = futs[fut]
                predictions[i] = fut.result()  # re-raises like the serial path
        return predictions

    return [_predict_one(i, rec) for i, rec in enumerate(records)]


def write_predictions(predictions: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect six-layer judge predictions for a model.")
    parser.add_argument("--eval-set", type=Path,
                        default=_THIS / "eval.jsonl")
    parser.add_argument("--model", type=str, default=os.environ.get("EVAL_MODEL"))
    parser.add_argument("--config", type=Path,
                        default=_DATA_FILTER / "prompt_config.json")
    parser.add_argument("--gold-source", choices=["target", "actual"], default="target")
    parser.add_argument("--probs", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output predictions.jsonl path.")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Number of records scored in parallel via a thread pool "
                             "(default: 32). Use 1 for the original serial behavior.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.model:
        parser.error("No model given (set --model or EVAL_MODEL).")

    config = st.load_config(args.config)
    client = st.build_client()
    records = load_records(args.eval_set, args.limit)

    predictions = predict(
        records, config, client, args.model,
        gold_source=args.gold_source, probs=args.probs, verbose=args.verbose,
        concurrency=args.concurrency,
    )
    write_predictions(predictions, args.output)
    st.log.info("Wrote %d predictions to %s", len(predictions), args.output)


if __name__ == "__main__":
    main()
