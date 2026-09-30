#!/usr/bin/env python3
"""
Benchmark orchestrator: for one or many candidate models, run predict → evaluate
and build a leaderboard. Defaults come from configs/benchmark.json; CLI overrides.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS / ".." / "data_filter"))

import prompt_variants as pv  # noqa: E402
import score_trajectory as st  # noqa: E402

from predict import load_records, predict, write_predictions  # noqa: E402
from evaluate import DEFAULT_EXCLUDE_TYPES, evaluate, format_report, update_leaderboard  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(_THIS / ".env")
except ImportError:
    pass


def load_defaults() -> dict:
    cfg_path = _THIS / "configs" / "benchmark.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (_THIS / path).resolve()


def main() -> None:
    defaults = load_defaults()
    parser = argparse.ArgumentParser(description="Run the hierarchical-labeling benchmark.")
    parser.add_argument("--eval-set", type=str,
                        default=defaults.get("eval_set", "./eval.jsonl"))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (overrides --model and config).")
    parser.add_argument("--config", type=str,
                        default=defaults.get("judge_config", "../data_filter/prompt_config.json"))
    parser.add_argument("--prompt-output-spec", choices=list(pv.OUTPUT_SPEC_MODES),
                        default=defaults.get("prompt_output_spec", "full"),
                        help="Output-format section of each layer prompt. 'full' (default) is "
                             "the paper recipe (32B-RS SFT + GRPO, thinking-on, full JSON). "
                             "Use 'verdict_only' only if the checkpoint was trained on "
                             "trimmed prompts — otherwise train and eval prompts disagree.")
    parser.add_argument("--gold-source", choices=["target", "actual"],
                        default=defaults.get("gold_source", "target"))
    parser.add_argument("--probs", action="store_true", default=defaults.get("probs", False))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=defaults.get("output_dir", "results"))
    parser.add_argument("--concurrency", type=int, default=defaults.get("concurrency", 32),
                        help="Records scored in parallel per model (default: 32; 1 = serial).")
    parser.add_argument("--exclude-types", nargs="*",
                        default=defaults.get("exclude_types", list(DEFAULT_EXCLUDE_TYPES)),
                        help="Record types left out of the metrics (predictions still "
                             f"cover every type). Default: {' '.join(DEFAULT_EXCLUDE_TYPES)}. "
                             "Pass with no values to score everything.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--label", type=str, default=None,
                        help="Name written to the leaderboard, report, and predictions "
                             "folder. API requests still use --model (the served name). "
                             "Only valid with a single --model.")
    parser.add_argument("--model-options-json", type=str, default=None,
                        help="JSON object that REPLACES configs/benchmark.json "
                             "model_options for this run. Pass {} for a plain "
                             "chat call (no reasoning_effort / chat_template_kwargs).")
    args = parser.parse_args()

    # Resolve model list.
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.model:
        models = [args.model]
    elif os.environ.get("EVAL_MODEL"):
        models = [os.environ["EVAL_MODEL"]]
    else:
        models = defaults.get("models", [])
    if not models:
        parser.error("No models specified (use --model/--models, EVAL_MODEL, or configs/benchmark.json).")
    if args.label and len(models) != 1:
        parser.error("--label requires exactly one --model.")

    # Per-model options (scheme 3): a map keyed by model name in the config,
    # e.g. {"deepseek-v4-pro": {"reasoning_effort": "high"}}. Only models listed
    # here get extra request fields; every other model is called exactly as
    # before, so an unknown-parameter model can never be broken by this.
    model_options: dict = defaults.get("model_options", {}) or {}

    eval_set = resolve_path(args.eval_set)
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output)
    leaderboard = output_dir / "leaderboard.csv"

    config = pv.apply_to_config(st.load_config(config_path), args.prompt_output_spec)
    if args.prompt_output_spec != "full":
        st.log.info("Prompt output spec: %s (layer prompts ask for verdict only)", args.prompt_output_spec)
    client = st.build_client()
    records = load_records(eval_set, args.limit)
    st.log.info("Loaded %d eval samples from %s", len(records), eval_set)

    for model in models:
        label = args.label or model
        st.log.info("=== Benchmarking model: %s ===", label)
        if label != model:
            st.log.info("  api model: %s", model)
        opts = dict(model_options.get(model, {}) or {})
        if args.model_options_json is not None:
            overlay = json.loads(args.model_options_json)
            if not isinstance(overlay, dict):
                parser.error("--model-options-json must be a JSON object")
            opts = overlay
        if opts:
            st.log.info("  model_options: %s", opts)
        else:
            st.log.info("  model_options: (none)")
        model_dir = output_dir / label.replace("/", "_")
        predictions = predict(
            records, config, client, model,
            gold_source=args.gold_source, probs=args.probs, verbose=args.verbose,
            concurrency=args.concurrency,
            model_opts=opts,
        )
        write_predictions(predictions, model_dir / "predictions.jsonl")

        metrics = evaluate(predictions, exclude_types=args.exclude_types)
        with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(format_report(metrics, label))
        update_leaderboard(leaderboard, label, metrics)

    st.log.info("Leaderboard: %s", leaderboard)


if __name__ == "__main__":
    main()
