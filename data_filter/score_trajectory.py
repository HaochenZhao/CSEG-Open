#!/usr/bin/env python3
"""
Cross-source information integration capability evaluator for search agent trajectories.

Reads prompt_config.json and applies a six-layer hierarchical LLM-as-a-judge pipeline
to a single data record. The label assigned to the record is the layer number where
evaluation stops (the layer that failed, or L6 if all layers pass).

Records are independent, so `--all` scores them through a thread pool
(`--concurrency`, default 32) exactly like benchmark/predict.py does.

Environment variables:
    OPENAI_API_BASE       - Base URL for the OpenAI-compatible API server
    OPENAI_API_KEY        - API key
    OPENAI_TIMEOUT        - Per-request timeout in seconds (default 1800)
    JUDGE_RETRIES         - Retries per layer call (default 2)
    JUDGE_RETRY_BACKOFF   - Base seconds for the jittered retry backoff (default 2)

Usage:
    python score_trajectory.py --record path/to/record.json
    python score_trajectory.py --jsonl path/to/data.jsonl --index 0
    python score_trajectory.py --jsonl path/to/data.jsonl --all --concurrency 32
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Retry policy for a single layer call. Defaults match the previous behavior
# (2 retries) but the wait is now backed off with jitter, which is what keeps a
# high --concurrency run from turning one transient 429/503 into a lost layer.
_JUDGE_RETRIES = int(os.environ.get("JUDGE_RETRIES", "2"))
_JUDGE_RETRY_BACKOFF = float(os.environ.get("JUDGE_RETRY_BACKOFF", "2"))


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Source:
    index: int
    query: str
    title: str
    site: str
    url: str
    authority_level: str
    publish_time: str
    content: str

    def to_text(self) -> str:
        lines = [
            f"[来源 {self.index}]",
            f"  标题: {self.title}",
            f"  网站: {self.site}",
            f"  URL: {self.url}",
            f"  权威性: {self.authority_level}",
            f"  发布时间: {self.publish_time or '未知'}",
            f"  检索query: {self.query}",
            f"  内容摘要:\n{self._indent(self.content[:800])}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _indent(text: str, prefix: str = "    ") -> str:
        return "\n".join(prefix + line for line in text.splitlines())


def _zero_usage() -> dict:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
    }


def add_usage(*parts: Optional[dict]) -> dict:
    out = _zero_usage()
    for part in parts:
        if not part:
            continue
        for key in out:
            try:
                out[key] += int(part.get(key) or 0)
            except (TypeError, ValueError):
                pass
    return out


def extract_chat_usage(resp: Any) -> dict:
    """Pull prompt/completion/reasoning token counts from an OpenAI-style response.

    ``completion_tokens`` includes thinking tokens when the server generated
    them. ``reasoning_tokens`` is the thinking slice when the server reports
    ``usage.completion_tokens_details.reasoning_tokens`` (vLLM / OpenAI).
    """
    usage = _zero_usage()
    raw = getattr(resp, "usage", None)
    if raw is None:
        return usage
    if hasattr(raw, "model_dump"):
        dumped = raw.model_dump()
    elif isinstance(raw, dict):
        dumped = raw
    else:
        dumped = {
            "prompt_tokens": getattr(raw, "prompt_tokens", 0),
            "completion_tokens": getattr(raw, "completion_tokens", 0),
            "total_tokens": getattr(raw, "total_tokens", 0),
            "reasoning_tokens": getattr(raw, "reasoning_tokens", 0),
            "completion_tokens_details": getattr(raw, "completion_tokens_details", None),
        }

    def _as_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    usage["prompt_tokens"] = _as_int(dumped.get("prompt_tokens"))
    usage["completion_tokens"] = _as_int(dumped.get("completion_tokens"))
    usage["total_tokens"] = _as_int(dumped.get("total_tokens"))
    details = dumped.get("completion_tokens_details") or {}
    if hasattr(details, "model_dump"):
        details = details.model_dump()
    if not isinstance(details, dict):
        details = {}
    usage["reasoning_tokens"] = _as_int(
        details.get("reasoning_tokens") or dumped.get("reasoning_tokens")
    )
    if usage["total_tokens"] <= 0:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return usage


@dataclass
class LayerResult:
    layer_id: str
    verdict: str
    raw_response: dict
    prompt_used: str
    error: Optional[str] = None
    usage: Optional[dict] = None


@dataclass
class ScoringResult:
    label: int
    label_description: str
    pipeline_trace: list[LayerResult] = field(default_factory=list)
    # Per-layer verdicts and explanations extracted from each judge response
    layer_verdicts: dict[str, str] = field(default_factory=dict)
    layer_reasons: dict[str, str] = field(default_factory=dict)
    stop_reason: str = ""
    user_question: str = ""
    sources_count: int = 0
    final_answer_preview: str = ""
    usage: dict = field(default_factory=_zero_usage)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory parser  (messages format only)
# ─────────────────────────────────────────────────────────────────────────────
def parse_trajectory(record: dict) -> tuple[str, list[Source], str, str]:
    """
    Extract (user_question, sources, raw_sources_text, final_answer) from a record.

    Expected format — OpenAI messages list, stored as:
      - A Python list directly, OR
      - A JSON-encoded string, OR
      - A Python-repr string (ast.literal_eval)

    Message roles used:
      system    — ignored
      user      — the last one is the question being answered
      assistant — may carry tool_calls (search trigger) or the final answer
      tool      — search results; content uses [web_doc N begin]…[web_doc N end] markup

    Returns:
      user_question    — last user message content
      sources          — parsed Source objects (used for sources_count metadata only)
      raw_sources_text — full concatenation of all tool message contents, passed verbatim
                         to the LLM judge (no truncation, no reformatting)
      final_answer     — last assistant message without tool_calls
    """
    import ast

    raw = record.get("messages")
    if raw is None:
        raise ValueError("Record does not contain a 'messages' field.")

    if isinstance(raw, list):
        msgs = raw
    elif isinstance(raw, str):
        try:
            msgs = json.loads(raw)
        except json.JSONDecodeError:
            msgs = ast.literal_eval(raw)
    else:
        raise ValueError(f"Unsupported 'messages' type: {type(raw)}")

    # Last user-role message → question
    user_question = ""
    for m in reversed(msgs):
        if m.get("role") == "user":
            content = m.get("content", "")
            user_question = content.strip() if isinstance(content, str) else str(content)
            break

    # All tool-role messages → full raw text (verbatim) + parsed Source objects
    tool_parts = [
        m["content"]
        for m in msgs
        if m.get("role") == "tool" and isinstance(m.get("content"), str)
    ]
    raw_sources_text = "\n".join(tool_parts)
    sources = _extract_sources(raw_sources_text)

    # Last assistant message without tool_calls → final answer
    final_answer = ""
    for m in reversed(msgs):
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            content = m.get("content", "")
            final_answer = content.strip() if isinstance(content, str) else str(content)
            break

    return user_question, sources, raw_sources_text, final_answer


def _extract_sources(tool_text: str) -> list[Source]:
    """Parse [web_doc N begin]…[web_doc N end] blocks from concatenated tool messages."""
    sources: list[Source] = []

    doc_blocks = re.findall(
        r"\[web_doc (\d+) begin\](.*?)\[web_doc \1 end\]",
        tool_text,
        re.DOTALL,
    )
    for doc_num_str, doc_content in doc_blocks:
        doc_num = int(doc_num_str)

        def _field(tag: str, _c: str = doc_content) -> str:
            m = re.search(rf"\[{tag}\](.*?)(?=\n\[|\Z)", _c, re.DOTALL)
            return m.group(1).strip() if m else ""

        content_m = re.search(
            r"\[web_doc_content begin\](.*?)\[web_doc_content end\]",
            doc_content,
            re.DOTALL,
        )
        sources.append(
            Source(
                index=doc_num,
                query=_field("search query"),
                title=_field("web_doc_title"),
                site=_field("web_doc_site"),
                url=_field("web_doc_url"),
                authority_level=_field("web_doc_authority_level"),
                publish_time=_field("web_doc_publish_time"),
                content=content_m.group(1).strip() if content_m else "",
            )
        )

    return sources


def format_sources_text(sources: list[Source], max_sources: int = 20) -> str:
    if not sources:
        return "（未检索到任何来源文档）"
    selected = sources[:max_sources]
    parts = [s.to_text() for s in selected]
    if len(sources) > max_sources:
        parts.append(f"... 共 {len(sources)} 个来源，以上仅展示前 {max_sources} 个")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# LLM caller
# ─────────────────────────────────────────────────────────────────────────────
def build_client() -> OpenAI:
    base_url = os.environ.get("OPENAI_API_BASE")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    if not base_url:
        raise EnvironmentError(
            "OPENAI_API_BASE environment variable is not set. "
            "Please set it to the base URL of your OpenAI-compatible API server."
        )
    # Be tolerant of slow, heavy-thinking / uncapped responses: use a generous
    # per-request timeout (seconds) instead of the SDK's default. Override with
    # OPENAI_TIMEOUT. The serving stack may impose its own cap; raise that too
    # if you need a longer request.
    timeout = float(os.environ.get("OPENAI_TIMEOUT", "1800"))
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def call_judge(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    retries: Optional[int] = None,
    model_opts: Optional[dict] = None,
) -> tuple[dict, str, dict]:
    """
    Call the judge LLM and parse the JSON response.
    Returns (parsed_json_dict, raw_text, usage).

    ``usage`` sums every attempt that returned a response (including retries),
    so failed-then-retried calls still count toward token totals. Thinking
    tokens are inside ``completion_tokens``; ``reasoning_tokens`` is the
    thinking slice when the server reports it.

    Failed attempts are retried ``retries`` times (default ``JUDGE_RETRIES``)
    with an exponential, jittered wait, so a transient overload under high
    concurrency does not immediately burn the layer.

    ``model_opts`` is an opt-in, per-model options dict (see the ``model_options``
    map in the benchmark config). Recognized keys — all forwarded to the upstream
    via ``extra_body`` so they survive proxies that ignore top-level fields:

      - ``reasoning_effort``: e.g. "low" / "high" / "none" (thinking control).
      - ``thinking``: Anthropic extended-thinking block, e.g.
        ``{"type": "enabled", "budget_tokens": 32000}``. Claude emits no thinking
        without it — ``reasoning_effort`` on its own leaves reasoning_tokens at 0
        — so the two are set together for the Claude entries.
      - ``max_tokens``: OPTIONAL output cap. When left unset (the default) no
        ``max_tokens`` is sent at all, so the request follows the upstream's own
        (very large) default — no artificial truncation. Set it here only if you
        deliberately want to cap a specific model. When set, it is mirrored into
        ``extra_body`` because some OpenAI-compatible proxies ignore a top-level
        ``max_tokens``.
      - ``disable_default_search``: when true, tells the server not to auto-inject
        a default search tool. Keeps LLM-as-judge runs from silently going online.

    With no ``model_opts`` (and no ``max_tokens``) the request is exactly the
    plain chat call it always was, so models are never affected by these keys.
    """
    opts = dict(model_opts or {})
    effective_max_tokens = opts.get("max_tokens", max_tokens)

    extra_body: dict[str, Any] = {}
    if effective_max_tokens is not None:
        # Some proxies drop a top-level max_tokens; mirror it into extra_body.
        # A standard OpenAI/vLLM server accepts it there just the same.
        extra_body["max_tokens"] = int(effective_max_tokens)
    reasoning_effort = opts.get("reasoning_effort")
    if reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort
    thinking = opts.get("thinking")
    if thinking:
        extra_body["thinking"] = thinking
    if opts.get("disable_default_search"):
        extra_body["disable_default_search"] = True
    # Forward chat-template kwargs (e.g. {"enable_thinking": false}) so eval can
    # reproduce the exact template regime the model was trained under. vLLM reads
    # `chat_template_kwargs` from the request body.
    chat_template_kwargs = opts.get("chat_template_kwargs")
    if chat_template_kwargs:
        extra_body["chat_template_kwargs"] = chat_template_kwargs

    messages = [{"role": "user", "content": prompt}]

    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if effective_max_tokens is not None:
        create_kwargs["max_tokens"] = int(effective_max_tokens)
    if extra_body:
        create_kwargs["extra_body"] = extra_body

    attempts = (_JUDGE_RETRIES if retries is None else int(retries)) + 1
    last_exc: Optional[Exception] = None
    usage_acc = _zero_usage()
    for attempt in range(attempts):
        try:
            resp = client.chat.completions.create(**create_kwargs)
            usage_acc = add_usage(usage_acc, extract_chat_usage(resp))
            raw_text = resp.choices[0].message.content or ""
            parsed = _parse_json_from_text(raw_text)
            return parsed, raw_text, usage_acc
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                # Jitter matters here: without it, every worker that hit the same
                # overload would come back at the same instant and re-trigger it.
                delay = _JUDGE_RETRY_BACKOFF * (2 ** attempt) * (0.5 + random.random())
                log.warning(
                    "LLM call attempt %d/%d failed: %s. Retrying in %.1fs ...",
                    attempt + 1, attempts, exc, delay,
                )
                time.sleep(delay)

    raise RuntimeError(f"LLM call failed after {attempts} attempts: {last_exc}") from last_exc


def _parse_json_from_text(text: str) -> dict:
    """Extract and parse the first JSON object/block from text."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find ```json ... ``` block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find first { ... } block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response:\n{text[:500]}")


# ─────────────────────────────────────────────────────────────────────────────
# Scoring pipeline
# ─────────────────────────────────────────────────────────────────────────────
def score_record(
    record: dict,
    config: dict,
    client: OpenAI,
    model: str,
    verbose: bool = False,
    model_opts: Optional[dict] = None,
    tag: str = "",
) -> ScoringResult:
    """
    Run the full six-layer evaluation pipeline on a single record.
    Returns a ScoringResult with label and full pipeline trace.

    Holds no shared mutable state and never mutates ``config`` or ``record``, so
    it is safe to run on many records at once through a thread pool.

    ``model_opts`` is the per-model options dict forwarded to :func:`call_judge`
    (opt-in; ``None`` keeps the request unchanged apart from ``max_tokens``).
    ``tag`` prefixes this record's log lines (e.g. ``"[7/128] "``) so that logs
    from concurrently scored records stay attributable.
    """
    user_question, sources, raw_sources_text, final_answer = parse_trajectory(record)

    if verbose:
        log.info("%sUser question: %s", tag, user_question[:120])
        log.info("%sSources found: %d", tag, len(sources))
        log.info("%sRaw sources text length: %d chars", tag, len(raw_sources_text))
        log.info("%sFinal answer preview: %s", tag, final_answer[:120])

    template_vars = {
        "user_question": user_question,
        "retrieved_sources": raw_sources_text,   # verbatim, no truncation
        "final_answer": final_answer,
        "agent_search_queries": ", ".join(
            {s.query for s in sources if s.query}
        ),
    }

    # Build layer lookup
    layers_by_id: dict[str, dict] = {layer["id"]: layer for layer in config["layers"]}
    pipeline_cfg: dict = config["pipeline"]
    label_mapping: dict = pipeline_cfg["label_mapping"]
    label_descriptions: dict = pipeline_cfg["label_descriptions"]

    trace: list[LayerResult] = []
    current_layer_id: str = pipeline_cfg["entry_point"]
    stop_reason = "completed pipeline"

    while current_layer_id not in ("STOP", "DONE"):
        layer = layers_by_id.get(current_layer_id)
        if layer is None:
            stop_reason = f"unknown layer id: {current_layer_id}"
            break

        log.info("%sEvaluating layer: %s (%s)", tag, layer["id"], layer["name"])

        # Render prompt
        prompt = layer["prompt_template"].format(**template_vars)

        # Call LLM judge
        layer_usage: Optional[dict] = None
        try:
            parsed_response, raw_text, layer_usage = call_judge(
                client, prompt, model, model_opts=model_opts
            )
            verdict = parsed_response.get("verdict", "").lower().strip()
            error = None
        except Exception as exc:
            log.error("%sLayer %s evaluation failed: %s", tag, layer["id"], exc)
            verdict = "error"
            parsed_response = {}
            raw_text = str(exc)
            error = str(exc)

        layer_result = LayerResult(
            layer_id=layer["id"],
            verdict=verdict,
            raw_response=parsed_response,
            prompt_used=prompt,
            error=error,
            usage=layer_usage,
        )
        trace.append(layer_result)

        if verbose:
            log.info(
                "%s  Layer %s verdict: %s | reason: %s",
                tag,
                layer["id"],
                verdict,
                parsed_response.get("reason", "")[:100],
            )

        # Determine next layer
        routing = layer.get("routing", {})
        next_layer_id = routing.get(verdict)

        if next_layer_id is None:
            # Unknown verdict - treat as STOP
            log.warning(
                "%sLayer %s returned unrecognized verdict '%s'. Stopping.",
                tag,
                layer["id"],
                verdict,
            )
            stop_reason = f"unrecognized verdict '{verdict}' at layer {layer['id']}"
            current_layer_id = "STOP"
        else:
            current_layer_id = next_layer_id

    # Determine label from trace
    label, label_desc = _compute_label(trace, current_layer_id, label_mapping, label_descriptions)

    # Collect per-layer verdicts and reasons as flat dicts for easy access
    layer_verdicts: dict[str, str] = {lr.layer_id: lr.verdict for lr in trace}
    layer_reasons: dict[str, str] = {
        lr.layer_id: lr.raw_response.get("reason", lr.error or "")
        for lr in trace
    }

    return ScoringResult(
        label=label,
        label_description=label_desc,
        pipeline_trace=trace,
        layer_verdicts=layer_verdicts,
        layer_reasons=layer_reasons,
        stop_reason=stop_reason,
        user_question=user_question,
        sources_count=len(sources),
        final_answer_preview=final_answer[:300],
        usage=add_usage(*(lr.usage for lr in trace)),
    )


def _compute_label(
    trace: list[LayerResult],
    terminal_state: str,
    label_mapping: dict,
    label_descriptions: dict,
) -> tuple[int, str]:
    """
    Determine label from the pipeline trace and terminal state.

    Label logic (mapping key format: "<layer_id>_<verdict>"):
      L1/fail → 1, L2/fail → 2, L3/fail → 3,
      L4/conflict_missed → 4,
      L5_consensus/fail or L5_conflict/fail → 5,
      L6/fail → 6, L6/pass → 6

    If there's an error verdict, fall back to the last successfully evaluated layer.
    """
    if not trace:
        return 1, label_descriptions.get("1", "评估失败，未能完成任何层级判断")

    # Find the layer that caused a STOP (or DONE for L6)
    last = trace[-1]
    mapping_key = f"{last.layer_id}_{last.verdict}"

    # Normalize L5 variants to same label
    label_int = label_mapping.get(mapping_key)

    if label_int is None:
        # Handle edge cases: error verdicts → use the previous layer's result
        for result in reversed(trace[:-1]):
            key = f"{result.layer_id}_{result.verdict}"
            label_int = label_mapping.get(key)
            if label_int is not None:
                break
        if label_int is None:
            label_int = 1

    return label_int, label_descriptions.get(str(label_int), "")


def score_records(
    records: list[dict],
    config: dict,
    client: OpenAI,
    model: str,
    verbose: bool = False,
    concurrency: int = 1,
    model_opts: Optional[dict] = None,
    print_summary: bool = True,
) -> list[Optional[ScoringResult]]:
    """Score a batch of records, in parallel when ``concurrency > 1``.

    Each record is driven through the sequential six-layer pipeline, but records
    are independent of each other and the work is pure network I/O, so they are
    scored through a thread pool (the OpenAI client is thread-safe). This is the
    same parallelization benchmark/predict.py uses, with two guarantees:

    - **Input-aligned output** — slot ``i`` always holds the result for
      ``records[i]`` regardless of completion order, so ``--concurrency`` never
      reorders the scored JSONL.
    - **Per-record isolation** — a record that raises gets ``None`` in its slot
      and the run continues, matching the serial loop's log-and-skip behavior;
      one malformed trajectory cannot take down a 5k-record job.

    ``concurrency=1`` reproduces the original serial path exactly.
    """
    n = len(records)
    results: list[Optional[ScoringResult]] = [None] * n
    done = 0

    def _score_safe(i: int) -> tuple[Optional[ScoringResult], Optional[BaseException]]:
        """Score one record, returning the failure instead of raising it."""
        try:
            return score_record(
                record=records[i],
                config=config,
                client=client,
                model=model,
                verbose=verbose,
                model_opts=model_opts,
                tag=f"[{i + 1}/{n}] " if n > 1 else "",
            ), None
        except Exception as exc:  # noqa: BLE001 — one record must not kill the batch
            return None, exc

    def _collect(i: int, result: Optional[ScoringResult], exc: Optional[BaseException]) -> None:
        """Record one finished slot. Runs on the caller's thread only."""
        nonlocal done
        done += 1
        if exc is not None:
            log.error("(%d/%d done) record %d failed: %s", done, n, i + 1, exc)
            if verbose:
                traceback.print_exception(type(exc), exc, exc.__traceback__)
            return
        results[i] = result
        log.info(
            "(%d/%d done) record %d -> label %s (%s)",
            done, n, i + 1, result.label, result.stop_reason,
        )
        if print_summary:
            print(format_result_summary(result), flush=True)

    concurrency = max(1, int(concurrency))
    if concurrency > 1 and n > 1:
        log.info("Scoring %d records with concurrency %d ...", n, concurrency)
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(_score_safe, i): i for i in range(n)}
            for fut in as_completed(futs):
                _collect(futs[fut], *fut.result())
    else:
        for i in range(n):
            _collect(i, *_score_safe(i))

    failed = sum(1 for r in results if r is None)
    if failed:
        log.warning("%d/%d records could not be scored (see errors above).", failed, n)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────
def format_result_summary(result: ScoringResult) -> str:
    sep = "─" * 70
    lines = [
        f"\n{'═' * 70}",
        "  SCORING RESULT",
        f"{'═' * 70}",
        f"  Label        : {result.label}",
        f"  Description  : {result.label_description}",
        f"  Sources found: {result.sources_count}",
        f"  Question     : {result.user_question[:80]}",
        f"  Answer prev. : {result.final_answer_preview[:100]}",
        f"\n  Pipeline Trace:",
        f"  {sep}",
    ]
    for lr in result.pipeline_trace:
        status = "✓" if lr.verdict in ("pass", "consistent", "conflict_identified") else "✗"
        reason = lr.raw_response.get("reason", lr.error or "")[:80]
        lines.append(f"  [{status}] {lr.layer_id:20s} verdict={lr.verdict:25s} | {reason}")
    lines.append(f"  {sep}")
    lines.append(f"  Stop reason: {result.stop_reason}")
    lines.append(f"{'═' * 70}\n")
    return "\n".join(lines)


def result_to_dict(result: ScoringResult) -> dict:
    return {
        # ── Primary label ──────────────────────────────────────────────────
        "label": result.label,
        "label_description": result.label_description,
        # ── Per-layer verdicts (flat, for easy filtering/grouping) ─────────
        "layer_verdicts": result.layer_verdicts,
        # ── Per-layer explanations (flat, one sentence per evaluated layer) ─
        "layer_reasons": result.layer_reasons,
        # ── Metadata ───────────────────────────────────────────────────────
        "stop_reason": result.stop_reason,
        "user_question": result.user_question,
        "sources_count": result.sources_count,
        "final_answer_preview": result.final_answer_preview,
        "usage": result.usage,
        # ── Full trace with all judge response fields ──────────────────────
        "pipeline_trace": [
            {
                "layer_id": lr.layer_id,
                "verdict": lr.verdict,
                "reason": lr.raw_response.get("reason", ""),
                "details": {
                    k: v
                    for k, v in lr.raw_response.items()
                    if k not in ("verdict", "reason")
                },
                "error": lr.error,
                "usage": lr.usage,
            }
            for lr in result.pipeline_trace
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a search agent trajectory using the six-layer LLM-as-a-judge pipeline."
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--record",
        type=Path,
        help="Path to a single JSON record file.",
    )
    source_group.add_argument(
        "--jsonl",
        type=Path,
        help="Path to a JSONL file (use with --index or --all).",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Line index (0-based) to score from the JSONL file (default: 0).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Score all records in the JSONL file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "prompt_config.json",
        help="Path to prompt_config.json (default: prompt_config.json next to this script).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("JUDGE_MODEL", "gpt-4o"),
        help="LLM model name to use for judging (default: gpt-4o or JUDGE_MODEL env var).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSONL file to write scored results.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("JUDGE_CONCURRENCY", "32")),
        help="Number of records scored in parallel via a thread pool "
             "(default: 32 or JUDGE_CONCURRENCY). Use 1 for serial scoring.",
    )
    parser.add_argument(
        "--disable-default-search",
        action="store_true",
        help="Ask the server not to auto-inject a default search tool. Without this "
             "flag a judge backend that searches by default can go online instead of "
             "ruling on the retrieved sources alone.",
    )
    parser.add_argument(
        "--model-options-json",
        type=str,
        default=None,
        help="JSON object merged into model_opts (chat_template_kwargs, max_tokens, ...).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logs (also re-enables per-request HTTP logs).",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # One HTTP line per layer per record buries the actual results once a few
        # dozen records are in flight; --verbose brings them back.
        for noisy in ("httpx", "httpcore", "openai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    # Load config
    config = load_config(args.config)
    log.info("Loaded config version: %s", config.get("version"))

    # Build LLM client
    client = build_client()
    log.info("Using model: %s", args.model)

    # Load records, keeping each one's source line index so the scored output
    # stays traceable back to the input even when some records fail.
    records: list[dict] = []
    source_indices: list[int] = []
    if args.record:
        with open(args.record, "r", encoding="utf-8") as f:
            records.append(json.load(f))
        source_indices.append(0)
    elif args.jsonl:
        with open(args.jsonl, "r", encoding="utf-8") as f:
            all_lines = [line.strip() for line in f if line.strip()]
        if args.all:
            records = [json.loads(line) for line in all_lines]
            source_indices = list(range(len(records)))
            log.info("Scoring all %d records from %s", len(records), args.jsonl)
        else:
            if args.index >= len(all_lines):
                sys.exit(
                    f"Index {args.index} out of range: file has {len(all_lines)} records."
                )
            records.append(json.loads(all_lines[args.index]))
            source_indices.append(args.index)
            log.info("Scoring record %d from %s", args.index, args.jsonl)

    # Score records (in parallel unless --concurrency 1); slot i stays aligned
    # with records[i], and a failed record leaves None behind instead of aborting.
    model_opts = json.loads(args.model_options_json) if args.model_options_json else {}
    if not isinstance(model_opts, dict):
        raise SystemExit("--model-options-json must be an object")
    if args.disable_default_search:
        model_opts["disable_default_search"] = True
    if model_opts.get("disable_default_search"):
        log.info("Default search tool injection disabled for %s", args.model)
    if model_opts.get("chat_template_kwargs"):
        log.info("chat_template_kwargs=%s", model_opts["chat_template_kwargs"])
    model_opts = model_opts or None

    results = score_records(
        records,
        config=config,
        client=client,
        model=args.model,
        verbose=args.verbose,
        concurrency=args.concurrency,
        model_opts=model_opts,
    )
    scored = [(source_indices[i], res) for i, res in enumerate(results) if res is not None]
    all_results = [res for _, res in scored]

    # Write output
    if args.output and scored:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for src_index, res in scored:
                f.write(json.dumps({"index": src_index, **result_to_dict(res)}, ensure_ascii=False))
                f.write("\n")
        log.info("Wrote %d results to %s", len(scored), args.output)

    # Print label distribution if multiple records
    if len(all_results) > 1:
        from collections import Counter
        dist = Counter(r.label for r in all_results)
        print("\nLabel distribution:")
        for lbl in sorted(dist.keys()):
            pct = dist[lbl] / len(all_results) * 100
            print(f"  Label {lbl}: {dist[lbl]:>5d} records ({pct:.1f}%)")


if __name__ == "__main__":
    main()
