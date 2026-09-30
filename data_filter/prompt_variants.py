"""Output-format variants for the six-layer judge prompts.

The shipped ``prompt_config*.json`` templates ask the judge for a rich JSON object
(``verdict`` plus per-layer detail fields such as ``key_claims`` / ``evidence_roots`` /
``adjudication_analysis``). That full spec is what the frontier baselines are scored
against, so it must stay untouched in the config files.

The distilled RM, however, is trained to emit **only** ``{"verdict": ...}``:

  * only ``verdict`` is ever scored (``benchmark/gold.py``, ``rewards/layer_reward.py``);
  * the free-text detail fields quote the retrieved sources, which are full of bare
    backslashes from scraped markdown/LaTeX (``\\mathrm``, ``\\~``, ``$1\\%$``). Copying
    those into a JSON string yields either a hard ``json.loads`` failure (``\\m``) or a
    silent corruption (``\\t`` -> TAB), and the training targets contain no escaped
    literal backslash to learn from;
  * ``verdict`` is drawn from a closed vocabulary, so a verdict-only target has zero
    escaping surface by construction.

Asking for five fields while supervising one teaches the model to ignore its own format
instructions, so the prompt has to be trimmed in lockstep with the target. This module
performs that rewrite, and lives in ``data_filter`` so training and evaluation
share one definition and never drift apart.
"""

from __future__ import annotations

# Marker that ends every layer template's instruction body. Verified to occur exactly
# once per layer and to open the final section of the template.
_OUTPUT_FORMAT_HEADER = "## Output Format"

OUTPUT_SPEC_MODES = ("full", "verdict_only")


def allowed_verdicts(layer: dict) -> list[str]:
    """Legal verdicts for a layer, in config order (the ``routing`` keys).

    Mirrors ``rewards/layer_reward.py:LAYER_VERDICTS`` and ``benchmark/gold.py``, but
    reads the config so a routing change cannot silently desync the prompt.
    """
    return list((layer.get("routing") or {}).keys())


def _verdict_only_section(verdicts: list[str]) -> str:
    """Build the replacement ``## Output Format`` section.

    Braces are doubled because the template is rendered with ``str.format(**tvars)``.
    The example is deliberately fence-free and single-line so it matches the training
    target byte-for-byte.
    """
    options = " or ".join(f'"{v}"' for v in verdicts)
    return (
        f"{_OUTPUT_FORMAT_HEADER}\n"
        "Respond with ONLY a single-line JSON object containing exactly one key, "
        "`verdict`. No markdown fence, no reasoning, no additional text or fields.\n"
        f'{{{{"verdict": {options}}}}}'
    )


def rewrite_output_format(prompt_template: str, layer: dict, mode: str = "verdict_only") -> str:
    """Return ``prompt_template`` with its output spec set to ``mode``.

    ``full`` returns the template unchanged (frontier-baseline comparability).
    """
    if mode not in OUTPUT_SPEC_MODES:
        raise ValueError(f"mode must be one of {OUTPUT_SPEC_MODES}, got {mode!r}")
    if mode == "full":
        return prompt_template

    head, sep, _tail = prompt_template.partition(_OUTPUT_FORMAT_HEADER)
    if not sep:
        raise ValueError(
            f"layer {layer.get('id')!r} template has no {_OUTPUT_FORMAT_HEADER!r} section; "
            "the verdict-only rewrite would silently keep the full spec."
        )
    verdicts = allowed_verdicts(layer)
    if not verdicts:
        raise ValueError(f"layer {layer.get('id')!r} has no routing keys to derive verdicts from.")
    return head + _verdict_only_section(verdicts)


def apply_to_config(config: dict, mode: str = "verdict_only") -> dict:
    """Rewrite every layer's output spec in a **copy** of ``config``."""
    if mode == "full":
        return config
    layers = [
        {**layer, "prompt_template": rewrite_output_format(layer["prompt_template"], layer, mode)}
        for layer in config["layers"]
    ]
    return {**config, "layers": layers}
