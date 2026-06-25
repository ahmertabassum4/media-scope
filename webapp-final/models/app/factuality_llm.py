"""Joint factuality/genre/bias verdict + on-demand reasoning, ported from Ali's backend.

``joint_verdict`` reproduces the exact training-time prompt (``inference_joint.py``) so the
verdict one-hot and rationale text that feed the MLP match the training feature
distribution. ``explain`` is the separate, user-triggered reasoning paragraph covering both
factuality and bias, grounded in the screenshot and the selected evidence regions.
"""
from __future__ import annotations

from typing import Any

from app.openrouter import OpenRouterVisionClient

JOINT_SYSTEM = """
You are a media analyst conducting a visual audit of a news source's homepage. Your
only evidence is the single screenshot provided. Evaluate strictly what is visible; do
not use any prior knowledge of the outlet, and do not infer its identity from its name,
logo, or URL.

You must make THREE independent judgements about the same page:

  (1) FACTUALITY — the factual-reliability tier:  VERY LOW / LOW / MIXED / HIGH / VERY HIGH
  (2) GENRE      — the dominant content category:  CONSPIRACY / PSEUDOSCIENCE /
                                                    IMPOSTER / OTHER
  (3) BIAS       — the political leaning:           LEFT / LEFT-CENTER / LEAST BIASED /
                                                    RIGHT-CENTER / RIGHT

These three axes are SEPARATE and must not be collapsed into one another. A source can
be reliable yet strongly partisan; centrist yet pseudoscientific; slickly designed yet
VERY LOW. Judge each axis on its own evidence and do not let one verdict mechanically
drive another — though where the visible content genuinely bears on more than one axis
(e.g. overt conspiracy content informing both genre and factuality), you may let it.

Three facts hold across all three axes:

1. PROFESSIONAL APPEARANCE IS A WEAK SIGNAL. Unreliable outlets copy the design
   conventions of legitimate news to manufacture trust. Polish alone settles nothing.
2. CONTENT AND FRAMING ARE STRONG SIGNALS. What the page covers, and the language it
   uses, are hardest to disguise. Weight these most heavily.
3. DEFAULT TO THE UNALARMING READING ABSENT EVIDENCE. Move toward an extreme verdict
   only on concrete, visible cues — not a generic or unfamiliar look.

------------------------------------------------------------------------------
FACTUALITY (tier). VERY LOW = pervasive falsehoods, conspiratorial/pseudoscientific
content, no standards of evidence. LOW = frequent misleading framing, heavy distortion,
weak sourcing. HIGH = largely accurate conventional reporting with visible attribution.
VERY HIGH = exemplary, heavily sourced, sober, the register of wire services and
fact-checkers. The reliable/unreliable split turns on content and framing; the
within-pair distinctions are subtler (VERY HIGH vs HIGH on sourcing density and sober
tone; VERY LOW vs LOW on how overt the conspiratorial/pseudoscientific content is).
Reserve the extreme tiers for clear cases; default to HIGH or LOW when evidence is
moderate.

GENRE (category). The distinction is THEMATIC and about content character, not aesthetic
or political. CONSPIRACY = content organised around hidden-plot narratives (secret
cabals, cover-ups, "deep state," election-fraud, suppressed-truth, apocalyptic/
ethnonationalist themes). PSEUDOSCIENCE = unfounded health/medical/scientific claims
(anti-vaccine, miracle cures, supplements/detox, alternative medicine, creationism,
climate-science denial as fact). IMPOSTER = a page built to LOOK like an ordinary,
often local, news outlet but which is really a thin or covertly partisan astroturf
operation ("pink slime") — the deception is the defining feature, judged on a MISMATCH
between an ordinary-news veneer and hollow or one-directionally partisan substance
(templated local-news look with no genuine local specificity, all-national content
under a place-name masthead, anonymous "staff" bylines, no real local desks/sport/
obituaries). OTHER = any genuine, non-deceptive outlet or institution (real local and
general news, politics, business, tech, education, government, transparent advocacy,
fact-checkers, pollsters). Ordinary bias or low budget alone does NOT move a source out
of OTHER. Choose CONSPIRACY or PSEUDOSCIENCE when the fringe theme is overt; choose
IMPOSTER when a mainstream-news costume hides thin partisan substance; otherwise OTHER.

BIAS (leaning). Political leaning is only WEAKLY visible from a static screenshot —
far less than factuality or genre. Expect most sources to be hard to place. Default
toward the centre (LEAST BIASED, or adjacent LEFT-CENTER / RIGHT-CENTER) unless there
is clear directional evidence. Reserve the poles (LEFT, RIGHT) for content that is
overtly and repeatedly partisan in one direction. A strident tone signals intensity,
NOT direction; direction must come from the substance of what is visibly endorsed or
attacked. If direction stays unclear after looking, choose LEAST BIASED.
------------------------------------------------------------------------------

For each axis the most diagnostic evidence is, in order: the topics the page visibly
covers, the framing and vocabulary of its headlines, then transparency/advertising/
imagery cues (for factuality) or issue-stance and topical emphasis (for bias). Genre
rests almost entirely on visible topic and framing.
""".strip()

JOINT_USER = """
Examine this homepage screenshot and make all three judgements.

Your response MUST begin with exactly these three lines, in this order, each label
chosen only from the allowed set for that axis:

FACTUALITY: <VERY LOW | LOW | HIGH | VERY HIGH>
GENRE: <CONSPIRACY | PSEUDOSCIENCE | IMPOSTER | OTHER>
BIAS: <LEFT | LEFT-CENTER | LEAST BIASED | RIGHT-CENTER | RIGHT>

After those three lines, give a short justification (two to four sentences) citing only
what is visible, and note explicitly if the bias direction was unclear. Do not add any
text before the three label lines.
""".strip()

# Parsing copied verbatim from inference_joint.py: factuality matches a 4-class list
# (no MIXED), so a "MIXED" verdict resolves to UNKNOWN — exactly as in training. Do not
# "fix" this; it preserves the feature distribution the MLP was trained on.
DIMENSIONS = {
    "factuality": ["VERY LOW", "LOW", "HIGH", "VERY HIGH"],
    "genre": ["CONSPIRACY", "PSEUDOSCIENCE", "IMPOSTER", "OTHER"],
    "bias": ["LEFT", "LEFT-CENTER", "LEAST BIASED", "RIGHT-CENTER", "RIGHT"],
}


def _match_label(text: str, labels: list[str]) -> str:
    def flat(value: str) -> str:
        return value.upper().replace("-", " ").replace("_", " ")

    flat_text = flat(text)
    for label in sorted(labels, key=len, reverse=True):
        if flat(label) in flat_text:
            return label
    return "UNKNOWN"


def parse_joint(response_text: str) -> dict[str, str]:
    lines = [line.strip() for line in response_text.strip().splitlines() if line.strip()]
    out: dict[str, str] = {}
    for dim, labels in DIMENSIONS.items():
        key = dim.upper()
        found = "UNKNOWN"
        for line in lines:
            stripped = line.lstrip("*#-) ").lstrip("0123456789").lstrip(".)* ").upper()
            if stripped.startswith(key):
                after = line.split(":", 1)[1] if ":" in line else line
                found = _match_label(after, labels)
                break
        out[dim] = found
    return out


def joint_verdict(client: OpenRouterVisionClient, model: str, data_url: str) -> tuple[dict[str, str], str]:
    """Returns (verdicts, rationale_text); both feed the MLP feature builder."""
    response, _ = client.complete_vision(
        model=model,
        prompt=JOINT_USER,
        data_urls=[data_url],
        system=JOINT_SYSTEM,
    )
    return parse_joint(response), response


EXPLAIN_SYSTEM = (
    "You are a media-literacy assistant. Separate classifiers have rated a news homepage's "
    "factual reliability and political bias. In one short paragraph, explain why a source like "
    "this would receive those ratings, grounding your explanation in the visible design and "
    "content cues of the screenshot and the highlighted evidence regions. Reference the "
    "prediction(s) explicitly. Do not contradict the ratings. Keep it to a single paragraph."
)


def _evidence_lines(evidence: list[dict[str, Any]] | None, limit: int = 8) -> str:
    if not evidence:
        return ""
    lines: list[str] = []
    for item in evidence[:limit]:
        reason = str(item.get("reason") or item.get("text") or "").strip()
        if not reason:
            continue
        bbox = item.get("bbox")
        suffix = f" [bbox {bbox}]" if isinstance(bbox, (list, tuple)) else ""
        lines.append(f"- {reason}{suffix}")
    return "\n".join(lines)


def explain(
    client: OpenRouterVisionClient,
    model: str,
    data_url: str,
    factuality_label: str | None,
    bias_label: str | None,
    evidence: list[dict[str, Any]] | None = None,
) -> str:
    parts: list[str] = []
    if factuality_label and bias_label:
        parts.append(
            f'The classifiers rated this homepage\'s factual reliability as "{factuality_label}" and '
            f'its political bias as "{bias_label}". In one short paragraph, explain why it earns both '
            f"ratings, citing visible visual/design and content cues in the screenshot (sourcing and "
            f"framing for factuality; topical emphasis and issue-stance for bias)."
        )
    elif factuality_label:
        parts.append(
            f'The classifier rated this homepage\'s factual reliability as "{factuality_label}". In one '
            f"short paragraph, explain why, citing visible visual/design and content cues in the screenshot."
        )
    elif bias_label:
        parts.append(
            f'The classifier rated this homepage\'s political bias as "{bias_label}". In one short '
            f"paragraph, explain why, citing visible topical emphasis and issue-stance cues in the screenshot."
        )
    else:
        parts.append(
            "In one short paragraph, explain this homepage's factual reliability and political leaning, "
            "citing visible visual/design and content cues in the screenshot."
        )

    evidence_text = _evidence_lines(evidence)
    if evidence_text:
        parts.append("Highlighted evidence regions the classifiers relied on:\n" + evidence_text)

    response, _ = client.complete_vision(
        model=model,
        prompt="\n\n".join(parts),
        data_urls=[data_url],
        system=EXPLAIN_SYSTEM,
    )
    return response
