import os
import requests
from dotenv import load_dotenv

from config import JOINT_MODEL_ID, EXPLAIN_MODEL_ID

# ---- OPENROUTER_API_KEY loaded here (paste your key in backend/.env) ----
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
# -------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Joint prompt + parser copied verbatim from inference_joint.py (engineered),
# the exact prompt that produced the training verdict/rationale features.
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

# DIMENSIONS / parsing copied verbatim from inference_joint.py: factuality matches a
# 4-class list (no MIXED), so a "MIXED" verdict resolves to UNKNOWN — exactly as in
# training. Do not "fix" this; it preserves the feature distribution.
DIMENSIONS = {
    "factuality": ["VERY LOW", "LOW", "HIGH", "VERY HIGH"],
    "genre":      ["CONSPIRACY", "PSEUDOSCIENCE", "IMPOSTER", "OTHER"],
    "bias":       ["LEFT", "LEFT-CENTER", "LEAST BIASED", "RIGHT-CENTER", "RIGHT"],
}


def _match_label(text, labels):
    def flat(s):
        return s.upper().replace("-", " ").replace("_", " ")
    t = flat(text)
    for label in sorted(labels, key=len, reverse=True):
        if flat(label) in t:
            return label
    return "UNKNOWN"


def parse_joint(response_text):
    lines = [ln.strip() for ln in response_text.strip().splitlines() if ln.strip()]
    out = {}
    for dim, labels in DIMENSIONS.items():
        key = dim.upper()
        found = "UNKNOWN"
        for ln in lines:
            stripped = ln.lstrip("*#-) ").lstrip("0123456789").lstrip(".)* ").upper()
            if stripped.startswith(key):
                after = ln.split(":", 1)[1] if ":" in ln else ln
                found = _match_label(after, labels)
                break
        out[dim] = found
    return out


def _chat(model_id, system, user_text, image_b64, timeout=90):
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY missing — set it in backend/.env")
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            },
        ],
    }
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=payload, timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if "choices" not in data:
        raise ValueError(f"No choices in response: {data.get('error', data)}")
    return data["choices"][0]["message"]["content"]


def joint_verdict(image_b64):
    """Internal call: returns (verdicts_dict, rationale_text) used to build features."""
    response = _chat(JOINT_MODEL_ID, JOINT_SYSTEM, JOINT_USER, image_b64)
    return parse_joint(response), response


EXPLAIN_SYSTEM = (
    "You are a media-literacy assistant. Separate classifiers have rated a news "
    "homepage's factual reliability and (when provided) its political bias. In one "
    "short paragraph, explain why a source like this would receive those ratings, "
    "grounding your explanation in the visible design and content cues of the "
    "screenshot. Reference the prediction(s) explicitly. Do not contradict the "
    "ratings. Keep it to a single paragraph."
)


def explain(image_b64, factuality_label, bias_label=None):
    if bias_label:
        user = (
            f"The classifiers rated this homepage's factual reliability as "
            f"\"{factuality_label}\" and its political bias as \"{bias_label}\". In one "
            f"short paragraph, explain why it earns both ratings, citing visible "
            f"visual/design and content cues in the screenshot (sourcing and framing for "
            f"factuality; topical emphasis and issue-stance for bias)."
        )
    else:
        user = (
            f"The classifier rated this homepage's factual reliability as "
            f"\"{factuality_label}\". In one short paragraph, explain why, citing visible "
            f"visual/design and content cues in the screenshot."
        )
    return _chat(EXPLAIN_MODEL_ID, EXPLAIN_SYSTEM, user, image_b64)
