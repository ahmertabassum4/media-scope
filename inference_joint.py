# Joint prediction: one screenshot -> factuality + genre + bias in a single call.
#   python inference_joint.py --prompt engineered
#   python inference_joint.py --prompt simple --limit 10


import os
import csv
import json
import time
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv

from inference_multiclass import encode_image, query_model, load_index, MODELS

load_dotenv()

SAMPLE_INDEX = "snapshot_sample_index.csv"
IMAGE_DIR    = "snapshot-samples"
RESULTS_DIR  = Path("results")

# Dimensions predicted jointly. Each maps to (ground-truth column, label set).
# The keyed-line label (e.g. "FACTUALITY") is the dict key, upper-cased.
DIMENSIONS = {
    "factuality": ("factuality",  ["VERY LOW", "LOW", "HIGH", "VERY HIGH"]),
    "genre":      ("genre_class", ["CONSPIRACY", "PSEUDOSCIENCE", "IMPOSTER", "OTHER"]),
    "bias":       ("bias",        ["LEFT", "LEFT-CENTER", "LEAST BIASED",
                                   "RIGHT-CENTER", "RIGHT"]),
}


# Prompts
JOINT_SYSTEM = """
You are a media analyst conducting a visual audit of a news source's homepage. Your
only evidence is the single screenshot provided. Evaluate strictly what is visible; do
not use any prior knowledge of the outlet, and do not infer its identity from its name,
logo, or URL.

You must make THREE independent judgements about the same page:

  (1) FACTUALITY — the factual-reliability tier:  VERY LOW / LOW / HIGH / VERY HIGH
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

JOINT_SIMPLE_SYSTEM = (
    "You are evaluating a news source based solely on the visual appearance of its "
    "homepage screenshot. Do not use any prior knowledge about this outlet. Make three "
    "separate judgements: its factual-reliability tier, its content category, and its "
    "political leaning. These are independent axes."
)

JOINT_SIMPLE_USER = """
Examine this website screenshot and make all three judgements. Begin with exactly these
three lines:

FACTUALITY: <VERY LOW | LOW | HIGH | VERY HIGH>
GENRE: <CONSPIRACY | PSEUDOSCIENCE | IMPOSTER | OTHER>
BIAS: <LEFT | LEFT-CENTER | LEAST BIASED | RIGHT-CENTER | RIGHT>

GENRE note: IMPOSTER = a page dressed up to look like ordinary (often local) news but
really thin or covertly partisan "pink slime"; OTHER = any genuine news or institution.

Then add a brief explanation (2-4 sentences) citing only what is visible.
""".strip()

PROMPTS = {
    "engineered": (JOINT_SYSTEM, JOINT_USER),
    "simple":     (JOINT_SIMPLE_SYSTEM, JOINT_SIMPLE_USER),
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
    for dim, (_, labels) in DIMENSIONS.items():
        key = dim.upper()
        found = "UNKNOWN"
        for ln in lines:
            # Drop leading list/markdown markers: "1. ", "- ", "**", "#", ") "
            stripped = ln.lstrip("*#-) ").lstrip("0123456789").lstrip(".)* ").upper()
            if stripped.startswith(key):
                after = ln.split(":", 1)[1] if ":" in ln else ln
                found = _match_label(after, labels)
                break
        out[dim] = found
    return out


def run(prompt_name, limit):
    system, user_text = PROMPTS[prompt_name]
    rows = load_index(SAMPLE_INDEX)
    if limit:
        rows = rows[:limit]

    RESULTS_DIR.mkdir(exist_ok=True)

    for short_name, model_id in MODELS.items():
        out_path = RESULTS_DIR / f"{short_name}_joint_{prompt_name}.jsonl"

        done = set()
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["media_name"])
                    except (json.JSONDecodeError, KeyError):
                        pass

        print(f"Task  : joint")
        print(f"Model : {model_id}")
        print(f"Prompt: {prompt_name}")
        print(f"Output: {out_path}")
        if done:
            print(f"Resuming — {len(done)} already done")

        with open(out_path, "a", encoding="utf-8") as out_f:
            for i, row in enumerate(rows):
                media_name = row["media_name"]

                if media_name in done:
                    print(f"  [{i+1}/{len(rows)}] skip  {media_name}")
                    continue

                # Ground truth for each dimension. We keep the row even if one axis is missing a label, recording it as empty so eval can skip it per-dimension rather than dropping the whole image
                truth = {}
                for dim, (col, labels) in DIMENSIONS.items():
                    val = (row.get(col) or "").strip().upper()
                    truth[dim] = val if val in labels else ""

                img_filename = Path(row["image_path"]).name
                img_path = os.path.join(IMAGE_DIR, img_filename)

                if not os.path.exists(img_path):
                    print(f"  [{i+1}/{len(rows)}] MISSING IMAGE — {img_path}")
                    continue

                try:
                    image_b64 = encode_image(img_path)
                    response  = query_model(model_id, system, user_text, image_b64)
                    verdicts  = parse_joint(response)

                    record = {
                        "media_name": media_name,
                        "model": short_name,
                        "task": "joint",
                        "prompt": prompt_name,
                        "verdicts": verdicts,  
                        "ground_truth": truth,         
                        "full_response": response,
                        "factuality": row["factuality"],
                    }

                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

                    miss = [d for d, v in verdicts.items() if v == "UNKNOWN"]
                    tag = "Successful" if not miss else f"missing:{','.join(miss)}"
                    summary = " ".join(f"{d[0].upper()}={verdicts[d]}" for d in DIMENSIONS)
                    print(f"  [{i+1}/{len(rows)}] {tag:<18} {summary:<40} {media_name}")

                except requests.HTTPError as e:
                    print(f"  [{i+1}/{len(rows)}] HTTP ERROR {e.response.status_code} — {media_name}")
                    time.sleep(5)

                except Exception as e:
                    print(f"  [{i+1}/{len(rows)}] ERROR — {media_name}: {e}")

                time.sleep(1)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=["simple", "engineered"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.prompt, args.limit)