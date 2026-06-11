# inference_joint_mixed.py
# Joint prediction on the MIXED subset: factuality (5-class) + genre + bias in one
# call. Reuses the original engine helpers; only the factuality label space, the
# joint prompts, and the output folder change.
#
#   python inference_joint_mixed.py --prompt engineered
#   python inference_joint_mixed.py --prompt simple --limit 10

import os
import json
import time
import argparse
from pathlib import Path

import requests

from inference_multiclass import encode_image, query_model, load_index, MODELS
from inference_joint import _match_label
from mixed_config import (
    FACT_LABELS, BIAS_LABELS, GENRE_LABELS,
    SAMPLE_INDEX, SAMPLE_IMAGE_DIR, RESULTS_DIRS,
)

IMAGE_DIR   = SAMPLE_IMAGE_DIR
RESULTS_DIR = RESULTS_DIRS["joint"]

# Each dimension maps to (ground-truth column, label set). Factuality is 5-class.
DIMENSIONS = {
    "factuality": ("factuality",  FACT_LABELS),
    "genre":      ("genre_class", GENRE_LABELS),
    "bias":       ("bias",        BIAS_LABELS),
}


JOINT_SYSTEM_MIXED = """
You are a media analyst conducting a visual audit of a news source's homepage. Your
only evidence is the single screenshot provided. Evaluate strictly what is visible; do
not use any prior knowledge of the outlet, and do not infer its identity from its name,
logo, or URL.

You must make THREE independent judgements about the same page:

  (1) FACTUALITY — the factual-reliability tier:
        VERY LOW / LOW / MIXED / HIGH / VERY HIGH
  (2) GENRE      — the dominant content category:
        CONSPIRACY / PSEUDOSCIENCE / IMPOSTER / OTHER
  (3) BIAS       — the political leaning:
        LEFT / LEFT-CENTER / LEAST BIASED / RIGHT-CENTER / RIGHT

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
FACTUALITY (5-class tier). Decide how dependable the reporting is from visible cues —
story selection, headline language, sourcing and attribution, opinion-vs-news framing,
and any reliability red flags.

  VERY LOW  — pervasive falsehoods, conspiratorial or pseudoscientific content, no
              standards of evidence or correction.
  LOW       — frequent misleading framing, heavy bias-driven distortion, weak or
              absent sourcing, but not wholly fabricated.
  MIXED     — a genuine but uneven outlet: real reporting alongside recurring lapses
              (failed fact checks, one-sided or thin sourcing, loaded or sensational
              framing, opinion blurred into news, promotion of contested claims). Above
              LOW because not systematically deceptive; below HIGH because not
              consistently dependable. Use it on positive evidence of unevenness, NOT
              as a hedge for uncertainty.
  HIGH      — largely accurate, conventional reporting with visible attribution and
              ordinary editorial structure; minor bias acceptable.
  VERY HIGH — exemplary: sober, heavily sourced, transparent; the register of wire
              services, fact-checkers, research institutes, and pollsters.

Boundary tips: HIGH vs MIXED turns on whether reliability red flags are ABSENT (HIGH)
or RECURRING (MIXED); MIXED vs LOW turns on whether the page is merely UNEVEN (MIXED)
or DOMINATED by distortion (LOW).

------------------------------------------------------------------------------
GENRE. CONSPIRACY / PSEUDOSCIENCE = pages dominated by that content. IMPOSTER = a page
dressed up to look like ordinary (often local) news but really thin or covertly
partisan "pink slime". OTHER = any genuine news outlet or institution.

------------------------------------------------------------------------------
BIAS. Political leaning on the LEFT…RIGHT scale. Leaning is only weakly visible from a
screenshot; default to LEAST BIASED unless partisan framing is clearly on display.
These are independent axes.

------------------------------------------------------------------------------
For each axis the most diagnostic evidence is, in order: the topics the page visibly
covers, the framing and vocabulary of its headlines, then transparency / advertising /
imagery cues (for factuality) or issue-stance and topical emphasis (for bias). Genre
rests almost entirely on visible topic and framing. For factuality specifically, treat
content and headline framing as decisive, with attribution, advertising integrity, and
any synthetic imagery as contributing signals.
""".strip()

JOINT_USER_MIXED = """
Examine this website screenshot and make all three judgements. Begin with exactly these
three lines, each a single label:

FACTUALITY: <VERY LOW | LOW | MIXED | HIGH | VERY HIGH>
GENRE: <CONSPIRACY | PSEUDOSCIENCE | IMPOSTER | OTHER>
BIAS: <LEFT | LEFT-CENTER | LEAST BIASED | RIGHT-CENTER | RIGHT>

Then add a brief explanation (3-5 sentences) citing only what is visible — note in
particular any cues that pushed factuality toward or away from MIXED.
""".strip()

JOINT_SIMPLE_SYSTEM_MIXED = (
    "You are a media analyst. From a single homepage screenshot, judge a source on "
    "three independent axes — factual reliability, content genre, and political "
    "leaning — using only what is visible. Reliable-looking design does not guarantee "
    "reliability."
)

JOINT_SIMPLE_USER_MIXED = """
Examine this website screenshot and make all three judgements. Begin with exactly these
three lines:

FACTUALITY: <VERY LOW | LOW | MIXED | HIGH | VERY HIGH>
GENRE: <CONSPIRACY | PSEUDOSCIENCE | IMPOSTER | OTHER>
BIAS: <LEFT | LEFT-CENTER | LEAST BIASED | RIGHT-CENTER | RIGHT>

FACTUALITY note: MIXED = a genuine outlet that is uneven — some sound reporting but
recurring reliability red flags (sensational framing, thin sourcing, opinion as news).
GENRE note: IMPOSTER = a page dressed up as ordinary local news but thin or covertly
partisan "pink slime"; OTHER = any genuine news or institution.

Then add a brief explanation (2-4 sentences) citing only what is visible.
""".strip()

PROMPTS = {
    "engineered": (JOINT_SYSTEM_MIXED, JOINT_USER_MIXED),
    "simple":     (JOINT_SIMPLE_SYSTEM_MIXED, JOINT_SIMPLE_USER_MIXED),
}


def parse_joint(response_text):
    lines = [ln.strip() for ln in response_text.strip().splitlines() if ln.strip()]
    out = {}
    for dim, (_, labels) in DIMENSIONS.items():
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


def run(prompt_name, limit):
    system, user_text = PROMPTS[prompt_name]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_index(SAMPLE_INDEX)
    if limit:
        rows = rows[:limit]

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

                truth = {}
                for dim, (col, labels) in DIMENSIONS.items():
                    val = (row.get(col) or "").strip().upper()
                    truth[dim] = val if val in labels else ""

                img_path = os.path.join(IMAGE_DIR, Path(row["image_path"]).name)
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
                        "factuality": row.get("factuality", ""),
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

                    miss = [d for d, v in verdicts.items() if v == "UNKNOWN"]
                    tag = "ok" if not miss else f"missing:{','.join(miss)}"
                    summ = " ".join(f"{d[0].upper()}={verdicts[d]}" for d in DIMENSIONS)
                    print(f"  [{i+1}/{len(rows)}] {tag:<16} {summ:<44} {media_name}")

                except requests.HTTPError as e:
                    code = e.response.status_code if e.response is not None else "?"
                    print(f"  [{i+1}/{len(rows)}] HTTP ERROR {code} — {media_name}")
                    time.sleep(5)
                except Exception as e:
                    print(f"  [{i+1}/{len(rows)}] ERROR — {media_name}: {e}")

                time.sleep(1)

    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", choices=["simple", "engineered"], required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    run(args.prompt, args.limit)