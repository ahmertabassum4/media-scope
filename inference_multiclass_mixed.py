# inference_multiclass_mixed.py
# Multiclass inference on the MIXED 5-class subset. Reuses the original engine
# (encode_image / query_model / parse_verdict / load_index / MODELS) and only
# overrides the factuality label space + rubric and the output folders.
#
#   python inference_multiclass_mixed.py --task factuality --prompt engineered
#   python inference_multiclass_mixed.py --task bias --prompt simple
#   python inference_multiclass_mixed.py --task genre --prompt engineered --limit 10

import json
import time
import argparse
from pathlib import Path

import requests

from inference_multiclass import (
    encode_image, query_model, parse_verdict, load_index, MODELS,
    GENRE_SYSTEM, GENRE_USER, BIAS_SYSTEM, BIAS_USER, SIMPLE, IMAGE_DIR as _ORIG_IMG_DIR,
)
from mixed_config import (
    FACT_LABELS, BIAS_LABELS, GENRE_LABELS,
    SAMPLE_INDEX, SAMPLE_IMAGE_DIR, RESULTS_DIRS,
)

IMAGE_DIR = SAMPLE_IMAGE_DIR


# 5-class factuality rubric (MIXED reinstated). Mirrors the original engineered
# prompt's structure but adds an explicit, comprehensive MIXED tier and a sharper
# account of where MIXED sits between the reliable and unreliable halves.
FACT_SYSTEM_MIXED = """
You are a media-credibility analyst conducting a visual audit of a news source's
homepage. Your only evidence is the single screenshot provided. Evaluate strictly
what is visible; do not use any prior knowledge of the outlet, and do not infer the
outlet's identity from its name, logo, or URL.

Predict the factual-reliability tier a professional rater would assign the source, on
a FIVE-point scale:

  VERY LOW  — pervasive falsehoods, conspiratorial or pseudoscientific content,
              no discernible standards of evidence or correction.
  LOW       — frequent misleading framing, heavy bias-driven distortion, weak or
              absent sourcing, but not wholly fabricated.
  MIXED     — a genuine but uneven outlet: some sound reporting alongside recurring
              lapses — failed fact checks, unreliable or one-sided sourcing, loaded
              or sensational framing, opinion blurred into news, or promotion of
              contested claims. Not systematically deceptive (so above LOW), but not
              consistently dependable either (so below HIGH). This is the default for
              pages that look like ordinary news yet show clear reliability warning
              signs without tipping into overt distortion.
  HIGH      — largely accurate, conventional reporting with visible attribution and
              ordinary editorial structure; minor bias acceptable.
  VERY HIGH — exemplary: sober, heavily sourced, transparent, the visual register of
              wire services, fact-checkers, research institutes, and pollsters.

Internalise four facts before you begin:

1. PROFESSIONAL APPEARANCE IS A WEAK SIGNAL. Unreliable outlets copy the design
   conventions of legitimate news — mastheads, section navigation, clean typography,
   bylines — to manufacture trust. Polish alone must not drive the tier. A slick site
   can be VERY LOW; a plain site can be HIGH.

2. CONTENT AND FRAMING ARE STRONG SIGNALS. What the page covers, and the language it
   uses, are far harder to disguise. Weight these most heavily when separating the
   reliable tiers (HIGH / VERY HIGH) from the unreliable tiers (LOW / VERY LOW) and
   from the uneven middle (MIXED).

3. MIXED IS A REAL CATEGORY, NOT A HEDGE. Choose MIXED when the page shows genuine
   journalistic form but also concrete reliability red flags — sensational or
   emotionally loaded headlines next to straight reports, heavy opinion framing,
   prominent unsourced or single-source claims, advocacy or partisan campaigning
   presented as news. Do NOT use MIXED merely because you are uncertain; use it when
   the visible evidence positively points to an uneven outlet.

4. THE TIER BOUNDARIES DIFFER IN KIND. The reliable/unreliable split turns mainly on
   content and framing. The within-pair distinctions are subtler: VERY HIGH vs HIGH
   turns on the density of sourcing, transparency, and sober tone; VERY LOW vs LOW on
   how overtly conspiratorial or pseudoscientific the visible content is; HIGH vs
   MIXED on whether reliability red flags are absent (HIGH) or recurring (MIXED); and
   MIXED vs LOW on whether the page is uneven (MIXED) or dominated by distortion (LOW).
   Reserve the extreme tiers (VERY LOW, VERY HIGH) for clear cases. When evidence is
   moderate, default to the inner tiers — and when an otherwise ordinary-looking news
   page shows recurring but non-dominating reliability red flags, MIXED is the natural
   landing point rather than HIGH or LOW.

Assess the homepage against five dimensions, in descending order of importance:

A. CONTENT, TOPICAL FOCUS & FRAMING (most diagnostic). Does visible coverage cluster
   around conspiratorial, pseudoscientific, hyper-partisan, or single-issue advocacy
   themes? One-sided thematic obsession pushes toward the LOW tiers; broad, mundane,
   multi-topic coverage pushes toward the HIGH tiers. Watch also for one-sided framing
   of ordinary topics and moral/identity-loaded language. Coverage that is recognisably
   real news yet carries persistent slant or selective emphasis — without collapsing
   into single-issue obsession — points toward MIXED.

B. HEADLINE LANGUAGE, TONE & SPECIFICITY. Sensationalism, fear-based wording,
   ALL-CAPS, exclamations, loaded terms, and vague or low-vocabulary phrasing push
   down. Measured tone with concrete specificity — named people, places, dates,
   institutions — pushes up, and dense, sober specificity is the mark of VERY HIGH.
   Intermittent sensational or loaded headlines sitting beside otherwise straight,
   specific reporting are a hallmark of MIXED.

C. TRANSPARENCY & ATTRIBUTION. Bylines, datelines, mastheads, section navigation,
   visible ownership/funding/"About" disclosure. Presence is weakly positive (easily
   faked); conspicuous absence is mildly negative. Heavy, institutional transparency
   supports VERY HIGH; partial or uneven attribution is consistent with MIXED.

D. ADVERTISING & MONETISATION INTEGRITY. High ad-to-content ratio, chumbox clickbait,
   and predatory ad categories (miracle cures, diet pills, "one weird trick," crypto)
   push down. Treat junk programmatic ads as a contributing signal, not decisive on
   their own — low-budget legitimate locals serve the same networks.

E. SYNTHETIC OR MANIPULATED IMAGERY. Plausibly AI-generated or doctored lead images
   presented as real push down. Be conservative; flag only clear cases.

Weigh the dimensions together, giving A and B the most influence, and render a single
tier.
""".strip()

FACT_USER_MIXED = """
Examine this homepage screenshot and assign the source's factual-reliability tier.

Your response MUST begin with exactly one label on its own line, chosen from exactly
these five: VERY LOW, LOW, MIXED, HIGH, VERY HIGH. Then give a brief structured
assessment of one to two sentences for each dimension (A Content, B Headlines,
C Transparency, D Ads, E Imagery), citing only what is visible. Conclude with one
sentence naming the signals that decided the tier, and in particular what separated it
from the adjacent tier — for an inner-tier verdict, say what kept it from MIXED (or what
pushed it into MIXED). If a dimension is not visible, write "not visible" rather than
guessing.
""".strip()

# Simple-variant factuality, extended to 5 classes.
FACT_SIMPLE_MIXED = (
    SIMPLE["factuality"][0].replace(
        "VERY LOW, LOW, HIGH, VERY HIGH", "VERY LOW, LOW, MIXED, HIGH, VERY HIGH"
    ) if False else
    # SIMPLE[...] is a (system, user) tuple; rebuild both with the 5-class set.
    (
        "You are a media-credibility analyst. Judge a news homepage from a single "
        "screenshot alone, using only what is visible. Reliable-looking design does "
        "not guarantee reliability.",
        "Examine this website screenshot and judge how factually reliable the source "
        "is.\n\nYour response must begin with exactly one label on its own line, "
        "chosen from: VERY LOW, LOW, MIXED, HIGH, VERY HIGH. Then give a brief "
        "explanation (3-5 sentences), citing only what is visible.",
    )
)


TASKS = {
    "factuality": {
        "column": "factuality",
        "labels": FACT_LABELS,
        "engineered": (FACT_SYSTEM_MIXED, FACT_USER_MIXED),
        "simple": FACT_SIMPLE_MIXED,
    },
    "genre": {
        "column": "genre_class",
        "labels": GENRE_LABELS,
        "engineered": (GENRE_SYSTEM, GENRE_USER),
        "simple": SIMPLE["genre"],
    },
    "bias": {
        "column": "bias",
        "labels": BIAS_LABELS,
        "engineered": (BIAS_SYSTEM, BIAS_USER),
        "simple": SIMPLE["bias"],
    },
}


def run(task_name, prompt_name, limit):
    task = TASKS[task_name]
    system, user_text = task[prompt_name]
    labels = task["labels"]
    gt_column = task["column"]

    results_dir = RESULTS_DIRS[task_name]
    results_dir.mkdir(exist_ok=True)

    rows = load_index(SAMPLE_INDEX)
    if limit:
        rows = rows[:limit]

    for short_name, model_id in MODELS.items():
        out_path = results_dir / f"{short_name}_{task_name}_{prompt_name}.jsonl"

        done = set()
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["media_name"])
                    except (json.JSONDecodeError, KeyError):
                        pass

        print(f"Task  : {task_name}")
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

                ground_truth = (row.get(gt_column) or "").strip().upper()
                if ground_truth not in labels:
                    print(f"  [{i+1}/{len(rows)}] no label "
                          f"({gt_column!r}={ground_truth!r}) — {media_name}")
                    continue

                img_path = Path(IMAGE_DIR) / Path(row["image_path"]).name
                if not img_path.exists():
                    print(f"  [{i+1}/{len(rows)}] MISSING IMAGE — {img_path}")
                    continue

                try:
                    image_b64 = encode_image(str(img_path))
                    response  = query_model(model_id, system, user_text, image_b64)
                    verdict   = parse_verdict(response, labels)

                    record = {
                        "media_name": media_name,
                        "model": short_name,
                        "task": task_name,
                        "prompt": prompt_name,
                        "verdict": verdict,
                        "ground_truth": ground_truth,
                        "full_response": response,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

                    tag = "ok" if verdict != "UNKNOWN" else "UNKNOWN"
                    print(f"  [{i+1}/{len(rows)}] {tag:<8} {verdict:<12} {media_name}")

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
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--prompt", choices=["simple", "engineered"], required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    run(args.task, args.prompt, args.limit)