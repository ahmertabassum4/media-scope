# If u want to run this just run (python inference_test.py --prompt simple) for using the simple prompt template or (python run_inference_test.py --prompt engineered) for prompt engineered prompt

import os
import csv
import base64
import json
import time
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv
# Imports to fix images that are too large
from PIL import Image
import io

load_dotenv()
API_KEY = os.getenv("OPENROUTER_KEY")

SAMPLE_INDEX = "snapshot_sample_index.csv"
IMAGE_DIR    = "snapshot-samples"
RESULTS_DIR  = Path("results")

# Deepseek VLM not available in OpenRouter yet
MODELS = {
    "claude-sonnet-4-6":  "anthropic/claude-sonnet-4-6",
    "gpt-5-5":            "openai/gpt-5.5",
    "gemini-2-5-flash-lite": "google/gemini-2.5-flash-lite",
    "qwen2-5-vl-72b":     "qwen/qwen2.5-vl-72b-instruct",
    "gemma-4-31b":        "google/gemma-4-31b-it",
    "kimi-k2-6":          "moonshotai/kimi-k2.6",
}

# Prompts for the models, I just asked claude to help me craft these prompts, may update with some better grounding in future
SYSTEM_SIMPLE = (
    "You are evaluating whether a website is likely to be factually reliable "
    "based solely on the visual appearance of its homepage screenshot. "
    "Do not use any prior knowledge about this outlet."
)

USER_SIMPLE = (
    "Examine this website screenshot and determine whether the source is likely "
    "FACTUAL or NOT FACTUAL. Pl\n\n"
    "Your response must begin with exactly one word on its own line: either "
    "\"FACTUAL\" or \"NOT FACTUAL\". Then provide a brief explanation "
    "(3–5 sentences), citing only what is visible in the screenshot."
)

SYSTEM_ENGINEERED = """
You are a media-credibility analyst conducting a visual audit of a news source's
homepage. Your only evidence is the single screenshot provided. Evaluate strictly
what is visible; do not use any prior knowledge of the outlet, and do not infer the
outlet's identity from its name, logo, or URL.

Your task is to predict whether the source would be rated as factually RELIABLE or
UNRELIABLE by professional raters. Internalise three facts about this task before you
begin:

1. PROFESSIONAL APPEARANCE IS A WEAK SIGNAL. Unreliable outlets routinely copy the
   design conventions of legitimate news — mastheads, section navigation, clean
   typography, bylines — precisely to manufacture a trustworthy first impression.
   Layout quality, colour scheme, and visual polish are easily faked and must NOT,
   on their own, drive your verdict. A slick site can be unreliable; a plain or
   cluttered site can be reliable.

2. CONTENT AND FRAMING ARE STRONG SIGNALS. What the page chooses to cover, and the
   language it uses to cover it, are far harder to disguise. Weight these heavily.

3. DEFAULT TO RELIABLE ABSENCE OF EVIDENCE. Most visible cues are ambiguous. Only
   move toward UNRELIABLE when you can point to concrete, visible red flags — not
   merely a generic or unfamiliar look. Resist the tendency to over-flag a source as
   unreliable on the basis of aesthetics or tone alone.

Assess the homepage against five dimensions, in descending order of importance:

A. CONTENT, TOPICAL FOCUS & FRAMING (most diagnostic). Two things matter here.
   (i) Topic selection: does the visible coverage cluster around conspiratorial,
   pseudoscientific, hyper-partisan, or single-issue advocacy themes (e.g.
   anti-vaccine, election fraud, "deep state," miracle cures, ethnonationalist or
   apocalyptic framing)? One-sided thematic obsession is strong evidence of
   unreliability; broad, mundane, multi-topic coverage is evidence of reliability.
   (ii) Framing of ordinary topics: even when topics are mainstream, look for
   one-sided framing, framing by omission, or moral/identity-loaded language
   (heavy appeals to loyalty vs. betrayal, patriot vs. traitor, purity vs.
   contamination). A page can cover normal news yet present every item through a
   single ideological lens - this is a negative signal.

B. HEADLINE LANGUAGE, TONE & SPECIFICITY. Negative signals: sensationalism,
   emotionally manipulative or fear-based wording, ALL-CAPS, exclamation- or
   rhetorical-question framing, loaded or derogatory terms, headlines that
   editorialise rather than report, and conspicuously simple, low-vocabulary, or
   vague phrasing. Positive signals: measured, neutral tone AND concrete
   specificity - named people, places, dates, institutions, and verifiable
   particulars. Specific, attributable, fact-grounded headlines indicate
   reliability; vague, abstract, or emotionally totalising headlines do not, even
   when their tone seems calm.

C. TRANSPARENCY & ATTRIBUTION. Look for visible signs of accountability: bylines
   and datelines on stories; a clear masthead and section/category navigation;
   named editorial responsibility; and any visible ownership, funding, or "About"
   disclosure (often in the header or footer). Presence of these is a weakly
   positive signal (they are easily faked); their conspicuous absence - anonymous
   articles, no datelines, no organisational identity - is a mild negative signal.

D. ADVERTISING & MONETISATION INTEGRITY. Assess both the volume and the
   character of advertising. Negative signals: a high ad-to-content ratio;
   ads blended to mimic editorial articles ("chumbox" clickbait grids,
   "you won't believe..." teasers); and low-quality or predatory ad
   categories - casinos and gambling, diet pills and weight-loss "tricks,"
   miracle cures and supplements, "one weird trick" eyesight/blood-sugar
   ads, crypto and get-rich schemes, sexual or shock content. Clear
   separation and labelling of advertising, or mainstream/brand-name
   advertisers, is a mild positive signal. IMPORTANT: junk programmatic ads
   alone are weak evidence - legitimate but low-budget local outlets often
   serve the same ad networks. Treat predatory ads as a contributing
   negative that strengthens a verdict already supported by dimensions A or
   B, not as sufficient grounds on their own.

E. SYNTHETIC OR MANIPULATED IMAGERY. Are lead images plausibly AI-generated or
   doctored (uncanny faces, physically impossible details, emotionally inflammatory
   composites presented as real events)? Visible reliance on such imagery is a
   negative signal. Be conservative: only flag clear cases.

Weigh the dimensions together, giving A and B the most influence, then render a single
binary verdict. A source can fail on polish yet pass overall, or look polished yet fail
on content.
"""

USER_ENGINEERED = """
Examine this homepage screenshot and judge whether the source is likely FACTUAL
(reliable) or NOT FACTUAL (unreliable).

Your response MUST begin with exactly one word on its own line: either "FACTUAL" or
"NOT FACTUAL". Then, on the following lines, give a brief structured assessment of one
to two sentences for each of the five dimensions (A Content, B Headlines, C
Transparency, D Ads/Opinion, E Imagery), citing only what is visible in the screenshot.
Conclude with one sentence stating which signals were decisive. If a dimension is not
visible or not assessable, write "not visible" rather than guessing.
"""

PROMPTS = {
    "simple": (SYSTEM_SIMPLE, USER_SIMPLE),
    "engineered": (SYSTEM_ENGINEERED, USER_ENGINEERED),
}

# Helpers functions
def encode_image(path: str, max_height = 8000):
    with Image.open(path) as img:
        w, h = img.size
        if h > max_height:
            ratio = max_height / h
            img = img.resize((int(w * ratio), max_height), Image.LANCZOS)
        
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def query_model(model_id, system, user_text, image_b64):
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type":      "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ],
    }

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type":  "application/json",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()

    data = resp.json()
    
    if "choices" not in data:
        # Print the actual error so we can see what's happening
        raise ValueError(f"No choices in response: {data.get('error', data)}")
    
    return data["choices"][0]["message"]["content"]

# Pull the first token out of the response and normalise to FACTUAL / NOT FACTUAL / UNKNOWN
def parse_verdict(response_text):
    first_line = response_text.strip().splitlines()[0].strip().upper()

    # Handle "NOT FACTUAL" which spans two tokens
    if first_line.startswith("NOT FACTUAL"):
        return "NOT FACTUAL"
    if first_line.startswith("NOT"):
        return "NOT FACTUAL"
    if first_line.startswith("FACTUAL"):
        return "FACTUAL"

    # If they're not the first 2 tokens, we scan for the keyword anywhere in the first line as a fallback (for smaller models)
    if "NOT FACTUAL" in first_line:
        return "NOT FACTUAL"
    if "FACTUAL" in first_line:
        return "FACTUAL"

    return "UNKNOWN"




# Main
def load_index(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run(prompt_name, limit):
    system, user_text = PROMPTS[prompt_name]
    rows = load_index(SAMPLE_INDEX)

    if limit:
        rows = rows[:limit]

    RESULTS_DIR.mkdir(exist_ok=True)

    for short_name, model_id in MODELS.items():
        out_path = RESULTS_DIR / f"{short_name}_{prompt_name}.jsonl"

        # Figure out which images are already done so we can resume
        done = set()
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["media_name"])
                    except (json.JSONDecodeError, KeyError):
                        pass

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

                img_filename = Path(row["image_path"]).name
                img_path = os.path.join(IMAGE_DIR, img_filename)

                if not os.path.exists(img_path):
                    print(f"  [{i+1}/{len(rows)}] MISSING IMAGE — {img_path}")
                    continue

                try:
                    image_b64 = encode_image(img_path)
                    response  = query_model(model_id, system, user_text, image_b64)
                    verdict   = parse_verdict(response)

                    record = {
                        "media_name": media_name,
                        "model": short_name,
                        "prompt": prompt_name,
                        "verdict": verdict,
                        "full_response": response,
                        "ground_truth": int(row["trustworthiness"]),
                        "factuality": row["factuality"],
                    }

                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

                    status = "Successful" if verdict != "UNKNOWN" else "?"
                    print(f"  [{i+1}/{len(rows)}] {status} {verdict:<14} {media_name}")

                except requests.HTTPError as e:
                    print(f"  [{i+1}/{len(rows)}] HTTP ERROR {e.response.status_code} — {media_name}")
                    time.sleep(5)

                except Exception as e:
                    print(f"  [{i+1}/{len(rows)}] ERROR — {media_name}: {e}")

                time.sleep(1)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        choices=["simple", "engineered"],
        required=True,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    run(args.prompt, args.limit)