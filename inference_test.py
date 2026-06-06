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

SYSTEM_ENGINEERED = (
    "You are a media credibility analyst performing a visual audit of a website's "
    "homepage. Your sole evidence is the provided screenshot. Do not draw on any "
    "prior knowledge of the outlet — evaluate only what is visibly present.\n\n"
    "Assess the page against these five dimensions:\n\n"
    "1. TRANSPARENCY & ATTRIBUTION — visible bylines, author names, dates, "
    "masthead, or named institutional affiliation. Absence is a negative signal.\n\n"
    "2. PROFESSIONAL PRESENTATION — consistent typography, structured layout, "
    "coherent colour scheme, high-quality imagery, absence of visual clutter "
    "or broken design.\n\n"
    "3. ADVERTISING INTEGRITY — excessive, intrusive, or deceptive ads "
    "(pop-ups, banners mimicking editorial content). High ad-to-content ratio "
    "is a red flag.\n\n"
    "4. HEADLINE & CONTENT TONE — sensationalist, emotionally manipulative, "
    "or conspiratorial language; excessive capitalisation or alarming punctuation. "
    "Measured, neutral tone is a positive signal.\n\n"
    "5. EDITORIAL STRUCTURE — clear separation of news, opinion, and advertising; "
    "visible section navigation; structured article listings with dates and "
    "categories.\n\n"
    "After assessing each dimension, render a holistic binary verdict."
)

USER_ENGINEERED = (
    "Examine this website screenshot and determine whether the source is likely "
    "FACTUAL or NOT FACTUAL.\n\n"
    "Your response must begin with exactly one word on its own line: either "
    "\"FACTUAL\" or \"NOT FACTUAL\". Then provide a structured assessment "
    "covering all five dimensions (1–2 sentences each), followed by a brief "
    "holistic justification. Base everything solely on what is visible."
)

PROMPTS = {
    "simple": (SYSTEM_SIMPLE,     USER_SIMPLE),
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

                    status = "Good" if verdict != "UNKNOWN" else "?"
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