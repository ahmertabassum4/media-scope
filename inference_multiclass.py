# Run with:
#   python inference_multiclass.py --task factuality --prompt engineered
#   python inference_multiclass.py --task genre --prompt simple
#   python inference_multiclass.py --task bias --prompt engineered

import os
import csv
import base64
import json
import time
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()
API_KEY = os.getenv("OPENROUTER_KEY")

SAMPLE_INDEX = "snapshot_sample_index.csv"
IMAGE_DIR    = "snapshot-samples"
RESULTS_DIR  = Path("results")

MODELS = {
    "claude-sonnet-4-6":  "anthropic/claude-sonnet-4-6",
    "gpt-5-5":            "openai/gpt-5.5",
    "gemini-2-5-flash-lite": "google/gemini-2.5-flash-lite",
    "qwen2-5-vl-72b":     "qwen/qwen2.5-vl-72b-instruct",
    "gemma-4-31b":        "google/gemma-4-31b-it",
    "kimi-k2-6":          "moonshotai/kimi-k2.6",
}

# FACTUALITY (4-class)

FACT_SYSTEM = """
You are a media-credibility analyst conducting a visual audit of a news source's
homepage. Your only evidence is the single screenshot provided. Evaluate strictly
what is visible; do not use any prior knowledge of the outlet, and do not infer the
outlet's identity from its name, logo, or URL.

Your task is to predict the factual-reliability tier a professional rater would assign
the source, on a four-point scale:

  VERY LOW  — pervasive falsehoods, conspiratorial or pseudoscientific content,
              no discernible standards of evidence or correction.
  LOW       — frequent misleading framing, heavy bias-driven distortion, weak or
              absent sourcing, but not wholly fabricated.
  HIGH      — largely accurate, conventional reporting with visible attribution and
              ordinary editorial structure; minor bias acceptable.
  VERY HIGH — exemplary: sober, heavily sourced, transparent, the visual register of
              wire services, fact-checkers, research institutes, and pollsters.

Internalise three facts before you begin:

1. PROFESSIONAL APPEARANCE IS A WEAK SIGNAL. Unreliable outlets copy the design
   conventions of legitimate news — mastheads, section navigation, clean typography,
   bylines — to manufacture trust. Polish alone must not drive the tier. A slick site
   can be VERY LOW; a plain site can be HIGH.

2. CONTENT AND FRAMING ARE STRONG SIGNALS. What the page covers, and the language it
   uses, are far harder to disguise. Weight these most heavily when separating the
   reliable tiers (HIGH / VERY HIGH) from the unreliable tiers (LOW / VERY LOW).

3. THE TIER BOUNDARIES DIFFER IN KIND. The reliable/unreliable split (HIGH+ vs LOW-)
   turns mainly on content and framing. The within-pair distinctions are subtler:
   VERY HIGH vs HIGH turns on the *density* of sourcing, transparency, and sober tone;
   VERY LOW vs LOW turns on how *overtly* conspiratorial or pseudoscientific the
   visible content is. Reserve the extreme tiers for clear cases and default to the
   inner tiers (HIGH, LOW) when evidence is moderate.

Assess the homepage against five dimensions, in descending order of importance:

A. CONTENT, TOPICAL FOCUS & FRAMING (most diagnostic). Does visible coverage cluster
   around conspiratorial, pseudoscientific, hyper-partisan, or single-issue advocacy
   themes? One-sided thematic obsession pushes toward the LOW tiers; broad, mundane,
   multi-topic coverage pushes toward the HIGH tiers. Watch also for one-sided framing
   of ordinary topics and moral/identity-loaded language.

B. HEADLINE LANGUAGE, TONE & SPECIFICITY. Sensationalism, fear-based wording,
   ALL-CAPS, exclamations, loaded terms, and vague or low-vocabulary phrasing push
   down. Measured tone with concrete specificity — named people, places, dates,
   institutions — pushes up, and dense, sober specificity is the mark of VERY HIGH.

C. TRANSPARENCY & ATTRIBUTION. Bylines, datelines, mastheads, section navigation,
   visible ownership/funding/"About" disclosure. Presence is weakly positive (easily
   faked); conspicuous absence is mildly negative. Heavy, institutional transparency
   supports VERY HIGH.

D. ADVERTISING & MONETISATION INTEGRITY. High ad-to-content ratio, chumbox clickbait,
   and predatory ad categories (miracle cures, diet pills, "one weird trick," crypto)
   push down. Treat junk programmatic ads as a contributing signal, not decisive on
   their own — low-budget legitimate locals serve the same networks.

E. SYNTHETIC OR MANIPULATED IMAGERY. Plausibly AI-generated or doctored lead images
   presented as real push down. Be conservative; flag only clear cases.

Weigh the dimensions together, giving A and B the most influence, and render a single
tier.
""".strip()

FACT_USER = """
Examine this homepage screenshot and assign the source's factual-reliability tier.

Your response MUST begin with exactly one label on its own line, chosen from exactly
these four: VERY LOW, LOW, HIGH, VERY HIGH. Then give a brief structured assessment of
one to two sentences for each dimension (A Content, B Headlines, C Transparency,
D Ads, E Imagery), citing only what is visible. Conclude with one sentence naming the
signals that decided the tier, and in particular what separated it from the adjacent
tier. If a dimension is not visible, write "not visible" rather than guessing.
""".strip()

# (4-class)

GENRE_SYSTEM = """
You are a media-content analyst examining a news source's homepage. Your only evidence
is the single screenshot provided. Evaluate strictly what is visible; do not use prior
knowledge of the outlet, and do not infer its identity from its name, logo, or URL.

Your task is to classify the homepage into exactly one of four content categories,
based on the dominant character of the visible content:

  CONSPIRACY    — content organised around hidden-plot narratives: secret cabals,
                  cover-ups, "deep state," election-fraud or globalist framing,
                  persecution of the in-group, suppressed-truth and "they don't want
                  you to know" rhetoric, apocalyptic or ethnonationalist themes.
  PSEUDOSCIENCE — content organised around unfounded health, medical, or scientific
                  claims: anti-vaccine material, miracle cures, supplements and
                  detoxes, alternative medicine, creationist or anti-evolution
                  material, climate-science denial framed as fact.
  IMPOSTER      — a page built to LOOK like an ordinary, often local, news outlet but
                  which is in fact a low-substance partisan or astroturf operation
                  ("pink slime"). The deception is the defining feature: the visual
                  shell mimics a community newspaper or wire-style outlet, yet the
                  actual content is thin, nationally-syndicated talking points, or
                  one-directional political messaging dressed as local reporting. See
                  the tells below — this class is judged on the MISMATCH between an
                  ordinary-news veneer and hollow or covertly partisan substance.
  OTHER         — any genuine, non-deceptive outlet or institution: real local and
                  general news, politics, business, technology, education, government,
                  transparent advocacy groups, fact-checkers, pollsters, entertainment.
                  Ordinary bias or low budget does NOT move a source out of OTHER; the
                  three other classes each require a specific, visible defect.

Key principle: the distinction is about the CHARACTER of the content, not its polish or
ordinary political slant. Choose CONSPIRACY or PSEUDOSCIENCE only when the visible
headlines and topics are themselves substantially about hidden plots or unfounded
science. Choose IMPOSTER only when a legitimate-news veneer is contradicted by visible
signs of hollow or astroturfed substance. When none of these specific defects is
visible, the answer is OTHER — that is the broad default.

How to tell IMPOSTER from OTHER (the hardest boundary). Pink-slime tells, when visible:
a generic or templated local-news look reused without genuine local specificity;
a masthead place-name paired with content that is entirely national politics; very few,
very similar stories all pushing one political direction; absent or vague bylines,
"staff report" everywhere, no real local desks, sports, weather, or obituaries; recently
minted or formulaic outlet names ("County Record", "State Wire", "City Times") with no
visible institutional history. A real local paper, by contrast, shows genuine local
miscellany — community events, local sport, classifieds, named local reporters. When a
page looks like ordinary news AND shows real local/topical breadth, default to OTHER;
reserve IMPOSTER for a visible veneer-vs-substance mismatch.

How to tell IMPOSTER from CONSPIRACY/PSEUDOSCIENCE: imposter pages wear a MAINSTREAM news
costume; conspiracy and pseudoscience pages wear their fringe content openly. If the
fringe theme is overt, prefer CONSPIRACY or PSEUDOSCIENCE; if the page hides thin
partisan substance behind a normal-news shell, prefer IMPOSTER.

Assess the homepage against four dimensions, in descending order of importance:

A. TOPIC SELECTION & SUBSTANCE (most diagnostic). What do the visible headlines and
   section labels actually concern, and is there genuine breadth behind the shell?
   Hidden-plot themes → CONSPIRACY; unfounded health/science → PSEUDOSCIENCE; a
   local/ordinary-news veneer over thin or one-directional national content → IMPOSTER;
   genuine, varied reportage or transparent institutional content → OTHER.

B. HEADLINE FRAMING & VOCABULARY. Suppressed-truth/betrayal framing → CONSPIRACY;
   cure/toxin/natural-remedy framing → PSEUDOSCIENCE; uniform partisan talking points
   under a neutral-looking masthead → IMPOSTER; sourced, event-driven phrasing → OTHER.

C. ATTRIBUTION & LOCAL SPECIFICITY. Named reporters, real datelines, local desks,
   community miscellany support OTHER. Their conspicuous absence beneath a local-news
   look — anonymous "staff" content, no genuine local material — supports IMPOSTER.

D. IMAGERY, SECTIONING & PRODUCT MIX. Supplement/survival storefronts lean
   PSEUDOSCIENCE/CONSPIRACY; templated stock-photo local-news shells with no real local
   sections lean IMPOSTER; conventional, well-populated news sections lean OTHER.

Weigh these together, giving A and B the most influence, and render a single category.
""".strip()

GENRE_USER = """
Examine this homepage screenshot and classify its dominant content category.

Your response MUST begin with exactly one label on its own line, chosen from exactly
these four: CONSPIRACY, PSEUDOSCIENCE, IMPOSTER, OTHER. Then give a brief structured
assessment of one to two sentences for each dimension (A Topics/substance, B Framing,
C Attribution/local specificity, D Imagery/sections), citing only what is visible.
Conclude with one sentence naming the visible features that decided the category — and
for IMPOSTER vs OTHER, state what tipped the veneer-vs-substance judgement. If a
dimension is not visible, write "not visible" rather than guessing.
""".strip()

# BIAS (5-class)

BIAS_SYSTEM = """
You are a media-bias analyst examining a news source's homepage. Your only evidence is
the single screenshot provided. Evaluate strictly what is visible; do not use prior
knowledge of the outlet, and do not infer its identity from its name, logo, or URL.

Your task is to place the source on a five-point political-leaning scale:

  LEFT, LEFT-CENTER, LEAST BIASED, RIGHT-CENTER, RIGHT

READ THIS FIRST — THE HONEST DIFFICULTY OF THIS TASK. Political leaning is only weakly
visible in a static homepage screenshot, and far less visible than reliability or
genre. A single screenshot captures a small, time-specific slice of coverage; tone and
layout are largely shared across the spectrum; and centrist and centre-leaning outlets
are often visually indistinguishable. You should EXPECT most sources to be hard to
place and many to be genuinely ambiguous. Do not manufacture confidence the image does
not support. Specifically:

  - Default toward the centre of the scale (LEAST BIASED, or the adjacent
    LEFT-CENTER / RIGHT-CENTER) unless there is clear, visible directional evidence.
  - Reserve the poles (LEFT, RIGHT) for cases where the visible content is overtly and
    repeatedly partisan in a single direction — not merely for a strong tone.
  - A strident or sensational tone indicates intensity, NOT direction; do not let it
    pull you to a pole on its own. Direction must come from the substance of what is
    visibly endorsed or attacked.

Assess only what bears on DIRECTION, in descending order of usefulness:

A. ISSUE SELECTION & STANCE (most diagnostic, when visible). Which causes, figures, or
   policies do the visible headlines champion or attack? Sympathetic coverage of
   progressive causes/figures and criticism of conservatives leans LEFT; the mirror
   leans RIGHT. Consistent one-directional stance across several visible items is the
   strongest available cue.

B. FRAMING & LOADED LANGUAGE WITH A DIRECTION. Identity- or loyalty-laden vocabulary
   that consistently valorises one side and denigrates the other. Note the direction,
   and discount it heavily if only one or two items are visible.

C. TOPICAL EMPHASIS. A heavy, near-exclusive focus on themes characteristically
   associated with one side (e.g. immigration-as-threat, culture-war grievance vs.
   social-justice, climate, inequality framing) is a mild directional cue.

D. NEUTRALITY MARKERS. Broad multi-topic coverage, wire-service content, and
   even-handed sourcing support LEAST BIASED.

Weigh these together. If, after assessing them, the direction remains genuinely
unclear, choose LEAST BIASED and say so. A confident pole verdict requires multiple,
mutually reinforcing, clearly directional visible cues.
""".strip()

BIAS_USER = """
Examine this homepage screenshot and place the source on the political-leaning scale.

Your response MUST begin with exactly one label on its own line, chosen from exactly
these five: LEFT, LEFT-CENTER, LEAST BIASED, RIGHT-CENTER, RIGHT. Then give a brief
structured assessment of one to two sentences for each dimension (A Issue stance,
B Framing, C Topical emphasis, D Neutrality markers), citing only what is visible.
State explicitly how confident the visible evidence makes you, and conclude with one
sentence naming the directional signals — or noting their absence, in which case
LEAST BIASED is the appropriate default. If a dimension is not visible, write
"not visible" rather than guessing.
""".strip()

# Simple baselines

def _simple_system(label_desc):
    return (
        "You are evaluating a news source based solely on the visual appearance of its "
        "homepage screenshot. Do not use any prior knowledge about this outlet. " + label_desc
    )

SIMPLE = {
    "factuality": (
        _simple_system("Judge its factual-reliability tier."),
        "Examine this website screenshot and decide its factual-reliability tier.\n\n"
        "Your response must begin with exactly one label on its own line, chosen from: "
        "VERY LOW, LOW, HIGH, VERY HIGH. Then give a brief explanation (3-5 sentences), "
        "citing only what is visible in the screenshot.",
    ),
    "genre": (
        _simple_system("Classify its dominant content category."),
        "Examine this website screenshot and classify its content.\n\n"
        "Your response must begin with exactly one label on its own line, chosen from: "
        "CONSPIRACY, PSEUDOSCIENCE, IMPOSTER, OTHER. CONSPIRACY = hidden-plot content; "
        "PSEUDOSCIENCE = unfounded health/science claims; IMPOSTER = a page dressed up "
        "to look like ordinary (often local) news but really thin or covertly partisan "
        "'pink slime'; OTHER = any genuine news or institution. Then give a brief "
        "explanation (3-5 sentences), citing only what is visible in the screenshot.",
    ),
    "bias": (
        _simple_system("Judge its political leaning. Note that leaning is only weakly "
                       "visible from a screenshot; default to the centre when unsure."),
        "Examine this website screenshot and judge its political leaning.\n\n"
        "Your response must begin with exactly one label on its own line, chosen from: "
        "LEFT, LEFT-CENTER, LEAST BIASED, RIGHT-CENTER, RIGHT. Then give a brief "
        "explanation (3-5 sentences), citing only what is visible in the screenshot.",
    ),
}

TASKS = {
    "factuality": {
        "column": "factuality",
        "labels": ["VERY LOW", "LOW", "HIGH", "VERY HIGH"],
        "engineered": (FACT_SYSTEM, FACT_USER),
        "simple": SIMPLE["factuality"],
    },
    "genre": {
        "column": "genre_class",
        "labels": ["CONSPIRACY", "PSEUDOSCIENCE", "IMPOSTER", "OTHER"],
        "engineered": (GENRE_SYSTEM, GENRE_USER),
        "simple": SIMPLE["genre"],
    },
    "bias": {
        "column": "bias",
        "labels": ["LEFT", "LEFT-CENTER", "LEAST BIASED", "RIGHT-CENTER", "RIGHT"],
        "engineered": (BIAS_SYSTEM, BIAS_USER),
        "simple": SIMPLE["bias"],
    },
}


# Helper functions

def encode_image(path: str, max_height=8000):
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
        raise ValueError(f"No choices in response: {data.get('error', data)}")

    return data["choices"][0]["message"]["content"]


def parse_verdict(response_text, labels):
    """Pull the label from the model's first line for a given label space.

    Matches the longest label first so multi-word labels (e.g. "LEFT-CENTER",
    "VERY LOW", "NOT FACTUAL") win over their prefixes. Falls back to scanning
    the first line, then returns UNKNOWN.
    """
    first_line = response_text.strip().splitlines()[0].strip().upper()

    # Collapse hyphens/underscores to spaces on both sides so "LEFT-CENTER", "LEFT_CENTER" and "LEFT CENTER" all match the same label. Longest label first, so "LEFT-CENTER" wins over "LEFT".
    def flat(s):
        return s.replace("-", " ").replace("_", " ")

    norm = flat(first_line)
    ordered = sorted(labels, key=len, reverse=True)

    for label in ordered:
        if norm.startswith(flat(label)):
            return label
    for label in ordered:
        if flat(label) in norm:
            return label
    return "UNKNOWN"


def load_index(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run(task_name, prompt_name, limit):
    task = TASKS[task_name]
    system, user_text = task[prompt_name]
    labels = task["labels"]
    gt_column = task["column"]

    rows = load_index(SAMPLE_INDEX)
    if limit:
        rows = rows[:limit]

    RESULTS_DIR.mkdir(exist_ok=True)

    for short_name, model_id in MODELS.items():
        out_path = RESULTS_DIR / f"{short_name}_{task_name}_{prompt_name}.jsonl"

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
                    print(f"  [{i+1}/{len(rows)}] no label ({gt_column!r}={ground_truth!r}) — {media_name}")
                    continue

                img_filename = Path(row["image_path"]).name
                img_path = os.path.join(IMAGE_DIR, img_filename)

                if not os.path.exists(img_path):
                    print(f"  [{i+1}/{len(rows)}] MISSING IMAGE — {img_path}")
                    continue

                try:
                    image_b64 = encode_image(img_path)
                    response  = query_model(model_id, system, user_text, image_b64)
                    verdict   = parse_verdict(response, labels)

                    record = {
                        "media_name": media_name,
                        "model": short_name,
                        "task": task_name,
                        "prompt": prompt_name,
                        "verdict": verdict,
                        "full_response": response,
                        "ground_truth": ground_truth,
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
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--prompt", choices=["simple", "engineered"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run(args.task, args.prompt, args.limit)