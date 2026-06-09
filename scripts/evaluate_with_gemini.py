import csv
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import google.genai as genai
import PIL.Image
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS_DIR = Path(os.environ.get("SCREENSHOTS_DIR", PROJECT_ROOT / "data" / "screenshots" / "factuality"))
METADATA_DIR = Path(os.environ.get("METADATA_DIR", PROJECT_ROOT / "data" / "metadata" / "media_metadata"))
RESULTS_FILE = Path(os.environ.get("RESULTS_FILE", PROJECT_ROOT / "results" / "gemini" / "gemini_features.csv"))
MODEL_NAME = "gemini-2.5-flash"
RPM_LIMIT = 60
MAX_WORKERS = 10
MAX_IMG_DIM = 6000
MAX_INLINE_BYTES = 5 * 1024 * 1024
VALID_LABELS = {"VERY HIGH", "HIGH", "LOW", "VERY LOW"}


SIGNAL_KEYS = [
    "s01_named_bylines",
    "s02_personal_brand",
    "s03_editorial_hierarchy",
    "s04_loaded_headlines",
    "s05_accusatory_questions",
    "s06_breaking_misuse",
    "s07_biased_categories",
    "s08_opinion_news_blurred",
    "s09_standard_sections",
    "s10_incoherent_mixing",
    "s11_merchandise_section",
    "s12_timestamps",
    "s13_local_features",
    "s14_ads_labeled",
    "s15_sponsored_distinguished",
    "s16_fear_based_ads",
    "s17_persecution_donation",
    "s18_ideological_mission",
    "s19_fringe_platforms",
    "s20_distrust_tagline",
]
FIELDNAMES = (
    ["filename", "ground_truth", "outlet_type"]
    + SIGNAL_KEYS
    + ["n_red_flags", "n_trust_signals", "gemini_verdict", "raw_response"]
)


def normalize_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())

SYSTEM_PROMPT = """You are a media credibility analyst. Your sole task is to evaluate
the factuality and journalistic trustworthiness of a media outlet
based EXCLUSIVELY on visual evidence present in a screenshot of
its front page. You must not use any prior knowledge, memory, or
external information about the outlet. Every claim in your
analysis must be directly traceable to something visible in
the image.

---

## YOUR EVALUATION FRAMEWORK

Assess the screenshot across the following five dimensions,
each grounded in established journalistic principles of
fairness, accuracy, transparency, accountability, and
independence.

---

### DIMENSION 1 — AUTHORSHIP & ACCOUNTABILITY
Principle: Credible journalism requires named, identifiable
journalists who are accountable for their work.

Look for:
1. Are journalist bylines visible and real (full names) or
   absent/pseudonymous (e.g. "Quoth the Raven")?
2. Is the outlet structured around an institution or a single
   personal brand (e.g. "Dr. T's Store", "Dr. T's Website")?
3. Is there evidence of editorial hierarchy (editors, editorial
   board, masthead)?

Score each signal: PRESENT / ABSENT / UNCLEAR

---

### DIMENSION 2 — HEADLINE & LANGUAGE NEUTRALITY
Principle: Factual journalism presents information descriptively.
Opinion and analysis must be clearly labeled and separated from
news reporting.

Look for:
4. Are headlines factual and descriptive, or emotionally loaded
   and conspiratorial?
5. Are question headlines used to imply accusations without
   evidence (e.g. "Who Is Controlling Him?")?
6. Is "BREAKING:" applied to stories with timestamps suggesting
   they are not recent?
7. Do article category or tag labels reveal editorial bias
   (e.g. "Gaslighting", "Gain of Function" as categories)?
8. Is opinion clearly separated from news, or blurred together?

Score each signal: PRESENT / ABSENT / UNCLEAR

---

### DIMENSION 3 — STRUCTURAL LEGITIMACY
Principle: Established journalistic outlets follow recognizable
institutional structures that reflect professional editorial
standards.

Look for:
9. Are standard newspaper sections visible (News, Sports,
   Obituaries, Public Notices, Archive, Comics)?
10. Is there an unusual or incoherent mixing of content types
    (e.g. Short Stories alongside war coverage)?
11. Is there a merchandise or store section in the navigation?
12. Are timestamps present on articles, indicating an active
    newsroom?
13. Are local community features visible (weather widget,
    Public Notices, eNewspaper), indicating roots in a
    specific community?

Score each signal: PRESENT / ABSENT / UNCLEAR

---

### DIMENSION 4 — TRANSPARENCY & INDEPENDENCE
Principle: Credible journalism discloses funding, conflicts of
interest, and the distinction between editorial and commercial
content. Outlets must not serve funders over audiences.

Look for:
14. Are advertisements clearly and explicitly labeled as such?
15. Is sponsored content visually distinguished from editorial
    content with clear markers?
16. What type of products are advertised? (Neutral commercial
    ads vs. fear-based products such as detox supplements,
    survival gear, or gold investment schemes)
17. Is there donation language that positions the outlet as
    persecuted (e.g. "Support Free Speech — Donate!") vs.
    standard subscription/support framing?
18. Is there a visible mission statement, and if so, does it
    reveal an ideological agenda (e.g. "Western restoration",
    "alternative health")?

Score each signal: PRESENT / ABSENT / UNCLEAR

---

### DIMENSION 5 — PLATFORM & SOCIAL SIGNALS
Principle: The platforms an outlet chooses to distribute on
and the language it uses to describe itself reveal its
relationship with mainstream journalistic norms.

Look for:
19. Which social media platforms are promoted?
    (Mainstream platforms vs. platforms associated with
    content moderation avoidance such as Bitchute or Telegram)
20. Does the outlet's tagline, branding, or self-description
    signal distrust of institutions, nihilism, or ideological
    positioning?

Score each signal: PRESENT / ABSENT / UNCLEAR

---

## OUTPUT FORMAT

Provide your analysis in the following structure:

**OUTLET TYPE**
State what kind of outlet this appears to be based solely on
visual evidence (e.g. local newspaper, trade publication,
activist organization, alternative media blog, personal brand).

**DIMENSION SCORES**
For each of the five dimensions, list the signals you observed,
what you saw, and your score (PRESENT / ABSENT / UNCLEAR).
Only reference things actually visible in the screenshot.

**RED FLAGS**
List any specific visual elements that indicate low factuality,
quoting or describing exactly what you see.

**TRUST SIGNALS**
List any specific visual elements that indicate high factuality,
quoting or describing exactly what you see.

**VERDICT**
Choose exactly one of four ratings:
- VERY HIGH FACTUALITY — exemplary institutional structure,
  rigorous sourcing, full transparency, named journalists,
  clear editorial standards
- HIGH FACTUALITY — solid institutional structure, mostly
  neutral language, transparent advertising, named journalists
- LOW FACTUALITY — multiple red flags across dimensions,
  ideological positioning, poor sourcing, limited transparency
- VERY LOW FACTUALITY — severe red flags, pseudoscience or
  conspiracy content, anonymous authorship, no editorial
  standards evident

Follow the verdict with a one-paragraph summary of your
reasoning, referencing only what was visible in the screenshot.

---

## STRICT RULES

- Never name or identify the outlet from memory
- Never use knowledge of the outlet from training data
- Never search the internet
- Every claim must reference something visible in the image
- If something is not visible, mark it UNCLEAR — do not assume
- Distinguish carefully between news content and advertising
- Treat advocacy organizations as a separate category —
  they can be transparent and legitimate without being neutral

---

## MACHINE-READABLE OUTPUT (REQUIRED)

After all of the prose above, end your response with a SINGLE fenced
```json code block — and nothing after it — using EXACTLY this schema
and these keys. Every signal value MUST be one of "PRESENT", "ABSENT",
or "UNCLEAR". PRESENT means the described condition is observed in the
screenshot; UNCLEAR means there is not enough visible to tell.

```json
{
  "outlet_type": "<short phrase, e.g. local newspaper>",
  "signals": {
    "s01_named_bylines": "PRESENT = real, full-name journalist bylines are visible",
    "s02_personal_brand": "PRESENT = built around a single personal brand, not an institution",
    "s03_editorial_hierarchy": "PRESENT = editors, an editorial board, or a masthead are visible",
    "s04_loaded_headlines": "PRESENT = headlines are emotionally loaded or conspiratorial",
    "s05_accusatory_questions": "PRESENT = question headlines imply accusations without evidence",
    "s06_breaking_misuse": "PRESENT = 'BREAKING:' is applied to stories that are not recent",
    "s07_biased_categories": "PRESENT = category or tag labels reveal editorial bias",
    "s08_opinion_news_blurred": "PRESENT = opinion is NOT clearly separated from news",
    "s09_standard_sections": "PRESENT = standard newspaper sections are visible (News, Sports, Obituaries...)",
    "s10_incoherent_mixing": "PRESENT = there is incoherent mixing of content types",
    "s11_merchandise_section": "PRESENT = a merchandise or store section appears in the navigation",
    "s12_timestamps": "PRESENT = article timestamps are present",
    "s13_local_features": "PRESENT = local community features are visible (weather, Public Notices, eNewspaper)",
    "s14_ads_labeled": "PRESENT = advertisements are clearly labeled as ads",
    "s15_sponsored_distinguished": "PRESENT = sponsored content is visually distinguished from editorial content",
    "s16_fear_based_ads": "PRESENT = fear-based products are advertised (detox supplements, survival gear, gold schemes)",
    "s17_persecution_donation": "PRESENT = donation language frames the outlet as persecuted",
    "s18_ideological_mission": "PRESENT = a mission statement reveals an ideological agenda",
    "s19_fringe_platforms": "PRESENT = it promotes content-moderation-avoiding platforms (Bitchute, Telegram, Gab)",
    "s20_distrust_tagline": "PRESENT = the tagline/branding signals distrust of institutions or ideological positioning"
  },
  "n_red_flags": <integer count of distinct red-flag elements you listed>,
  "n_trust_signals": <integer count of distinct trust-signal elements you listed>,
  "verdict": "<one of: VERY HIGH | HIGH | LOW | VERY LOW>"
}
```

In your actual output, replace each signal's description with ONLY the
chosen value ("PRESENT" / "ABSENT" / "UNCLEAR"), replace the angle-bracket
placeholders with real values, and make the block valid JSON."""



def parse_verdict(text: str):
    upper = text.upper()

    for line in upper.splitlines():
        if "VERY HIGH FACTUALITY" in line:
            return "VERY HIGH"
        if "VERY LOW FACTUALITY" in line:
            return "VERY LOW"
        if "HIGH FACTUALITY" in line:
            return "HIGH"
        if "LOW FACTUALITY" in line:
            return "LOW"

    if "VERY HIGH FACTUALITY" in upper:
        return "VERY HIGH"
    if "VERY LOW FACTUALITY" in upper:
        return "VERY LOW"
    if "HIGH FACTUALITY" in upper:
        return "HIGH"
    if "LOW FACTUALITY" in upper:
        return "LOW"
    return "UNCLEAR"


def _safe_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _scan_last_json_object(text: str):
    decoder = json.JSONDecoder()
    found = None
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    found = obj
            except json.JSONDecodeError:
                continue
    return found


def load_image_bytes(path: Path):
    raw = path.read_bytes()
    if len(raw) <= MAX_INLINE_BYTES:
        return "image/png", raw

    img = PIL.Image.open(io.BytesIO(raw)).convert("RGB")
    if max(img.size) > MAX_IMG_DIM:
        scale = MAX_IMG_DIM / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "image/jpeg", buf.getvalue()


def parse_features(text: str) -> dict:
    feats = {k: "UNCLEAR" for k in SIGNAL_KEYS}
    feats["outlet_type"] = ""
    feats["n_red_flags"] = 0
    feats["n_trust_signals"] = 0
    feats["gemini_verdict"] = ""

    data = None
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            data = json.loads(fenced[-1])
        except Exception:
            data = None
    if data is None:
        data = _scan_last_json_object(text)

    if isinstance(data, dict):
        feats["outlet_type"] = str(data.get("outlet_type", "")).strip()
        signals = data.get("signals", {})
        if isinstance(signals, dict):
            for k in SIGNAL_KEYS:
                v = str(signals.get(k, "UNCLEAR")).strip().upper()
                feats[k] = v if v in ("PRESENT", "ABSENT", "UNCLEAR") else "UNCLEAR"
        feats["n_red_flags"] = _safe_int(data.get("n_red_flags", 0))
        feats["n_trust_signals"] = _safe_int(data.get("n_trust_signals", 0))
        verdict = str(data.get("verdict", "")).strip().upper()
        feats["gemini_verdict"] = verdict if verdict in VALID_LABELS else parse_verdict(text)
    else:
        feats["gemini_verdict"] = parse_verdict(text)

    return feats


def load_done(results_file: Path) -> dict:
    if not results_file.exists():
        return {}
    done = {}
    with open(results_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done[row["filename"]] = row
    return done


def _build_metadata_lookup(metadata_dir: Path) -> dict:
    lookup = {}
    for jf in metadata_dir.glob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            factuality = data.get("factuality", "").upper().strip()
            if factuality in VALID_LABELS:
                names = {jf.stem, jf.stem.replace("_", " "), data.get("media name", "")}
                for name in names:
                    if str(name).strip():
                        lookup[str(name)] = factuality
                        lookup[str(name).replace("_", " ")] = factuality
                        lookup[normalize_name(name)] = factuality
        except Exception:
            pass
    return lookup


def screenshot_candidates(stem: str) -> list[str]:
    base = re.sub(r"[_-]\d{8}[_-]\d{6}$", "", stem)
    return [
        stem,
        base,
        stem.replace("_", " "),
        base.replace("_", " "),
        normalize_name(stem),
        normalize_name(base),
    ]


def collect_tasks(screenshots_dir: Path, done: dict) -> list:
    meta = _build_metadata_lookup(METADATA_DIR)
    tasks = []
    for png in sorted(screenshots_dir.rglob("*.png")):
        if png.name in done:
            continue
        stem = png.stem
        truth = next((meta[name] for name in screenshot_candidates(stem) if name in meta), None)
        if truth is None:
            continue
        tasks.append((png.name, png, truth))
    return tasks


def main():
    if not SCREENSHOTS_DIR.exists():
        print(f"ERROR: screenshots folder not found: {SCREENSHOTS_DIR}")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        client = genai.Client()

    done = load_done(RESULTS_FILE)
    tasks = collect_tasks(SCREENSHOTS_DIR, done)
    all_results = list(done.values())

    print(f"Model       : {MODEL_NAME}")
    print(f"Already done: {len(done)}")
    print(f"To evaluate : {len(tasks)}")
    if tasks:
        eta_min = len(tasks) / RPM_LIMIT
        print(f"Est. time   : ~{eta_min:.0f} min at {RPM_LIMIT} RPM\n")

    csv_lock   = threading.Lock()
    counter    = {"n": len(done)}
    total      = len(done) + len(tasks)

    def evaluate_one(filename, path, truth):
        mime, data = load_image_bytes(path)
        image_part = {"inline_data": {"mime_type": mime, "data": data}}
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[SYSTEM_PROMPT, image_part],
        )
        response_text = response.text
        feats = parse_features(response_text)
        return {
            "filename":     filename,
            "ground_truth": truth,
            "raw_response": response_text.replace("\n", " "),
            **feats,
        }

    write_header = not RESULTS_FILE.exists() or os.path.getsize(RESULTS_FILE) == 0
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        futures = {}
        pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        try:
            for filename, path, truth in tasks:
                fut = pool.submit(evaluate_one, filename, path, truth)
                futures[fut] = filename

            for fut in as_completed(futures):
                filename = futures[fut]
                try:
                    row = fut.result()
                    verdict = row.get("gemini_verdict") or "UNCLEAR"
                    with csv_lock:
                        counter["n"] += 1
                        n = counter["n"]
                        print(f"[{n}/{total}] {filename} ... {verdict}")
                        writer.writerow(row)
                        csvfile.flush()
                        all_results.append(row)
                except Exception as e:
                    with csv_lock:
                        counter["n"] += 1
                        print(f"[{counter['n']}/{total}] {filename} ... ERROR: {e}")
        except KeyboardInterrupt:
            print("\nInterrupted — shutting down. Already-completed results are saved.")
            pool.shutdown(wait=False, cancel_futures=True)
            return
        else:
            pool.shutdown(wait=True)

    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
    )

    evaluated = [r for r in all_results if r.get("gemini_verdict") in VALID_LABELS]
    unclear   = [r for r in all_results if r.get("gemini_verdict") not in VALID_LABELS]

    print("\n" + "=" * 50)
    print("GEMINI-VERDICT BASELINE (zero-shot, the layer to beat)")
    print("=" * 50)
    print(f"Total outlets             : {len(all_results)}")
    print(f"UNCLEAR verdicts (skipped): {len(unclear)}")
    print(f"Evaluated                 : {len(evaluated)}")

    if not evaluated:
        print("No evaluated results yet.")
        return

    y_true = [r["ground_truth"]   for r in evaluated]
    y_pred = [r["gemini_verdict"] for r in evaluated]
    labels = ["VERY HIGH", "HIGH", "LOW", "VERY LOW"]

    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall accuracy  : {acc*100:.1f}%")
    print(f"Macro F1          : {f1_score(y_true, y_pred, labels=labels, average='macro',  zero_division=0)*100:.1f}%")
    print(f"Weighted F1       : {f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)*100:.1f}%")

    print("\n--- Per-class metrics ---")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print("--- Confusion matrix (rows=true, cols=pred) ---")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    col_w = max(len(l) for l in labels) + 2
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = f"{row_label:>{col_w}}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(len(labels)))
        print(row)

    def to_binary(label):
        return "reliable" if label in ("VERY HIGH", "HIGH") else "unreliable"

    y_true_bin = [to_binary(v) for v in y_true]
    y_pred_bin = [to_binary(v) for v in y_pred]
    bin_labels = ["reliable", "unreliable"]
    print("\n--- Binary collapse (reliable vs unreliable) ---")
    print(classification_report(y_true_bin, y_pred_bin, labels=bin_labels, zero_division=0))

    print(f"Full results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
