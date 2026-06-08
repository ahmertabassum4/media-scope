# Media Source Snapshot Tool

A pipeline for capturing full-page screenshots of media sources, filtered by factuality rating from the [Media Bias/Fact Check (MBFC)](https://mediabiasfactcheck.com) dataset.

---

## Project Structure

```
Ugrip/
├── data/
│   ├── 2291eng_dedup/          # 2,290 JSON files, one per media source
│   └── 2291eng_dedup.zip       # Compressed archive of the above
│
├── output/                     # 866 screenshots — HIGH / LOW / VERY HIGH / VERY LOW sources
├── Mixed_output/               # ~1,177 screenshots — MIXED factuality sources
├── rerun_output/               # Screenshots from rerun of previously failed URLs
├── error_output/               # Screenshots from retries with 60,000ms timeout
├── errors/                     # Screenshots captured but showing error pages (bot blocks, 403s, etc.)
│
├── snapshot_index.csv          # Index of 866 successful screenshots (non-MIXED)
├── mixed-snapshots.csv         # Index of ~1,177 successful screenshots (MIXED only)
├── snapshots.csv               # Merged index: snapshot_index.csv + mixed-snapshots.csv
│
├── errors.csv                  # Failed/problematic URLs from non-MIXED sources (179 entries)
├── mixed-errors.csv            # Failed/problematic URLs from MIXED sources (116 entries)
├── error.csv                   # Merged errors: errors.csv + mixed-errors.csv
│
├── snapshot.py                 # Core screenshot engine (single URL)
├── batch_snapshot.py           # Batch runner — currently targets MIXED factuality sources
├── postprocessing.py           # Post-processing: build mixed-snapshots.csv, merge CSVs
├── cleaning.ipynb              # Notebook for cleaning, enriching, and analysing snapshot metadata
└── README.md
```

---

## Data Format

Each file in `data/2291eng_dedup/` is a JSON object for one media source:

```json
{
  "genre":          "General News",
  "media name":     "7NEWS",
  "mbfc link":      "https://mediabiasfactcheck.com/7news/",
  "media link":     "https://7news.com.au",
  "label":          "Mixed",
  "bias":           "RIGHT-CENTER",
  "factuality":     "MIXED",
  "country":        "",
  "freedom_rating": "MOSTLY FREE",
  "media_type":     "TV",
  "traffic":        "",
  "credibility":    "MEDIUM CREDIBILITY",
  "wikipedia_article": "...",
  "articles":       [ { "id": "...", "link": "...", "text": "..." }, ... ]
}
```

**Factuality values** present in the dataset:

| Value | Count |
|---|---:|
| MIXED | 1,293 |
| HIGH | 801 |
| LOW | 120 |
| VERY LOW | 61 |
| VERY HIGH | 15 |

---

## Output CSVs

### `snapshot_index.csv`

Index of all **866 successfully captured** screenshots from non-MIXED sources.

| Column | Description |
|---|---|
| `media_name` | Human-readable name of the media source |
| `url` | Homepage URL |
| `image_path` | Relative path to the PNG file in `output/` |
| `timestamp` | Capture time (`YYYY-MM-DD HH:MM:SS`) |
| `country` | Country of origin from MBFC data |
| `factuality` | Factuality rating (`HIGH`, `LOW`, `VERY HIGH`, `VERY LOW`) |
| `trustworthiness` | Binary label: `1` = HIGH or VERY HIGH, `0` = LOW or VERY LOW |

### `mixed-snapshots.csv`

Index of all **~1,177 successfully captured** screenshots from MIXED sources. Same columns as `snapshot_index.csv`. `trustworthiness` is empty for all rows (not yet labelled).

### `snapshots.csv`

Full merged index combining `snapshot_index.csv` and `mixed-snapshots.csv`. Same columns. Use this for training/analysis across all factuality classes.

---

### `errors.csv`

Failed or problematic URLs from **non-MIXED** sources (179 entries).

| Column | Description |
|---|---|
| `filename` | Media source name |
| `url` | Homepage URL |
| `issue` | Category of the problem (see table below) |
| `timestamp` | Capture time if an error screenshot exists, else empty |

### `mixed-errors.csv`

Failed or problematic URLs from **MIXED** sources (116 entries). Same issue categories plus two additional columns.

| Column | Description |
|---|---|
| `filename` | PNG filename if a screenshot was taken, else empty |
| `name` | Media source name |
| `url` | Homepage URL |
| `issue` | Category of the problem |
| `factuality` | Always `MIXED` |

### `error.csv`

Full merged error log combining `errors.csv` and `mixed-errors.csv` (295 total entries). Columns are the union of both sources; missing fields are left empty.

| Column | Description |
|---|---|
| `filename` | PNG filename if a screenshot was taken, else empty |
| `name` | Media source name |
| `url` | Homepage URL |
| `issue` | Category of the problem |
| `timestamp` | Capture time if available |
| `factuality` | Factuality rating of the source |

**Issue categories:**

| Issue | Cause |
|---|---|
| Bot detected — Challenge failed | Anti-bot service blocked headless browser |
| Bot check — Cloudflare Verify you are human | Cloudflare Turnstile requires human interaction |
| Bot check — wp.com Checking your browser | WordPress.com security check |
| 403 Forbidden | Server refused access |
| Cloudflare — Sorry, you have been blocked | IP/fingerprint blocked by Cloudflare WAF |
| Cloudflare — Invalid SSL certificate (526) | Origin server has invalid SSL cert |
| Cloudflare — Web server is down (521) | Origin server not responding |
| Account suspended | Hosting account suspended |
| Blank — pure white | Page loaded but rendered nothing |
| Domain parked / for sale | Domain no longer active |
| Domain hijacked | Domain redirects to unrelated content |
| Subscription popup only | Paywall/newsletter modal blocked content |
| Page load timeout | Page did not load within timeout |
| DNS failure | Domain does not exist |
| SSL/TLS certificate error | Certificate invalid or expired |
| Connection reset / timed out | Network-level failure |

---

## Setup

### Requirements

- Python 3.10+
- A virtual environment (recommended)

### Install

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install playwright==1.60.0 Pillow

# 3. Install the Chromium browser used by Playwright
playwright install chromium
```

---

## Usage

### Single URL — `snapshot.py`

Capture one website manually:

```bash
python snapshot.py https://www.bbc.com
python snapshot.py https://www.reuters.com --output shots --full-page
python snapshot.py https://example.com --width 1440 --height 900 --format jpeg
```

| Flag | Default | Description |
|---|---|---|
| `url` | *(required)* | Website URL to capture |
| `-o / --output` | `output` | Directory to save the screenshot |
| `--full-page` | off | Capture the entire scrollable page |
| `--width` | `1366` | Viewport width in pixels |
| `--height` | `768` | Viewport height in pixels |
| `--format` | `png` | Image format: `png` or `jpeg` |
| `--timeout` | `45000` | Page load timeout in milliseconds |
| `--no-scroll` | off | Disable pre-capture scroll (faster, may miss lazy images) |
| `--settle` | `2000` | Extra wait in ms after scrolling, before capture |

Screenshots are saved as `<MediaName>_<timestamp>.png` (e.g. `BBC_News_20260604_120000.png`).

---

### Batch Run — `batch_snapshot.py`

Currently configured to capture all **MIXED** factuality sources (1,293 sources). Skips already-captured URLs by default.

```bash
# Capture all MIXED sources → Mixed_output/
python batch_snapshot.py --output Mixed_output

# Re-capture everything
python batch_snapshot.py --output Mixed_output --force

# Retry only URLs that previously timed out
python batch_snapshot.py --output Mixed_output --retry-timeouts --timeout 60000

# Retry all error-status URLs across log files
python batch_snapshot.py \
  --from-logs Mixed_output/batch_log.jsonl \
  --output Mixed_output --timeout 60000

# Preview URLs without capturing
python batch_snapshot.py --dry-run

# Tune parallelism (default: 4 workers)
python batch_snapshot.py --workers 8
```

To switch to a different factuality class, change `TARGET_FACTUALITY` at the top of `batch_snapshot.py`.

All runs append results to `<output_dir>/batch_log.jsonl`.

---

### Post-processing — `postprocessing.py`

Builds `mixed-snapshots.csv` and produces the final merged CSVs.

```bash
# Run all steps
python postprocessing.py

# Only build mixed-snapshots.csv (from Mixed_output/batch_log.jsonl)
python postprocessing.py --step index

# Only merge error CSVs → error.csv
python postprocessing.py --step merge-errors

# Only merge snapshot CSVs → snapshots.csv
python postprocessing.py --step merge-snaps
```

| Step | Input | Output |
|---|---|---|
| `index` | `Mixed_output/batch_log.jsonl`, `mixed-errors.csv`, source JSONs | `mixed-snapshots.csv` |
| `merge-errors` | `errors.csv` + `mixed-errors.csv` | `error.csv` |
| `merge-snaps` | `snapshot_index.csv` + `mixed-snapshots.csv` | `snapshots.csv` |

---

## Image Naming

Screenshots are saved as:

```
<MediaName>_<YYYYMMDD>_<HHMMSS>.png
```

For example: `BBC_News_20260604_120000.png`, `New_York_Times_20260604_130500.png`.

---

## Workflow

### Phase 1 — Non-MIXED sources (HIGH / LOW / VERY HIGH / VERY LOW)

```
1. python batch_snapshot.py
        ↓ captures all 997 qualifying sources → output/
        ↓ logs every attempt → output/batch_log.jsonl

2. Review images (cleaning.ipynb)
        ↓ detect blank/error/bot-blocked screenshots
        ↓ move bad ones → errors/
        ↓ record in errors.csv

3. python batch_snapshot.py --from-csv errors.csv --output rerun_output
        ↓ retry failed URLs

4. python batch_snapshot.py \
     --from-logs output/batch_log.jsonl rerun_output/batch_log.jsonl \
     --output error_output --timeout 60000
        ↓ retry persistent errors with higher timeout
```

### Phase 2 — MIXED sources

```
1. python batch_snapshot.py --output Mixed_output
        ↓ captures all 1,293 MIXED sources → Mixed_output/
        ↓ logs every attempt → Mixed_output/batch_log.jsonl

2. Visual review (automated via workflow)
        ↓ detect blank/error/bot-blocked screenshots
        ↓ move bad ones → errors/
        ↓ record in mixed-errors.csv

3. python postprocessing.py
        ↓ builds mixed-snapshots.csv (clean MIXED screenshots index)
        ↓ merges errors.csv + mixed-errors.csv → error.csv
        ↓ merges snapshot_index.csv + mixed-snapshots.csv → snapshots.csv
```

---

## `cleaning.ipynb`

A Jupyter notebook for post-processing and analysing the captured screenshots.

**What it does:**

1. Loads `snapshot_index.csv` into a pandas DataFrame
2. Strips timestamps from image filenames and updates paths in the CSV
3. Parses and formats the `timestamp` column to `YYYY-MM-DD HH:MM:SS`
4. Adds a `trustworthiness` binary column (`1` = HIGH/VERY HIGH factuality, `0` = LOW/VERY LOW)
5. Saves the enriched DataFrame back to `snapshot_index.csv`
6. Plots the trustworthiness distribution across the captured dataset

**Trustworthiness distribution (non-MIXED sources):**

| Factuality | Count | Trustworthiness |
|---|---:|---|
| HIGH | 705 | 1 |
| VERY HIGH | 11 | 1 |
| LOW | 97 | 0 |
| VERY LOW | 53 | 0 |

---

## Current Statistics

| Metric | Count |
|---|---:|
| Total sources in dataset | 2,290 |
| — MIXED sources | 1,293 |
| — Non-MIXED sources (HIGH / LOW / VERY HIGH / VERY LOW) | 997 |
| **Non-MIXED captures** | |
| Successfully captured (`output/`) | 866 |
| — Trustworthy (HIGH / VERY HIGH) | 716 |
| — Untrustworthy (LOW / VERY LOW) | 150 |
| Failed / problematic (`errors.csv`) | 179 |
| **MIXED captures** | |
| Successfully captured (`Mixed_output/`) | ~1,177 |
| — trustworthiness | *not yet labelled* |
| Failed / problematic (`mixed-errors.csv`) | 116 |
| — Capture-time errors (network/SSL/timeout) | 80 |
| — Visual errors (bot block, blank, 403, etc.) | 36 |
| **Merged totals** | |
| Total clean screenshots (`snapshots.csv`) | ~2,043 |
| Total error records (`error.csv`) | 295 |
