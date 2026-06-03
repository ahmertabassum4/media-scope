# Media Source Snapshot Tool

A pipeline for capturing full-page screenshots of media sources, filtered by factuality rating from the [Media Bias/Fact Check (MBFC)](https://mediabiasfactcheck.com) dataset.

---

## Project Structure

```
Ugrip/
├── data/
│   ├── 2291eng_dedup/          # 2,290 JSON files, one per media source
│   └── 2291eng_dedup.zip       # Compressed archive of the above
├── output/                     # 866 successful screenshots (named by media name)
├── rerun_output/               # Screenshots from rerun of previously failed URLs
├── error_output/               # Screenshots from retries with 60,000ms timeout
├── errors/                     # Screenshots captured but showing error pages (bot blocks, 403s, etc.)
├── snapshot_index.csv          # Index of all successful screenshots with metadata + trustworthiness
├── errors.csv                  # Index of all failed/problematic URLs
├── shots/                      # Ad-hoc single snapshots (from snapshot.py CLI)
├── snapshot.py                 # Core screenshot engine (single URL)
├── batch_snapshot.py           # Batch runner over the full dataset
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

Sources with `MIXED` or `MOSTLY FACTUAL` are excluded from capture (1,293 excluded, 997 qualifying).

---

## Output CSVs

### `snapshot_index.csv`

Index of all **866 successfully captured** screenshots.

| Column | Description |
|---|---|
| `media_name` | Human-readable name of the media source |
| `url` | Homepage URL |
| `image_path` | Relative path to the PNG file in `output/` |
| `timestamp` | File modification time (`YYYY-MM-DD HH:MM:SS`) |
| `country` | Country of origin from MBFC data |
| `factuality` | Factuality rating (`HIGH`, `LOW`, `VERY HIGH`, `VERY LOW`) |
| `trustworthiness` | Binary label: `1` = HIGH or VERY HIGH factuality, `0` = LOW or VERY LOW |

### `errors.csv`

Index of all **179 failed or problematic** URLs (together with `snapshot_index.csv` covers all 997 qualifying sources minus duplicates).

| Column | Description |
|---|---|
| `filename` | Media source name |
| `url` | Homepage URL |
| `issue` | Category of the problem (see table below) |
| `timestamp` | File modification time if an error screenshot exists, else empty |

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
| Access denied (CDN) | CDN-level IP block |
| Blank — pure white | Page loaded but rendered nothing |
| Subscription popup only | Paywall/newsletter modal blocked content |
| Page load timeout | Page did not load within timeout |
| DNS failure | Domain does not exist |
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

Captures all 997 qualifying sources. Skips already-captured URLs by default.

```bash
# Standard run — skips already-captured
python batch_snapshot.py

# Re-capture everything
python batch_snapshot.py --force

# Retry only URLs that previously timed out
python batch_snapshot.py --retry-timeouts --timeout 60000

# Rerun from a CSV file (must have a 'url' column)
python batch_snapshot.py --from-csv errors.csv --output rerun_output --timeout 30000

# Retry all error-status URLs across multiple log files
python batch_snapshot.py \
  --from-logs output/batch_log.jsonl rerun_output/batch_log.jsonl \
  --output error_output --timeout 60000

# Preview URLs without capturing
python batch_snapshot.py --dry-run

# Tune parallelism (default: 4 workers)
python batch_snapshot.py --workers 6
```

All runs append results to `<output_dir>/batch_log.jsonl`.

---

## Image Naming

Screenshots are saved as:

```
<MediaName>_<YYYYMMDD>_<HHMMSS>.png
```

For example: `BBC_News_20260604_120000.png`, `New_York_Times_20260604_130500.png`.

The `snapshot_index.csv` maps every file back to its URL, country, and factuality rating.

---

## Workflow

```
1. python batch_snapshot.py
        ↓ captures all 997 qualifying sources → output/
        ↓ logs every attempt → output/batch_log.jsonl

2. Analyse images (cleaning.ipynb)
        ↓ detect blank/error/bot-blocked screenshots
        ↓ move bad ones → errors/
        ↓ update errors.csv

3. python batch_snapshot.py --from-csv errors.csv --output rerun_output
        ↓ retry failed URLs

4. python batch_snapshot.py \
     --from-logs output/batch_log.jsonl rerun_output/batch_log.jsonl \
     --output error_output --timeout 60000
        ↓ retry persistent errors with higher timeout
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

**Trustworthiness distribution (from captured sources):**

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
| Qualifying sources (non-MIXED) | 997 |
| Successfully captured (`output/`) | 866 |
| — Trustworthy (HIGH / VERY HIGH) | 716 |
| — Untrustworthy (LOW / VERY LOW) | 150 |
| Failed / problematic (`errors.csv`) | 179 |
| Error screenshots (`errors/`) | 81 |
