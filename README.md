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
├── Mixed_output/               # 1,177 screenshots — MIXED factuality sources
├── rerun_output/               # Screenshots from rerun of previously failed URLs
├── error_output/               # Screenshots from retries with 60,000ms timeout
├── errors/                     # Screenshots captured but showing error pages (bot blocks, 403s, etc.)
│
├── tmp/                        # Source index files (inputs to postprocessing.py)
│   ├── snapshot_index.csv      # 866 successful screenshots — non-MIXED sources
│   ├── mixed-snapshots.csv     # 1,177 successful screenshots — MIXED sources
│   ├── errors.csv              # 179 failed/problematic URLs — non-MIXED sources
│   ├── mixed-errors.csv        # 116 failed/problematic URLs — MIXED sources
│   └── Mixed_output/           # Mirror of Mixed_output/ used during post-processing
│
├── snapshots.csv               # FINAL merged snapshot index (2,050 rows)
├── error.csv                   # FINAL merged error log (240 rows)
│                               # Invariant: snapshots.csv + error.csv = 2,290 (one row per JSON source)
│
├── snapshot.py                 # Core screenshot engine (single URL)
├── batch_snapshot.py           # Batch runner — currently targets MIXED factuality sources
├── postprocessing.py           # Post-processing: build mixed-snapshots.csv, merge & deduplicate CSVs
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
| **Total** | **2,290** |

---

## Output CSVs

### Final merged outputs

These are the two authoritative files. Together they cover every JSON source exactly once:

> **`rows(snapshots.csv)` + `rows(error.csv)` = 2,290**

#### `snapshots.csv` — 2,050 rows

All successfully captured screenshots across all factuality classes.

| Column | Description |
|---|---|
| `media_name` | Human-readable name of the media source |
| `url` | Homepage URL |
| `image_path` | Relative path to the PNG file |
| `timestamp` | Capture time (`YYYY-MM-DD HH:MM:SS`) |
| `country` | Country of origin from MBFC data |
| `factuality` | Factuality rating (`HIGH`, `LOW`, `VERY HIGH`, `VERY LOW`, `MIXED`) |
| `trustworthiness` | Binary label: `1` = HIGH/VERY HIGH, `0` = LOW/VERY LOW, empty = MIXED (not yet labelled) |

#### `error.csv` — 240 rows

All sources that could not be successfully captured, across all factuality classes.

| Column | Description |
|---|---|
| `filename` | PNG filename if a screenshot was taken (visual error), else empty |
| `name` | Media source name |
| `url` | Homepage URL |
| `issue` | Category of the problem (see table below) |
| `timestamp` | Capture time if available, else empty |
| `factuality` | Factuality rating of the source |

---

### Source files (`tmp/`)

These are the per-class, per-status input files that `postprocessing.py` reads to produce the merged outputs above.

| File | Rows | Covers |
|---|---:|---|
| `tmp/snapshot_index.csv` | 866 | non-MIXED successful captures |
| `tmp/mixed-snapshots.csv` | 1,177 | MIXED successful captures |
| `tmp/errors.csv` | 179 | non-MIXED failures |
| `tmp/mixed-errors.csv` | 116 | MIXED failures |

**Note:** These four files sum to 2,338 — more than 2,290. `postprocessing.py` removes the 48 excess rows during merging (see [Deduplication](#deduplication) below).

---

### Issue categories (`error.csv`)

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

## Deduplication

The raw source files contain three categories of duplicate rows that `postprocessing.py` resolves before writing the final merged CSVs.

### 1 — Shared URLs (11 pairs)

Eleven pairs of JSON source files point to the same `media link` URL (two different MBFC entries for the same website). These are treated as two distinct sources sharing one captured image. Both sources appear in `snapshots.csv` with the same `image_path`.

```
https://emirates247.com        → 2 JSON entries
https://micatholictribune.com  → 2 JSON entries
... (9 more pairs)
```

### 2 — Cross-file duplicates (56 URLs)

56 URLs appear in both `snapshot_index.csv` (successful capture) and `errors.csv` (recorded from a failed retry run). The successful capture takes priority: these rows are kept in `snapshots.csv` and removed from `error.csv`.

### 3 — Internal duplicates (3 URLs)

Two URLs appear twice in the snapshot source files and one URL appears twice in the error source files, likely from overlapping capture runs. First occurrence is kept, duplicate is dropped.

### Summary

| Category | Rows removed |
|---|---:|
| Cross-file duplicates (snap wins over error) | 56 |
| Internal duplicates in error source files | 1 |
| Internal duplicates in snapshot source files | 2 |
| — offset by shared-URL pairs (each adds +1 row) | −11 |
| **Net reduction** | **48** |

`2,338 (raw) − 48 (removed) + 0 (added) = 2,290 ✓`

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

Builds `tmp/mixed-snapshots.csv` and produces the final deduplicated merged CSVs.

```bash
# Run all steps (recommended)
python postprocessing.py

# Only build tmp/mixed-snapshots.csv (from tmp/Mixed_output/batch_log.jsonl)
python postprocessing.py --step index

# Only merge snapshot CSVs → snapshots.csv
python postprocessing.py --step merge-snaps

# Only merge error CSVs → error.csv
python postprocessing.py --step merge-errors
```

| Step | Reads from `tmp/` | Writes |
|---|---|---|
| `index` | `Mixed_output/batch_log.jsonl`, `mixed-errors.csv`, source JSONs | `tmp/mixed-snapshots.csv` |
| `merge-snaps` | `snapshot_index.csv` + `mixed-snapshots.csv` | `snapshots.csv` |
| `merge-errors` | `errors.csv` + `mixed-errors.csv` + source JSONs | `error.csv` |

Running all steps also prints a verification line confirming `snapshots.csv + error.csv == 2,290`.

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
        ↓ record in tmp/errors.csv

3. python batch_snapshot.py --from-csv tmp/errors.csv --output rerun_output
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

2. Visual review (automated)
        ↓ detect blank/error/bot-blocked screenshots
        ↓ move bad ones → errors/
        ↓ record in tmp/mixed-errors.csv

3. python postprocessing.py
        ↓ builds tmp/mixed-snapshots.csv
        ↓ deduplicates and merges all source CSVs
        ↓ writes snapshots.csv (2,050 rows) + error.csv (240 rows)
        ↓ verifies: snapshots.csv + error.csv == 2,290 ✓
```

---

## `cleaning.ipynb`

A Jupyter notebook for post-processing and analysing the captured screenshots.

**What it does:**

1. Loads `tmp/snapshot_index.csv` into a pandas DataFrame
2. Strips timestamps from image filenames and updates paths in the CSV
3. Parses and formats the `timestamp` column to `YYYY-MM-DD HH:MM:SS`
4. Adds a `trustworthiness` binary column (`1` = HIGH/VERY HIGH factuality, `0` = LOW/VERY LOW)
5. Saves the enriched DataFrame back to `tmp/snapshot_index.csv`
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
| Failed / problematic (`tmp/errors.csv`) | 179 |
| **MIXED captures** | |
| Successfully captured (`Mixed_output/`) | 1,177 |
| — trustworthiness | *not yet labelled* |
| Failed / problematic (`tmp/mixed-errors.csv`) | 116 |
| — Capture-time errors (network/SSL/timeout) | 80 |
| — Visual errors (bot block, blank, 403, etc.) | 36 |
| **Final merged outputs** | |
| Clean screenshots (`snapshots.csv`) | **2,050** |
| Error records (`error.csv`) | **240** |
| **Total (= JSON source count)** | **2,290 ✓** |
