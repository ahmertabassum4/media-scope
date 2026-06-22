# Media Source Snapshot Tool

A pipeline for capturing full-page screenshots of media sources using the [Media Bias/Fact Check (MBFC)](https://mediabiasfactcheck.com) dataset, covering three source groups: non-MIXED factuality sources, MIXED factuality sources, and questionable sources.

---

## Project Structure

```
Ugrip/
├── raw_data/
│   ├── 2291eng_dedup/              # 2,290 JSON files, one per media source
│   └── 2291eng_dedup.zip           # Compressed archive of the above
│
├── output/                         # 866 screenshots — HIGH / LOW / VERY HIGH / VERY LOW sources
├── errors/                         # 231 screenshots showing error pages (bot blocks, 403s, etc.)
│
├── tmp/
│   ├── questionable_sources/       # 1,676 screenshots — questionable sources (timestamps stripped)
│   ├── questionable_sources_with_dates.csv  # 1,924 questionable sources (input)
│   ├── qs_snapshots.csv            # Index of tmp/questionable_sources/ screenshots
│   ├── missing_qs.csv              # URLs missing from tmp/questionable_sources/ (for rerun)
│   ├── Mixed_output/               # 1,177 screenshots — MIXED factuality sources
│   ├── rerun_output/               # Screenshots from rerun of previously failed URLs
│   ├── rerun_qs/                   # Output dir for rerun of missing questionable sources
│   ├── error_output/               # Screenshots from retries with 60,000ms timeout
│   ├── snapshot_index.csv          # 866 successful screenshots — non-MIXED sources
│   ├── mixed-snapshots.csv         # 1,177 successful screenshots — MIXED sources
│   ├── errors.csv                  # 179 failed/problematic URLs — non-MIXED sources
│   └── mixed-errors.csv            # 116 failed/problematic URLs — MIXED sources
│
├── snapshots/                      # Scripts and final output CSVs
│   ├── snapshot.py                 # Core screenshot engine (single URL)
│   ├── batch_snapshot.py           # Batch runner — targets questionable sources by default
│   ├── postprocessing.py           # Post-processing: build mixed-snapshots.csv, merge CSVs
│   ├── strip_timestamps.py         # Strip _YYYYMMDD_HHMMSS from filenames and update CSVs
│   ├── sample_images.py            # Sample image utility
│   ├── testing.ipynb               # Testing notebook
│   └── data/
│       ├── snapshots.csv           # FINAL merged snapshot index (3,723 rows)
│       ├── error.csv               # FINAL merged error log (488 rows)
│       └── snapshot_sample_index.csv
│
├── cleaning.ipynb                  # Notebook for cleaning, enriching, and analysing snapshots
└── README.md
```

---

## Data Format

### MBFC source JSON (`raw_data/2291eng_dedup/`)

Each file is a JSON object for one media source:

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

**Factuality values** in the MBFC dataset:

| Value | Count |
|---|---:|
| MIXED | 1,293 |
| HIGH | 801 |
| LOW | 120 |
| VERY LOW | 61 |
| VERY HIGH | 15 |
| **Total** | **2,290** |

### Questionable sources CSV (`tmp/questionable_sources_with_dates.csv`)

1,924 sources flagged as questionable by MBFC, with columns:

| Column | Description |
|---|---|
| `media_source_name` | Human-readable source name (used as filename slug) |
| `media_source_url` | Homepage URL |
| `mbfc_report_url` | Link to the MBFC report page |
| `bias_rating` | Political bias label |
| `factual_reporting` | Factuality rating |
| `country` | Country of origin |
| `freedom_rating` | Press freedom rating |
| `media_type` | Type of media outlet |
| `credibility_rating` | MBFC credibility label |

---

## Output CSVs

### `snapshots/data/snapshots.csv` — 3,723 rows

All successfully captured screenshots across all source groups.

| Column | Description |
|---|---|
| `media_name` | Human-readable name of the media source |
| `url` | Homepage URL |
| `image_path` | Relative path to the PNG file |
| `timestamp` | Capture time (`YYYY-MM-DD HH:MM:SS`) |
| `country` | Country of origin |
| `factuality` | Factuality rating |
| `trustworthiness` | Binary: `1` = HIGH/VERY HIGH, `0` = LOW/VERY LOW, empty = MIXED/questionable |

**Breakdown by factuality (`snapshots/data/snapshots.csv`):**

| Factuality | Count |
|---|---:|
| MIXED | 2,449 |
| HIGH | 714 |
| LOW | 337 |
| VERY LOW | 193 |
| VERY HIGH | 11 |
| **Total** | **3,723** |

### `snapshots/data/error.csv` — 488 rows

All sources that could not be successfully captured, across all source groups.

| Column | Description |
|---|---|
| `filename` | PNG filename if a screenshot was taken (visual error), else empty |
| `name` | Media source name |
| `url` | Homepage URL |
| `issue` | Category of the problem (see table below) |
| `timestamp` | Capture time if available, else empty |
| `factuality` | Factuality rating of the source |

### `tmp/qs_snapshots.csv`

Index of screenshots in `tmp/questionable_sources/`. Same columns as `snapshots.csv`. Merged into `snapshots/snapshots.csv`.

---

### Source files (`tmp/`)

Per-class index files used by `postprocessing.py`:

| File | Rows | Covers |
|---|---:|---|
| `tmp/snapshot_index.csv` | 866 | non-MIXED successful captures |
| `tmp/mixed-snapshots.csv` | 1,177 | MIXED successful captures |
| `tmp/qs_snapshots.csv` | 1,676 | Questionable sources successful captures |
| `tmp/errors.csv` | 179 | non-MIXED failures |
| `tmp/mixed-errors.csv` | 116 | MIXED failures |

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
| Cloudflare Error 522 — Connection timed out | Origin server failed to respond |
| Cloudflare Error 1001/1016 — DNS error | Domain could not be resolved |
| Account suspended | Hosting account suspended |
| Blank — pure white | Page loaded but rendered nothing |
| Domain parked / for sale | Domain no longer active |
| Domain hijacked | Domain redirects to unrelated content |
| Site currently unavailable | Hosting provider error page |
| Maintenance / coming soon | Site is temporarily down |
| Page load timeout | Page did not load within timeout |
| DNS failure | Domain does not exist |
| SSL/TLS certificate error | Certificate invalid or expired |
| Connection reset / timed out | Network-level failure |
| Visual error — moved to errors/ | Screenshot captured but showed an error page |
| Not captured | No screenshot attempt recorded |

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

All scripts live in `snapshots/`. Run them from the **project root** so relative paths (e.g. `tmp/`, `output/`) resolve correctly.

### Single URL — `snapshots/snapshot.py`

```bash
python snapshots/snapshot.py https://www.bbc.com
python snapshots/snapshot.py https://example.com --output shots --full-page
python snapshots/snapshot.py https://example.com --width 1440 --height 900 --format jpeg
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

Screenshots are saved as `<MediaName>_<YYYYMMDD>_<HHMMSS>.png`.

---

### Batch Run — `snapshots/batch_snapshot.py`

Default source: `tmp/questionable_sources_with_dates.csv`. Default output: `tmp/questionable_sources/`.

```bash
# Capture all questionable sources → tmp/questionable_sources/
python snapshots/batch_snapshot.py

# Use a different sources CSV
python snapshots/batch_snapshot.py --sources-csv tmp/questionable_sources_with_dates.csv

# Custom output directory
python snapshots/batch_snapshot.py --output tmp/rerun_qs

# Re-capture even if screenshot already exists
python snapshots/batch_snapshot.py --force

# Retry only URLs that previously timed out
python snapshots/batch_snapshot.py --retry-timeouts --timeout 60000

# Retry all error-status URLs across log files
python snapshots/batch_snapshot.py \
  --from-logs tmp/questionable_sources/batch_log.jsonl \
  --output tmp/rerun_qs --timeout 60000

# Rerun only URLs listed in a CSV (must have a 'url' column)
python snapshots/batch_snapshot.py --from-csv tmp/missing_qs.csv --output tmp/rerun_qs

# Preview URLs without capturing
python snapshots/batch_snapshot.py --dry-run

# Tune parallelism (default: 4 workers)
python snapshots/batch_snapshot.py --workers 8
```

All runs append results to `<output_dir>/batch_log.jsonl`.

#### Rerunning missing sources

To capture sources whose images are missing from `tmp/questionable_sources/`:

```bash
# 1. Regenerate the missing-sources CSV
python - << 'EOF'
import csv, re, sys
from pathlib import Path
sys.path.insert(0, 'snapshots')
from snapshot import slugify_name, slugify_url

ts = re.compile(r"_\d{8}_\d{6}$")
captured = {ts.sub("", f.stem) for f in Path("tmp/questionable_sources").glob("*.png")}

missing = []
with open("tmp/questionable_sources_with_dates.csv") as fh:
    for row in csv.DictReader(fh):
        url = row.get("media_source_url", "").strip()
        name = row.get("media_source_name", "").strip() or url
        if url and (slugify_name(name) if name != url else slugify_url(url)) not in captured:
            missing.append({"url": url, "name": name})

with open("tmp/missing_qs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["url", "name"])
    w.writeheader(); w.writerows(missing)
print(f"{len(missing)} missing sources → tmp/missing_qs.csv")
EOF

# 2. Run the batch
python snapshots/batch_snapshot.py --from-csv tmp/missing_qs.csv --output tmp/rerun_qs
```

---

### Post-processing — `snapshots/postprocessing.py`

Builds `tmp/mixed-snapshots.csv` and produces the final deduplicated merged CSVs for MBFC sources.

```bash
python snapshots/postprocessing.py              # run all steps
python snapshots/postprocessing.py --step index         # build tmp/mixed-snapshots.csv only
python snapshots/postprocessing.py --step merge-snaps   # merge snapshot CSVs → snapshots.csv
python snapshots/postprocessing.py --step merge-errors  # merge error CSVs → error.csv
```

### Strip timestamps — `snapshots/strip_timestamps.py`

Removes `_YYYYMMDD_HHMMSS` from filenames in `tmp/questionable_sources/` and updates `tmp/qs_snapshots.csv` to match.

```bash
python snapshots/strip_timestamps.py
```

---

## Image Naming

Screenshots are saved with timestamps at capture time:

```
<MediaName>_<YYYYMMDD>_<HHMMSS>.png
```

For the questionable sources batch, timestamps are stripped from final filenames using `strip_timestamps.py`, leaving clean names like `BBC_News.png`. The capture timestamp is preserved in the `timestamp` column of the index CSV.

---

## Workflow

### Phase 1 — Non-MIXED sources (HIGH / LOW / VERY HIGH / VERY LOW)

```
1. python snapshots/batch_snapshot.py --sources-csv <non-mixed-csv> --output output/
        ↓ captures all qualifying sources → output/
        ↓ logs every attempt → output/batch_log.jsonl

2. Visual review
        ↓ detect blank/error/bot-blocked screenshots
        ↓ move bad ones → errors/
        ↓ record in tmp/errors.csv

3. python snapshots/batch_snapshot.py --from-csv tmp/errors.csv --output rerun_output
        ↓ retry failed URLs

4. python snapshots/batch_snapshot.py \
     --from-logs output/batch_log.jsonl rerun_output/batch_log.jsonl \
     --output error_output --timeout 60000
        ↓ retry persistent errors with higher timeout
```

### Phase 2 — MIXED sources

```
1. python snapshots/batch_snapshot.py --sources-csv <mixed-csv> --output Mixed_output/
        ↓ captures all 1,293 MIXED sources → Mixed_output/
        ↓ logs every attempt → Mixed_output/batch_log.jsonl

2. Visual review
        ↓ detect blank/error/bot-blocked screenshots
        ↓ move bad ones → errors/
        ↓ record in tmp/mixed-errors.csv

3. python snapshots/postprocessing.py
        ↓ builds tmp/mixed-snapshots.csv
        ↓ deduplicates and merges all source CSVs
        ↓ writes snapshots/data/snapshots.csv + snapshots/data/error.csv
```

### Phase 3 — Questionable sources

```
1. python snapshots/batch_snapshot.py
        ↓ reads tmp/questionable_sources_with_dates.csv (1,924 sources)
        ↓ captures screenshots → tmp/questionable_sources/
        ↓ logs every attempt → tmp/questionable_sources/batch_log.jsonl

2. Automated visual review (60 parallel agents)
        ↓ detect error pages, blank pages, parked domains, bot blocks
        ↓ move bad ones → errors/

3. python snapshots/strip_timestamps.py
        ↓ strips _YYYYMMDD_HHMMSS from filenames
        ↓ updates tmp/qs_snapshots.csv

4. Merge into global index
        ↓ tmp/qs_snapshots.csv → appended to snapshots/data/snapshots.csv
        ↓ missing/error entries → appended to snapshots/data/error.csv
```

---

## `cleaning.ipynb`

A Jupyter notebook for post-processing and analysing the captured screenshots.

**What it does:**

1. Loads snapshot index CSVs into pandas DataFrames
2. Strips timestamps from image filenames and updates paths
3. Parses and formats the `timestamp` column to `YYYY-MM-DD HH:MM:SS`
4. Adds a `trustworthiness` binary column (`1` = HIGH/VERY HIGH, `0` = LOW/VERY LOW)
5. Saves the enriched DataFrames back to their source CSVs
6. Plots factuality distributions and comparison charts across source groups

---

## Current Statistics

| Metric | Count |
|---|---:|
| **MBFC dataset** | |
| Total MBFC sources | 2,290 |
| — MIXED sources | 1,293 |
| — Non-MIXED (HIGH / LOW / VERY HIGH / VERY LOW) | 997 |
| **Questionable sources** | |
| Total questionable sources | 1,924 |
| — Successfully captured (`tmp/questionable_sources/`) | 1,676 |
| — Error / not captured (`snapshots/error.csv`) | 248 |
| **Non-MIXED captures** | |
| Successfully captured (`output/`) | 866 |
| — Trustworthy (HIGH / VERY HIGH) | 716 |
| — Untrustworthy (LOW / VERY LOW) | 150 |
| Failed / problematic | 131 |
| **MIXED captures** | |
| Successfully captured (`Mixed_output/`) | 1,177 |
| Failed / problematic | 116 |
| **errors/ directory** | |
| Total error screenshots | 231 |
| **Final merged outputs** | |
| Clean screenshots (`snapshots/data/snapshots.csv`) | **3,723** |
| Error records (`snapshots/data/error.csv`) | **488** |
