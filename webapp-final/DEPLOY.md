# Deploying MediaScope to Hugging Face Spaces

This app normally runs as three Docker Compose services. A Hugging Face Space runs **one
container on one port**, so for Spaces everything is merged into the root `Dockerfile`
(frontend + backend + models, supervised by `supervisor`, fronted by nginx on port
`7860`). The models service is GPU-accelerated on a **T4**.

## What it costs

| Item | Cost |
|---|---|
| HF account / deploying a Space | **Free** (no Pro required) |
| Free CPU hardware | $0 |
| **T4 small GPU** (recommended here) | **$0.40/hr**, billed only while awake |
| Game responses → private HF Dataset | **$0** |
| OpenRouter `gpt-5.5` (analysis only — the game never calls it) | pay-per-use, billed by OpenRouter |

Set the Space to **sleep after inactivity** so you only pay for the T4 while it's
actually used. A T4 is far more than enough — the local models use ~1.2 GB of its 16 GB.

## One-time setup

### 1. Create the Space
- huggingface.co → **New Space**
- **SDK: Docker** (blank/custom), **Space hardware: Nvidia T4 small**
- Under **Settings → Sleep time**, set it to sleep after idle (e.g. 15 min).

### 2. Add secrets (Settings → Variables and secrets)
Secrets:
- `OPENROUTER_API_KEY` — your OpenRouter key
- `HF_TOKEN` — a HF token with **Write** access (Settings → Access Tokens). Used to
  create/commit the game-responses Dataset.

Variables:
- `GAME_DATASET_REPO` — e.g. `your-username/mediascope-game-responses` (created
  automatically on first run, as a **private** dataset)
- `GAME_DATASET_PRIVATE` — `true`
- (optional) `BIAS_OPENROUTER_MODEL`, `FACTUALITY_OPENROUTER_MODEL`, `MAX_UPLOAD_MB`

> `MODELS_API_URL`, `DATA_DIR`, and `GAME_RESPONSES_PATH` are already baked into the
> Dockerfile for the single-container layout — don't override them.

### 3. Push the code
The Space needs the **contents of this `webapp-final/` folder** at the repo root (so the
root `Dockerfile` and `README.md` front matter are at the top of the Space repo):

```bash
# from inside webapp-final/
git init && git add . && git commit -m "MediaScope Space"
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push space main      # may need: git push space HEAD:main
```

The first build is slow (CUDA torch + Playwright Chromium + Paddle). Watch **Logs**.

## Where the game data lands
Each submitted answer is appended to a CSV and batch-committed to
`GAME_DATASET_REPO/game_responses.csv` (default: every 20 answers or 300s). Columns:
`timestamp, session_id, item_id, selected_bias, correct_bias, bias_correct,
selected_factuality, correct_factuality, factuality_correct, score`.

On restart the logger re-downloads the existing CSV first, so data accumulates across
sleeps/rebuilds instead of being overwritten.

## GPU notes
- `models/requirements.txt` installs **CUDA 12.4** torch wheels; the base image is
  `nvidia/cuda:12.4.1`. DINOv2, SigLIP, and MiniLM all move onto the GPU automatically
  (`torch.cuda.is_available()`).
- **PaddleOCR stays on CPU** on purpose — `paddlepaddle-gpu` is very picky about CUDA
  versions and isn't worth the build risk for tiny OCR models. To try GPU OCR later,
  swap `paddlepaddle` → a matching `paddlepaddle-gpu` build and pass a GPU device in
  `models/app/evidence.py`.
- If you ever want a **pure-CPU** Space (free tier), revert the torch lines in
  `models/requirements.txt` to the `+cpu` wheels and use any `python:3.11` base image.

## Local sanity check (optional, needs Docker)
```bash
docker build -t mediascope .
docker run --rm -p 7860:7860 --env-file .env mediascope
# open http://localhost:7860
```
(Without `--gpus all` it runs on CPU locally; that's fine for testing the wiring.)
