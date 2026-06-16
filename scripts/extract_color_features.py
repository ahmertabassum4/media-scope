#!/usr/bin/env python3
"""Extract deterministic color / contrast / layout features from outlet screenshots.

No LLM, no internet. Pure PIL + numpy over the landing-page screenshots. Output is a
JSONL keyed by the same filename convention as the Sonnet feature files, so it merges
straight into the experiment table.

Features (all numeric, scale-invariant):
  c_brightness_mean      mean luminance (0..1)             clean/pro pages run bright
  c_brightness_std       luminance spread
  c_rms_contrast         RMS contrast (== brightness_std) garish high-contrast pages
  c_whitespace_ratio     fraction of near-white pixels     clean professional layout
  c_dark_ratio           fraction of near-black pixels      dark/heavy themes
  c_saturation_mean      mean HSV saturation               fringe blogs over-saturate
  c_saturation_std       saturation spread
  c_colorfulness         Hasler-Susstrunk colorfulness
  c_palette_size         distinct 4-bit-quantized colors / 4096   clutter / clash
  c_hue_entropy          Shannon entropy of hue histogram (0..1)  color discipline
  c_warm_ratio           red+orange+yellow pixels          "alarmist colors" proxy
  c_red_ratio            red/crimson pixels
  c_blue_ratio           blue pixels                       (weak) directional cue
  c_green_ratio          green pixels
  c_edge_density         fraction of strong-gradient pixels  layout clutter
  c_edge_mean            mean gradient magnitude

Usage:
  /opt/anaconda3/bin/python3.12 scripts/extract_color_features.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # full-page screenshots can be very tall

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_DIRS = [
    Path("/Users/delyan.hristov/Downloads/output"),
    ROOT / "data" / "screenshots" / "factuality",
    ROOT / "data" / "screenshots" / "mixed_output",
]
OUT_PATH = ROOT / "data" / "features" / "color_features.jsonl"
MAX_SIDE = 1024  # downscale longest side; color stats are scale-invariant


def norm_key(name: str) -> str:
    """Normalize a filename/outlet name to a join key shared across feature sources."""
    name = re.sub(r"\.(png|jpg|jpeg|metadata)$", "", name, flags=re.I)
    name = re.sub(r"(_\d+)+$", "", name)  # strip dedup (_1) & capture timestamp (_YYYYMMDD_HHMMSS)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _rgb_to_hsv_arrays(arr: np.ndarray):
    """arr: HxWx3 float in [0,1]. Returns (hue[0,360), sat[0,1], val[0,1])."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    diff = mx - mn
    val = mx
    sat = np.where(mx > 1e-6, diff / (mx + 1e-12), 0.0)

    hue = np.zeros_like(mx)
    safe = diff > 1e-6
    # red is max
    idx = safe & (mx == r)
    hue[idx] = (60 * ((g[idx] - b[idx]) / diff[idx]) + 360) % 360
    # green is max
    idx = safe & (mx == g)
    hue[idx] = 60 * ((b[idx] - r[idx]) / diff[idx]) + 120
    # blue is max
    idx = safe & (mx == b)
    hue[idx] = 60 * ((r[idx] - g[idx]) / diff[idx]) + 240
    return hue, sat, val


def compute_features(path: str) -> dict:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # luminance / contrast
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    brightness_mean = float(lum.mean())
    brightness_std = float(lum.std())
    whitespace_ratio = float((lum > 0.92).mean())
    dark_ratio = float((lum < 0.20).mean())

    # saturation
    hue, sat, val = _rgb_to_hsv_arrays(arr)
    saturation_mean = float(sat.mean())
    saturation_std = float(sat.std())

    # Hasler-Susstrunk colorfulness
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )

    # palette size: distinct colors after 4-bit/channel quantization
    q = (arr * 15).astype(np.uint16)
    codes = (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]
    palette_size = float(np.unique(codes).size / 4096.0)

    # hue entropy over colorful pixels
    colorful = sat > 0.15
    if colorful.sum() > 32:
        hist, _ = np.histogram(hue[colorful], bins=36, range=(0, 360))
        p = hist / hist.sum()
        p = p[p > 0]
        hue_entropy = float(-(p * np.log(p)).sum() / np.log(36))
    else:
        hue_entropy = 0.0

    # hue-band ratios (colorful, mid-tone pixels)
    band = colorful & (lum > 0.10) & (lum < 0.95)
    total = max(1, lum.size)
    h_band = hue[band]
    red_ratio = float(((h_band < 20) | (h_band >= 340)).sum() / total)
    warm_ratio = float(((h_band < 60) | (h_band >= 340)).sum() / total)
    green_ratio = float(((h_band >= 60) & (h_band < 170)).sum() / total)
    blue_ratio = float(((h_band >= 170) & (h_band < 270)).sum() / total)

    # edge density via luminance gradient
    gy, gx = np.gradient(lum)
    mag = np.sqrt(gx * gx + gy * gy)
    edge_density = float((mag > 0.10).mean())
    edge_mean = float(mag.mean())

    return {
        "c_brightness_mean": round(brightness_mean, 5),
        "c_brightness_std": round(brightness_std, 5),
        "c_rms_contrast": round(brightness_std, 5),
        "c_whitespace_ratio": round(whitespace_ratio, 5),
        "c_dark_ratio": round(dark_ratio, 5),
        "c_saturation_mean": round(saturation_mean, 5),
        "c_saturation_std": round(saturation_std, 5),
        "c_colorfulness": round(colorfulness, 5),
        "c_palette_size": round(palette_size, 5),
        "c_hue_entropy": round(hue_entropy, 5),
        "c_warm_ratio": round(warm_ratio, 5),
        "c_red_ratio": round(red_ratio, 5),
        "c_blue_ratio": round(blue_ratio, 5),
        "c_green_ratio": round(green_ratio, 5),
        "c_edge_density": round(edge_density, 5),
        "c_edge_mean": round(edge_mean, 5),
    }


def _worker(args):
    key, path, basename = args
    try:
        feats = compute_features(path)
        return {"filename": basename, "key": key, "ok": True, "parsed": {"features": feats}}
    except Exception as exc:  # noqa: BLE001
        return {"filename": basename, "key": key, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def collect_images() -> dict:
    """Map normalized key -> (path, basename). First occurrence wins."""
    found: dict[str, tuple[str, str]] = {}
    for d in SCREENSHOT_DIRS:
        if not d.is_dir():
            continue
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    key = norm_key(fn)
                    found.setdefault(key, (os.path.join(root, fn), fn))
    return found


def main() -> int:
    images = collect_images()
    print(f"Found {len(images)} unique screenshots.", flush=True)
    tasks = [(k, p, b) for k, (p, b) in images.items()]

    rows = []
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futs = [pool.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            if row["ok"]:
                ok += 1
            else:
                fail += 1
            if i % 200 == 0:
                print(f"  {i}/{len(tasks)}  ok={ok} fail={fail}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows ({ok} ok, {fail} failed) -> {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
