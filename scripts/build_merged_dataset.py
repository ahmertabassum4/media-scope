#!/usr/bin/env python3
"""Merge all feature sources into one experiment-ready dataset, joined by outlet.

Sources (joined on a normalized outlet key):
  - data/features/sonnet_factuality_features.jsonl  -> sonnet_features (s001-s100, categorical)
                                                       sonnet_derived  (12 numeric)
  - data/features/sonnet_bias_features.jsonl        -> bias_features   (d/i, categorical)
  - data/features/site_metadata_features.jsonl      -> metadata_features (m*, categorical)
                                                       metadata_derived  (41 numeric)
                                                       + labels (factuality, bias)
  - data/features/color_features.jsonl              -> color_features  (c_*, 16 numeric)

Labels: factuality + bias from the metadata feature file, with media_metadata as fallback.
One output row per outlet that has Sonnet features (the labeled 1,096). Missing sources are
left as empty dicts so downstream experiments can impute / ablate per group.

Output: data/features/merged_dataset.jsonl

Usage:
  /opt/anaconda3/bin/python3.12 scripts/build_merged_dataset.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
F_SONNET = ROOT / "data" / "features" / "sonnet_factuality_features.jsonl"
F_BIAS = ROOT / "data" / "features" / "sonnet_bias_features.jsonl"
F_META = ROOT / "data" / "features" / "site_metadata_features.jsonl"
F_COLOR = ROOT / "data" / "features" / "color_features.jsonl"
F_MEDIA = ROOT / "data" / "metadata" / "media_metadata"
OUT = ROOT / "data" / "features" / "merged_dataset.jsonl"


def norm_key(name: str) -> str:
    name = re.sub(r"\.(png|jpg|jpeg|metadata|json)$", "", name, flags=re.I)
    name = re.sub(r"(_\d+)+$", "", name)  # strip dedup (_1) & capture timestamp (_YYYYMMDD_HHMMSS)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_media_label_lookup() -> dict:
    """Fallback labels from media_metadata/*.json (robust to concatenated objects)."""
    lookup: dict[str, tuple[str, str]] = {}
    dec = json.JSONDecoder()
    for fp in F_MEDIA.glob("*.json"):
        try:
            obj, _ = dec.raw_decode(fp.read_text().lstrip())
        except Exception:  # noqa: BLE001
            continue
        fac = (obj.get("factuality") or "").strip().upper()
        bias = (obj.get("bias") or "").strip().upper()
        name = obj.get("media name") or obj.get("media_name") or fp.stem
        for k in {norm_key(fp.stem), norm_key(str(name))}:
            if k:
                lookup.setdefault(k, (fac, bias))
    return lookup


def main() -> int:
    sonnet = {norm_key(r["filename"]): r for r in load_jsonl(F_SONNET)}
    bias = {norm_key(r["filename"]): r for r in load_jsonl(F_BIAS)}
    color = {r.get("key") or norm_key(r["filename"]): r for r in load_jsonl(F_COLOR) if r.get("ok")}

    meta_rows = load_jsonl(F_META)
    meta = {norm_key(r["filename"]): r for r in meta_rows}
    meta_labels = {
        norm_key(r["filename"]): (
            (r.get("ground_truth") or "").strip().upper(),
            (r.get("bias") or "").strip().upper(),
        )
        for r in meta_rows
    }
    media_labels = build_media_label_lookup()

    out_rows = []
    n_fac = n_bias = n_meta = n_color = 0
    for key, srow in sonnet.items():
        sp = srow.get("parsed", {}) or {}

        fac, bia = meta_labels.get(key, ("", ""))
        if not fac or not bia:
            mfac, mbia = media_labels.get(key, ("", ""))
            fac = fac or mfac
            bia = bia or mbia

        mrow = meta.get(key, {})
        mp = mrow.get("parsed", {}) or {}
        crow = color.get(key, {})
        cp = crow.get("parsed", {}) or {}
        brow = bias.get(key, {})
        bp = brow.get("parsed", {}) or {}

        merged = {
            "key": key,
            "filename": srow.get("filename"),
            "factuality": fac,
            "bias": bia,
            "has_metadata": bool(mp),
            "has_color": bool(cp),
            "has_bias": bool(bp),
            "sonnet_features": sp.get("features", {}) or {},
            "sonnet_derived": sp.get("derived", {}) or {},
            "bias_features": bp.get("features", {}) or {},
            "metadata_features": mp.get("features", {}) or {},
            "metadata_derived": mp.get("derived", {}) or {},
            "color_features": cp.get("features", {}) or {},
        }
        if fac:
            n_fac += 1
        if bia:
            n_bias += 1
        if mp:
            n_meta += 1
        if cp:
            n_color += 1
        out_rows.append(merged)

    with open(OUT, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    print(f"Merged {len(out_rows)} outlets -> {OUT}")
    print(f"  with factuality label : {n_fac}")
    print(f"  with bias label       : {n_bias}")
    print(f"  with metadata features: {n_meta}")
    print(f"  with color features   : {n_color}")
    # feature-group widths from a fully-populated row
    for r in out_rows:
        if r["has_metadata"] and r["has_color"]:
            print("  group widths -> "
                  f"sonnet={len(r['sonnet_features'])}+{len(r['sonnet_derived'])}d, "
                  f"bias={len(r['bias_features'])}, "
                  f"metadata={len(r['metadata_features'])}+{len(r['metadata_derived'])}d, "
                  f"color={len(r['color_features'])}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
