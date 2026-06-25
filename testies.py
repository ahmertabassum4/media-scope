import os
import json
import glob
from collections import Counter
import matplotlib.pyplot as plt

SNAPSHOT_INDEX = "snapshot_index.csv"
JSON_DIR       = "data/2291eng_dedup"

BIAS_ORDER = ["LEFT", "LEFT-CENTER", "LEAST BIASED", "RIGHT-CENTER", "RIGHT"]
COLORS = {
    "LEFT":         "#2166ac",
    "LEFT-CENTER":  "#67a9cf",
    "LEAST BIASED": "#7fbf7b",
    "RIGHT-CENTER": "#ef8a62",
    "RIGHT":        "#b2182b",
}

import csv
captured = set()
with open(SNAPSHOT_INDEX, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        captured.add(row["media_name"].strip().upper())

bias_by_name = {}
for path in glob.glob(os.path.join(JSON_DIR, "*.json")):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    name = d.get("media name", "").strip().upper()
    bias = (d.get("bias") or "").strip().upper()
    if name:
        bias_by_name[name] = bias

counts = Counter()
missing = 0
for name in captured:
    bias = bias_by_name.get(name)
    if bias in BIAS_ORDER:
        counts[bias] += 1
    else:
        missing += 1

print(f"captured: {len(captured)}  matched: {sum(counts.values())}  missing/unlabeled: {missing}")
for label in BIAS_ORDER:
    print(f"  {label}: {counts[label]}")

values = [counts[label] for label in BIAS_ORDER]

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.bar(BIAS_ORDER, values, color=[COLORS[l] for l in BIAS_ORDER], width=0.6)

for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.01,
            f"{v:,}", ha="center", va="bottom", fontsize=14, fontweight="bold")

ax.set_title("Distribution of Bias Ratings", fontsize=20, fontweight="bold", pad=20)
ax.set_xlabel("Bias Rating", fontsize=15)
ax.set_ylabel("Number of Sources", fontsize=15)
ax.set_ylim(0, max(values) * 1.15)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("bias_distribution.png", dpi=150)
plt.show()