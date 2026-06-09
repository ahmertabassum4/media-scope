import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "metadata" / "media_metadata"
OUTPUT_DIR = PROJECT_ROOT / "data" / "factuality_splits"
TARGETS = {
    "HIGH": OUTPUT_DIR / "high_factuality",
    "LOW": OUTPUT_DIR / "low_factuality",
}


def load_json(path):
    content = path.read_text(encoding="utf-8").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(content)
        return obj


def main():
    for folder in TARGETS.values():
        folder.mkdir(parents=True, exist_ok=True)

    counts = {label: 0 for label in TARGETS}
    skipped = 0
    errors = []

    for path in sorted(INPUT_DIR.glob("*.json")):
        try:
            data = load_json(path)
        except Exception as exc:
            errors.append((path.name, str(exc)))
            continue

        factuality = data.get("factuality", "").strip().upper()
        if factuality not in TARGETS:
            skipped += 1
            continue

        shutil.copy2(path, TARGETS[factuality] / path.name)
        counts[factuality] += 1

    print("Done")
    for label, folder in TARGETS.items():
        print(f"{label}: {counts[label]} files -> {folder}")
    print(f"Skipped: {skipped}")

    if errors:
        print(f"Parse errors: {len(errors)}")
        for name, err in errors:
            print(f"{name}: {err}")


if __name__ == "__main__":
    main()
