import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CYRILLIC = re.compile(r"[\u0400-\u04FF]")

REQUIRED_FILES = [
    "llm_client.py",
    "image_process.py",
    "bias5_pca_component_fusion_experiments.py",
    "bias5_top_ablation_experiments.py",
    "factuality_experiments.py",
    "factuality_prepare_features.py",
    "synthetic_factuality_sites.py",
    "synthetic_factuality_eval.py",
    "requirements.txt",
    "reports/experiment_methods_summary.md",
]


def validate_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")


def validate_python_syntax() -> None:
    for path in sorted(ROOT.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")


def validate_no_cyrillic() -> None:
    checked_paths = list(ROOT.glob("*.py")) + list((ROOT / "reports").glob("*.md")) + [ROOT / "requirements.txt"]
    offenders = []
    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        if CYRILLIC.search(text):
            offenders.append(str(path.relative_to(ROOT)))
    if offenders:
        raise SystemExit(f"Cyrillic text found in: {', '.join(offenders)}")


def main() -> None:
    validate_required_files()
    validate_python_syntax()
    validate_no_cyrillic()
    print("experiment package validation passed")


if __name__ == "__main__":
    main()
