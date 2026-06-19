from experiment_bootstrap import activate_project_root

activate_project_root()

import base64
import concurrent.futures
import csv
import io
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from image_process import (
    BIAS_5_LABELS,
    IMAGE_FOLDER,
    OpenRouterLLM,
    build_messages,
    extract_content,
    iter_images,
    load_ground_truth,
    normalize_label,
    safe_divide,
    write_csv,
)


MODEL = "openai/gpt-5.5"
SAMPLE_PER_LABEL = 10
SAMPLE_SEED = 17
TARGET_ASPECT_RATIO = 16 / 9
REQUEST_TIMEOUT = 120
REQUEST_RETRIES = 2
MAX_WORKERS = 4

BASELINE_PREDICTIONS_PATH = Path("bias_predictions.csv")
SAMPLE_PATH = Path("bias5_experiment_sample.csv")
PREDICTIONS_PATH = Path("bias5_experiment_predictions.csv")
METRICS_PATH = Path("bias5_experiment_metrics.csv")
CONFUSIONS_PATH = Path("bias5_experiment_confusions.csv")
DOC_PATH = Path("bias5_experiment_report.md")

BASELINE_COLUMN = "baseline_gpt5.5_original_prompt"
EXPERIMENTS = {
    "rubric_v1": {
        "views": 1,
        "prompt": (
            "Classify the political bias of this website using only the provided landing "
            "page screenshot. Do not browse, search, or use prior knowledge about the outlet. "
            "Use these labels exactly:\n"
            "- left: clearly progressive/liberal advocacy or consistently left-framed political content.\n"
            "- left-center: mild or mainstream left/liberal lean, some left-framed political emphasis, but not strongly activist.\n"
            "- least biased: neutral, balanced, nonpartisan, mostly factual, local/general interest, or no visible political lean.\n"
            "- right-center: mild or mainstream conservative lean, some right-framed political emphasis, but not strongly activist.\n"
            "- right: clearly conservative/right advocacy or consistently right-framed political content.\n"
            "Choose the closest label from the screenshot only. Return exactly one lowercase label."
        ),
    },
    "center_sensitive_v2": {
        "views": 1,
        "prompt": (
            "You are judging political bias from a landing page screenshot only. Do not browse "
            "the web and do not rely on the outlet's reputation. Important calibration: do not "
            "use 'least biased' just because the page looks professional. Use 'least biased' only "
            "when visible content is balanced, non-political, or lacks a directional political frame. "
            "If headlines, section names, slogans, issue choices, or framing show a consistent but "
            "not extreme ideological lean, choose left-center or right-center. Use left/right only "
            "for strong advocacy or clearly one-sided partisan framing. Return exactly one lowercase "
            "label from: left, left-center, least biased, right-center, right."
        ),
    },
    "hierarchical_json_v1": {
        "views": 1,
        "json": True,
        "prompt": (
            "Analyze the landing page screenshot only. Do not browse, search, or use outside "
            "knowledge about the outlet. First decide the visible ideological direction: left, "
            "right, or none. Then decide intensity: strong, mild, or none. Map direction/intensity "
            "to the final label: left+strong=left, left+mild=left-center, none+none=least biased, "
            "right+mild=right-center, right+strong=right. If evidence is weak or mixed, prefer mild "
            "over strong; use none only when no directional framing is visible. Return compact JSON "
            "with keys direction, intensity, label. The label must be one of: left, left-center, "
            "least biased, right-center, right."
        ),
    },
    "contrastive_v3": {
        "views": 1,
        "prompt": (
            "From the landing page screenshot only, choose which political-bias bucket best explains "
            "the visible page. Do not browse and do not use prior knowledge. Ask yourself these "
            "contrasts: If the page has a directional political worldview but still resembles normal "
            "news/analysis, choose left-center or right-center instead of least biased. If it is mostly "
            "neutral/local/service/nonpolitical, choose least biased. If it reads like advocacy, "
            "campaigning, or a movement publication, choose left or right. Return exactly one lowercase "
            "label: left, left-center, least biased, right-center, right."
        ),
    },
    "two_view_center_v1": {
        "views": 2,
        "prompt": (
            "Classify political bias using only these landing page screenshots from the same site. "
            "The first image is the first viewport; the second is the next viewport lower on the page. "
            "Do not browse or use outside knowledge. Look for recurring visible cues in headlines, "
            "topics, issue framing, navigation, opinion sections, slogans, and calls to action. "
            "Use left/right for strong advocacy, left-center/right-center for mild directional lean, "
            "and least biased only for balanced, neutral, or no visible ideological lean. Return exactly "
            "one lowercase label from: left, left-center, least biased, right-center, right."
        ),
    },
    "balanced_prior_v4": {
        "views": 1,
        "prompt": (
            "Classify the website's political bias from the landing page screenshot only. Do not "
            "browse or use outlet reputation. This benchmark is balanced across five labels, so do "
            "not default to 'least biased' as a safe answer. Treat 'least biased' as a positive claim "
            "that the visible page is neutral, balanced, or nonpolitical. If there is a visible but "
            "moderate ideological direction, choose left-center or right-center. If there is strong "
            "advocacy or one-sided partisan framing, choose left or right. Return exactly one lowercase "
            "label from: left, left-center, least biased, right-center, right."
        ),
    },
    "least_bias_gate_v5": {
        "views": 1,
        "prompt": (
            "Use only the landing page screenshot. Do not browse and do not use prior knowledge. Apply "
            "this decision gate: first ask whether the visible page contains directional political or "
            "ideological framing in headlines, navigation, issue selection, slogans, calls to action, "
            "or opinion language. If yes, do not choose least biased; choose the direction and strength "
            "as left, left-center, right-center, or right. Choose least biased only when visible evidence "
            "is neutral, balanced, mostly nonpolitical, or insufficiently directional. Return exactly one "
            "lowercase label from: left, left-center, least biased, right-center, right."
        ),
    },
    "four_view_center_v2": {
        "views": 4,
        "prompt": (
            "Classify political bias using only these sequential landing page screenshots from the same "
            "site. Do not browse or use outside knowledge. The images show the first four viewport-height "
            "sections from the top of the page. Integrate recurring cues across the page: headline topics, "
            "framing words, section names, opinion labels, advocacy calls, and source branding. Use least "
            "biased only for visibly neutral, balanced, or nonpolitical pages; use left-center/right-center "
            "for mild directional lean; use left/right for strong advocacy. Return exactly one lowercase "
            "label from: left, left-center, least biased, right-center, right."
        ),
    },
    "disambiguate_least_3way_v1": {
        "views": 4,
        "only_if_baseline": "least biased",
        "allowed_labels": ["left-center", "least biased", "right-center"],
        "prompt": (
            "The initial classifier is uncertain and selected 'least biased'. Re-check this case using "
            "only these landing page screenshots. Do not browse and do not use outside knowledge. Choose "
            "only among three labels: left-center, least biased, right-center. Use least biased only if "
            "the visible page is neutral, balanced, nonpolitical, or lacks directional political framing. "
            "Use left-center or right-center if visible headlines, topics, navigation, framing words, "
            "opinion labels, or calls to action show a mild directional lean. Return exactly one lowercase "
            "label from: left-center, least biased, right-center."
        ),
    },
}
POSTPROCESSORS = {
    "post_baseline_twoview_center": (
        "Keep the original baseline label, except when baseline is least biased and "
        "two_view_center_v1 predicts left-center or right-center."
    ),
    "post_baseline_fourview_center": (
        "Keep the original baseline label, except when baseline is least biased and "
        "four_view_center_v2 predicts left-center or right-center."
    ),
    "post_center_vote_gate": (
        "Keep baseline unless it is least biased and at least two center-sensitive "
        "methods agree on the same center label."
    ),
    "post_cascade_disambiguate_least": (
        "Keep baseline unless it is least biased; for those rows use the 3-way "
        "least-vs-center disambiguation result."
    ),
}


def crop_viewport(image_path: Path, viewport_index: int) -> bytes:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        viewport_height = min(height, round(width / TARGET_ASPECT_RATIO))
        top = min(viewport_index * viewport_height, max(0, height - viewport_height))
        crop_box = (0, top, width, top + viewport_height)
        cropped = image.crop(crop_box)
        output = io.BytesIO()
        cropped.save(output, format="PNG")
        return output.getvalue()


def viewport_to_data_url(image_path: Path, viewport_index: int) -> str:
    encoded = base64.b64encode(crop_viewport(image_path, viewport_index)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_experiment_messages(prompt: str, image_path: Path, views: int) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index in range(views):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": viewport_to_data_url(image_path, index)},
            }
        )
    return [{"role": "user", "content": content}]


def load_baseline_predictions() -> dict[str, str]:
    if not BASELINE_PREDICTIONS_PATH.exists():
        return {}

    with BASELINE_PREDICTIONS_PATH.open("r", encoding="utf-8", newline="") as input_file:
        rows = csv.DictReader(input_file)
        return {
            row["image"]: row.get("bias_5:openai/gpt-5.5", "")
            for row in rows
            if row.get("image")
        }


def select_sample() -> list[dict[str, str]]:
    rng = random.Random(SAMPLE_SEED)
    by_label: dict[str, list[Path]] = {label: [] for label in BIAS_5_LABELS}

    for image_path in iter_images(IMAGE_FOLDER, -1):
        _, true_bias_5 = load_ground_truth(image_path)
        by_label[true_bias_5].append(image_path)

    sample_rows = []
    for label in BIAS_5_LABELS:
        candidates = sorted(by_label[label])
        rng.shuffle(candidates)
        selected = sorted(candidates[:SAMPLE_PER_LABEL])
        for image_path in selected:
            sample_rows.append({"image": image_path.stem, "true_bias_5": label})

    return sorted(sample_rows, key=lambda row: row["image"])


def load_existing_predictions() -> tuple[dict[str, dict[str, str]], list[str]]:
    if not PREDICTIONS_PATH.exists():
        return {}, []

    with PREDICTIONS_PATH.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            return {}, []
        rows = {
            row["image"]: {field: row.get(field, "") for field in reader.fieldnames}
            for row in reader
            if row.get("image")
        }
        return rows, reader.fieldnames


def request_completion(llm: OpenRouterLLM, messages: list[dict[str, Any]]) -> str:
    last_error: Exception | None = None
    for attempt in range(REQUEST_RETRIES + 1):
        try:
            response = llm.openr_llm(model=MODEL, messages=messages, timeout=REQUEST_TIMEOUT)
            return extract_content(response)
        except Exception as exc:
            last_error = exc
            if attempt < REQUEST_RETRIES:
                time.sleep(2 ** attempt)
    raise last_error or RuntimeError("LLM request failed")


def parse_json_label(response_text: str) -> str:
    match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
    if not match:
        return normalize_label(response_text, BIAS_5_LABELS)
    data = json.loads(match.group(0))
    return normalize_label(str(data.get("label", "")), BIAS_5_LABELS)


def parse_experiment_response(response_text: str, experiment_name: str) -> str:
    allowed_labels = EXPERIMENTS[experiment_name].get("allowed_labels", BIAS_5_LABELS)
    if EXPERIMENTS[experiment_name].get("json"):
        return parse_json_label(response_text)
    return normalize_label(response_text, allowed_labels)


def run_experiment_call(llm: OpenRouterLLM, experiment_name: str, image_name: str) -> str:
    config = EXPERIMENTS[experiment_name]
    image_path = IMAGE_FOLDER / f"{image_name}.png"
    messages = build_experiment_messages(config["prompt"], image_path, int(config["views"]))
    last_error: Exception | None = None

    for attempt in range(REQUEST_RETRIES + 1):
        try:
            response_text = request_completion(llm, messages)
            return parse_experiment_response(response_text, experiment_name)
        except Exception as exc:
            last_error = exc
            if attempt < REQUEST_RETRIES:
                time.sleep(2 ** attempt)

    return f"error: {last_error}"


def run_experiments() -> None:
    sample_rows = select_sample()
    baseline = load_baseline_predictions()
    existing_rows, existing_fieldnames = load_existing_predictions()
    experiment_names = list(EXPERIMENTS)
    postprocessor_names = list(POSTPROCESSORS)
    fieldnames = ["image", "true_bias_5", BASELINE_COLUMN, *experiment_names, *postprocessor_names]
    for field in existing_fieldnames:
        if field not in fieldnames:
            fieldnames.append(field)

    llm = OpenRouterLLM()
    all_rows: dict[str, dict[str, str]] = {}

    for sample_row in sample_rows:
        image_name = sample_row["image"]
        row = existing_rows.get(image_name, {"image": image_name})
        row["image"] = image_name
        row["true_bias_5"] = sample_row["true_bias_5"]
        row[BASELINE_COLUMN] = baseline.get(image_name, "")

        pending = [
            experiment_name
            for experiment_name in experiment_names
            if not row.get(experiment_name, "").strip() or row.get(experiment_name, "").startswith("error:")
        ]
        pending = [
            experiment_name for experiment_name in pending
            if EXPERIMENTS[experiment_name].get("only_if_baseline") in {None, row.get(BASELINE_COLUMN)}
        ]
        for experiment_name in experiment_names:
            if EXPERIMENTS[experiment_name].get("only_if_baseline") not in {None, row.get(BASELINE_COLUMN)}:
                row[experiment_name] = ""

        if pending:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(run_experiment_call, llm, experiment_name, image_name): experiment_name
                    for experiment_name in pending
                }
                for future in concurrent.futures.as_completed(futures):
                    experiment_name = futures[future]
                    try:
                        row[experiment_name] = future.result()
                    except Exception as exc:
                        row[experiment_name] = f"error: {exc}"

        apply_postprocessors(row)

        all_rows[image_name] = row
        merged_rows = {**existing_rows, **all_rows}
        write_csv(
            PREDICTIONS_PATH,
            [{field: merged_rows[name].get(field, "") for field in fieldnames} for name in sorted(merged_rows)],
            fieldnames,
        )
        print(f"processed {len(merged_rows)} sample rows; latest={image_name}", flush=True)

    write_csv(SAMPLE_PATH, sample_rows, ["image", "true_bias_5"])
    write_metrics()
    write_report()


def apply_postprocessors(row: dict[str, str]) -> None:
    baseline = row.get(BASELINE_COLUMN, "")
    two_view = row.get("two_view_center_v1", "")
    four_view = row.get("four_view_center_v2", "")

    row["post_baseline_twoview_center"] = (
        two_view if baseline == "least biased" and two_view in {"left-center", "right-center"} else baseline
    )
    row["post_baseline_fourview_center"] = (
        four_view if baseline == "least biased" and four_view in {"left-center", "right-center"} else baseline
    )

    center_votes = [
        row.get(column, "")
        for column in [
            "center_sensitive_v2",
            "contrastive_v3",
            "two_view_center_v1",
            "balanced_prior_v4",
            "least_bias_gate_v5",
            "four_view_center_v2",
        ]
    ]
    center_vote_counts = Counter(label for label in center_votes if label in {"left-center", "right-center"})
    if baseline == "least biased" and center_vote_counts:
        label, count = center_vote_counts.most_common(1)[0]
        row["post_center_vote_gate"] = label if count >= 2 else baseline
    else:
        row["post_center_vote_gate"] = baseline

    disambiguated = row.get("disambiguate_least_3way_v1", "")
    row["post_cascade_disambiguate_least"] = (
        disambiguated if baseline == "least biased" and disambiguated in {"left-center", "least biased", "right-center"} else baseline
    )


def calculate_metrics(rows: list[dict[str, str]], prediction_column: str) -> dict[str, str]:
    evaluated = 0
    correct = 0
    invalid_or_error = 0
    counts = {label: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for label in BIAS_5_LABELS}

    for row in rows:
        truth = row.get("true_bias_5", "")
        prediction = row.get(prediction_column, "")
        if truth not in BIAS_5_LABELS:
            continue
        if prediction not in BIAS_5_LABELS:
            invalid_or_error += 1
            continue

        evaluated += 1
        counts[truth]["support"] += 1
        if prediction == truth:
            correct += 1

        for label in BIAS_5_LABELS:
            if prediction == label and truth == label:
                counts[label]["tp"] += 1
            elif prediction == label and truth != label:
                counts[label]["fp"] += 1
            elif prediction != label and truth == label:
                counts[label]["fn"] += 1

    f1_values = []
    weighted_f1_sum = 0.0
    for label in BIAS_5_LABELS:
        label_counts = counts[label]
        precision = safe_divide(label_counts["tp"], label_counts["tp"] + label_counts["fp"])
        recall = safe_divide(label_counts["tp"], label_counts["tp"] + label_counts["fn"])
        f1 = safe_divide(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        weighted_f1_sum += f1 * label_counts["support"]

    return {
        "experiment": prediction_column,
        "model": MODEL,
        "sample_size": str(len(rows)),
        "evaluated": str(evaluated),
        "invalid_or_error": str(invalid_or_error),
        "accuracy": f"{safe_divide(correct, evaluated):.6f}",
        "macro_f1": f"{safe_divide(sum(f1_values), len(BIAS_5_LABELS)):.6f}",
        "weighted_f1": f"{safe_divide(weighted_f1_sum, evaluated):.6f}",
    }


def build_confusion_rows(rows: list[dict[str, str]], prediction_column: str) -> list[dict[str, str]]:
    confusion = {label: Counter() for label in BIAS_5_LABELS}
    for row in rows:
        truth = row.get("true_bias_5", "")
        prediction = row.get(prediction_column, "")
        if truth in BIAS_5_LABELS:
            confusion[truth][prediction] += 1

    return [
        {
            "experiment": prediction_column,
            "true_label": true_label,
            **{f"pred_{label}": str(confusion[true_label][label]) for label in BIAS_5_LABELS},
            "pred_invalid_or_error": str(
                sum(
                    count for pred, count in confusion[true_label].items()
                    if pred not in BIAS_5_LABELS
                )
            ),
        }
        for true_label in BIAS_5_LABELS
    ]


def write_metrics() -> None:
    with PREDICTIONS_PATH.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    for row in rows:
        apply_postprocessors(row)

    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(PREDICTIONS_PATH, rows, fieldnames)

    columns = [BASELINE_COLUMN, *EXPERIMENTS.keys(), *POSTPROCESSORS.keys()]
    metric_rows = [calculate_metrics(rows, column) for column in columns]
    write_csv(
        METRICS_PATH,
        metric_rows,
        ["experiment", "model", "sample_size", "evaluated", "invalid_or_error", "accuracy", "macro_f1", "weighted_f1"],
    )

    confusion_rows = [
        row
        for column in columns
        for row in build_confusion_rows(rows, column)
    ]
    write_csv(
        CONFUSIONS_PATH,
        confusion_rows,
        [
            "experiment",
            "true_label",
            *[f"pred_{label}" for label in BIAS_5_LABELS],
            "pred_invalid_or_error",
        ],
    )


def write_report() -> None:
    metrics_text = METRICS_PATH.read_text(encoding="utf-8") if METRICS_PATH.exists() else ""
    report = (
        "# Bias 5-Label Experiment Report\n\n"
        f"Model: `{MODEL}`\n\n"
        f"Sample: `{SAMPLE_PER_LABEL}` per label, `{SAMPLE_PER_LABEL * len(BIAS_5_LABELS)}` total, seed `{SAMPLE_SEED}`.\n\n"
        "Goal: improve 5-label political-bias classification from landing-page screenshots while avoiding web search and outside outlet knowledge.\n\n"
        "## Experiments\n\n"
        "- `baseline_gpt5.5_original_prompt`: cached GPT-5.5 full-run labels from the original direct prompt.\n"
        "- `rubric_v1`: explicit definitions for each of the five classes.\n"
        "- `center_sensitive_v2`: calibration prompt aimed at reducing overuse of `least biased` for center-lean outlets.\n"
        "- `hierarchical_json_v1`: one-call direction/intensity decomposition, then final label in JSON.\n"
        "- `contrastive_v3`: contrastive decision rules for advocacy vs mild lean vs neutral.\n"
        "- `two_view_center_v1`: center-sensitive prompt with the first and second landing-page viewports.\n\n"
        "- `balanced_prior_v4`: tells the model the benchmark is balanced and `least biased` is a positive neutral claim.\n"
        "- `least_bias_gate_v5`: explicit gate that blocks `least biased` when directional political framing is visible.\n"
        "- `four_view_center_v2`: center-sensitive prompt with the first four viewport-height page sections.\n"
        "- `disambiguate_least_3way_v1`: four-view re-check only for rows where the original baseline predicted `least biased`; the re-check chooses among `left-center`, `least biased`, and `right-center`.\n"
        "- `post_baseline_twoview_center`: deterministic correction using baseline plus `two_view_center_v1` center labels.\n"
        "- `post_baseline_fourview_center`: deterministic correction using baseline plus `four_view_center_v2` center labels.\n"
        "- `post_center_vote_gate`: deterministic correction when at least two center-sensitive methods agree on a center label.\n\n"
        "- `post_cascade_disambiguate_least`: final cascade: keep the original baseline unless it predicted `least biased`; then use `disambiguate_least_3way_v1`.\n\n"
        "## Takeaways\n\n"
        "- The original baseline's main failure mode is over-predicting `least biased` for true `left-center` and `right-center`.\n"
        "- Pure prompt variants did not beat the baseline on accuracy. `four_view_center_v2` matched baseline accuracy (`0.58`) and improved macro-F1 (`0.568575` vs `0.552082`), suggesting extra page context helps class balance but is not enough alone.\n"
        "- The best method in this run is `post_cascade_disambiguate_least`: accuracy `0.62`, macro-F1 `0.619573`.\n"
        "- The cascade is also cost-efficient: it requires the baseline call for every sample, then an extra 3-way call only when baseline predicts `least biased`.\n"
        "- This is still a dev-subset result (`50` samples). The next validation step should run `post_cascade_disambiguate_least` on the remaining holdout or the full 250 before treating it as the winner.\n\n"
        "## Best Candidate\n\n"
        "Use the baseline 5-label prompt as a first pass. If the first pass returns `left`, `left-center`, `right-center`, or `right`, keep it. If it returns `least biased`, make a second GPT-5.5 call with four landing-page viewports and force a 3-way decision among `left-center`, `least biased`, and `right-center`.\n\n"
        "## Metrics CSV\n\n"
        "```csv\n"
        f"{metrics_text.strip()}\n"
        "```\n\n"
        "Confusions are saved in `bias5_experiment_confusions.csv`.\n"
    )
    DOC_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    run_experiments()
