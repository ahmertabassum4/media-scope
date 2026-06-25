"""Persist game answers to a CSV and batch-commit it to a private HF Dataset repo.

On a Hugging Face Space the container filesystem is ephemeral (it resets on
restart/rebuild/wake-from-sleep), so a plain local CSV would lose data. This logger
keeps a local CSV for durability *within* a run and periodically pushes the whole file
to a Dataset repo via ``huggingface_hub``. On startup it seeds the local CSV from the
repo so a restart appends to the existing data instead of overwriting it.

If no repo / token is configured the logger still writes the local CSV and simply skips
the push, so the game works in local dev without any HF credentials.
"""

from __future__ import annotations

import csv
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Column order written to the CSV. Stable so appends from different runs line up.
FIELDNAMES = [
    "timestamp",
    "session_id",
    "item_id",
    "selected_bias",
    "correct_bias",
    "bias_correct",
    "selected_factuality",
    "correct_factuality",
    "factuality_correct",
    "score",
]

REPO_FILENAME = "game_responses.csv"


class ResponseLogger:
    def __init__(
        self,
        csv_path: str | Path,
        repo_id: str | None = None,
        token: str | None = None,
        private: bool = True,
        flush_every: int = 20,
        flush_interval_seconds: float = 300.0,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.repo_id = repo_id or None
        self.token = token or None
        self.private = private
        self.flush_every = max(1, int(flush_every))
        self.flush_interval_seconds = max(10.0, float(flush_interval_seconds))

        self._lock = threading.Lock()
        self._pending = 0
        self._last_push = time.monotonic()
        self._api = None
        self._stop = threading.Event()
        self._timer: threading.Thread | None = None

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if self.repo_id and self.token:
            try:
                from huggingface_hub import HfApi

                self._api = HfApi(token=self.token)
                self._ensure_repo()
                self._seed_from_repo()
            except Exception as exc:  # noqa: BLE001 - never let logging break the game
                print(f"[responses] HF Dataset disabled ({type(exc).__name__}): {exc}")
                self._api = None

        self._ensure_header()

        if self._api is not None:
            self._timer = threading.Thread(target=self._periodic_flush, daemon=True)
            self._timer.start()

    # --- setup helpers -------------------------------------------------------

    def _ensure_header(self) -> None:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=FIELDNAMES).writeheader()

    def _ensure_repo(self) -> None:
        assert self._api is not None
        self._api.create_repo(
            repo_id=self.repo_id,
            repo_type="dataset",
            private=self.private,
            exist_ok=True,
        )

    def _seed_from_repo(self) -> None:
        """Pull the existing CSV from the repo so we append instead of overwrite."""
        assert self._api is not None
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return  # local file already present (e.g. mounted volume); keep it
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=REPO_FILENAME,
                token=self.token,
            )
            Path(downloaded).replace(self.csv_path)
            print(f"[responses] seeded local CSV from {self.repo_id}/{REPO_FILENAME}")
        except Exception:  # noqa: BLE001 - repo is empty / file not there yet
            pass

    # --- public API ----------------------------------------------------------

    def log(self, grade: dict[str, Any], session_id: str | None) -> None:
        """Append one graded answer and push to the repo if the batch is full."""
        bias = grade.get("bias", {})
        fact = grade.get("factuality", {})
        score = int(bool(bias.get("is_correct"))) + int(bool(fact.get("is_correct")))
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id or uuid.uuid4().hex,
            "item_id": grade.get("id", ""),
            "selected_bias": bias.get("selected", ""),
            "correct_bias": bias.get("correct_label", ""),
            "bias_correct": bool(bias.get("is_correct")),
            "selected_factuality": fact.get("selected", ""),
            "correct_factuality": fact.get("correct_label", ""),
            "factuality_correct": bool(fact.get("is_correct")),
            "score": score,
        }

        with self._lock:
            with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=FIELDNAMES).writerow(row)
            self._pending += 1
            should_push = self._api is not None and self._pending >= self.flush_every

        if should_push:
            self._push()

    def close(self) -> None:
        """Final flush on shutdown so the last partial batch isn't lost."""
        self._stop.set()
        self._push()

    # --- internals -----------------------------------------------------------

    def _periodic_flush(self) -> None:
        while not self._stop.wait(self.flush_interval_seconds):
            if self._pending > 0:
                self._push()

    def _push(self) -> None:
        if self._api is None:
            return
        with self._lock:
            if self._pending == 0:
                return
            pending = self._pending
        try:
            self._api.upload_file(
                path_or_fileobj=str(self.csv_path),
                path_in_repo=REPO_FILENAME,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"game responses: +{pending} answer(s)",
            )
            with self._lock:
                self._pending = max(0, self._pending - pending)
                self._last_push = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - keep the buffered rows for next try
            print(f"[responses] push failed ({type(exc).__name__}): {exc}")
