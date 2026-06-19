import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BIAS_LABELS = {"left", "left-center", "least biased", "right-center", "right"}
FACTUALITY_LABELS = {"very low", "low", "high", "very high"}


@dataclass(frozen=True)
class GameItem:
    id: str
    image: str
    bias: str
    factuality: str


class GameDataset:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.images_dir = data_dir / "images"
        self.items = self._load_items()
        self.by_id = {item.id: item for item in self.items}

    def _load_items(self) -> list[GameItem]:
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            return []

        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        items: list[GameItem] = []
        for row in rows:
            item = GameItem(
                id=str(row["id"]),
                image=str(row["image"]),
                bias=str(row["bias"]),
                factuality=str(row["factuality"]),
            )
            if item.bias not in BIAS_LABELS:
                raise ValueError(f"Unexpected bias label for {item.id}: {item.bias}")
            if item.factuality not in FACTUALITY_LABELS:
                raise ValueError(f"Unexpected factuality label for {item.id}: {item.factuality}")
            if not (self.images_dir / item.image).exists():
                raise ValueError(f"Missing game image for {item.id}: {item.image}")
            items.append(item)
        return items

    @property
    def count(self) -> int:
        return len(self.items)

    def random_item(self) -> GameItem:
        if not self.items:
            raise LookupError("Game dataset is empty")
        return random.choice(self.items)

    def get(self, item_id: str) -> GameItem:
        item = self.by_id.get(item_id)
        if item is None:
            raise LookupError("Game item not found")
        return item

    def image_path(self, item_id: str) -> Path:
        item = self.get(item_id)
        path = (self.images_dir / item.image).resolve()
        images_root = self.images_dir.resolve()
        if images_root not in path.parents:
            raise LookupError("Invalid game image path")
        return path

    def grade(self, item_id: str, bias: str, factuality: str) -> dict[str, Any]:
        item = self.get(item_id)
        normalized_bias = bias.strip().lower()
        normalized_factuality = factuality.strip().lower()
        return {
            "id": item.id,
            "bias": {
                "selected": normalized_bias,
                "correct_label": item.bias,
                "is_correct": normalized_bias == item.bias,
            },
            "factuality": {
                "selected": normalized_factuality,
                "correct_label": item.factuality,
                "is_correct": normalized_factuality == item.factuality,
            },
        }
