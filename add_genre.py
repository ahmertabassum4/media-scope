import csv

INDEX = "snapshot_sample_index.csv"

# Four-class genre target. Conspiracy, Pseudoscience and Imposter/Pink Slime
# each map to their own class; every other genre (local/general news, politics,
# advocacy, finance, tech, fact-checkers, pollsters, etc.) becomes OTHER.
def genre_class(genre):
    g = (genre or "").strip().lower()
    if g == "conspiracy":
        return "CONSPIRACY"
    if g == "pseudoscience":
        return "PSEUDOSCIENCE"
    if g in ("imposter/pink slime", "imposter", "pink slime"):
        return "IMPOSTER"
    return "OTHER"


def main():
    with open(INDEX, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)

    if "genre_class" not in fields:
        fields = fields + ["genre_class"]

    for row in rows:
        # Recompute every run so reclassifying (e.g. adding IMPOSTER) takes effect.
        row["genre_class"] = genre_class(row.get("genre", ""))

    with open(INDEX, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        counts[row["genre_class"]] = counts.get(row["genre_class"], 0) + 1
    print("genre_class counts:", dict(sorted(counts.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()