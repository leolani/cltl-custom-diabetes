"""
Extract emotion counts per author from capsules_with_event_details.json.zip.
Produces a CSV with emotions as rows and utterance authors as columns.
"""

import zipfile
import json
from collections import defaultdict

import pandas as pd

ZIP_PATH = "../../../data/capsules_with_event_details.json.zip"
JSON_NAME = "capsules_with_event_details.json"
OUTPUT_CSV = "data/emotion_by_author_counts.csv"


def load_data(zip_path: str, json_name: str) -> list:
    with zipfile.ZipFile(zip_path) as z:
        with z.open(json_name) as f:
            return json.load(f)


def build_emotion_author_counts(data: list) -> pd.DataFrame:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for item in data:
        turns = item[1]
        for turn in turns:
            author = turn.get("author", {}).get("label", "unknown")
            emotion_field = turn.get("perspective", {}).get("emotion", [])
            if isinstance(emotion_field, str):
                emotion_field = [emotion_field]
            for emotion in emotion_field:
                if emotion:
                    counts[emotion][author] += 1

    df = pd.DataFrame(counts).T.fillna(0).astype(int)
    df.index.name = "emotion"
    df = df.sort_index()
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def main():
    data = load_data(ZIP_PATH, JSON_NAME)
    df = build_emotion_author_counts(data)
    df.to_csv(OUTPUT_CSV)
    print(f"Saved {OUTPUT_CSV}  ({df.shape[0]} emotions x {df.shape[1]} authors)")
    print(df.to_string())


if __name__ == "__main__":
    main()