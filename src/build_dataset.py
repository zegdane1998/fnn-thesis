import os
import glob
import json
import re

import pandas as pd
from tqdm import tqdm

INPUT_DIR = "data/raw/politifact_articles"
OUT_PATH = "data/processed/politifact_articles.parquet"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def basic_clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    rows = []

    for fp in tqdm(files, desc="Merging JSONs"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                item = json.load(f)
        except Exception:
            continue

        text = basic_clean(item.get("text", "") or "")
        title = basic_clean(item.get("title", "") or "")
        if len(text.split()) < 50:
            continue

        rows.append(
            {
                "news_id": str(item.get("id", "")),
                "label": int(item.get("label", 0)),
                "title": title,
                "text": text,
                "source": "politifact",
            }
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print("Rows:", len(df))
    df.to_parquet(OUT_PATH, index=False)
    print("Saved ->", OUT_PATH)


if __name__ == "__main__":
    main()
