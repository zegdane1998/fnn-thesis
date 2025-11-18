import os
import time
import json
import warnings

import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from tqdm import tqdm
from typing import Optional

# Suppress XML-as-HTML warnings (some URLs return XML / RSS)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

FAKE_FILE = "FakeNewsNet/dataset/politifact_fake.csv"
REAL_FILE = "FakeNewsNet/dataset/politifact_real.csv"

OUTPUT_DIR = "data/raw/politifact_articles"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pick_url(row):
    """
    FakeNewsNet PolitiFact CSV usually has 'news_url'.
    Fallback to 'url' if needed.
    """
    for col in ("news_url", "url"):
        if col in row and isinstance(row[col], str) and row[col].startswith("http"):
            return row[col]
    return None


def get_article_text(url: str) -> Optional[str]:
    """Download and extract main text from a news article URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200 or not resp.text:
            return None
        content = resp.text

        # Basic HTML parsing (fine for most cases)
        soup = BeautifulSoup(content, "html.parser")

        # Prefer <article> if present
        article = soup.find("article")
        if article:
            text = article.get_text(separator=" ", strip=True)
        else:
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            text = " ".join(paragraphs)

        if not text:
            return None
        # Filter ultra-short pages (navigation, error pages, etc.)
        if len(text.split()) < 50:
            return None
        return text
    except Exception:
        return None


def process_file(filepath: str, label: int) -> int:
    df = pd.read_csv(filepath)
    saved = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(filepath)):
        url = pick_url(row)
        if not url:
            continue

        text = get_article_text(url)
        if not text:
            continue

        item = {
            "id": str(row.get("id", "")),
            "label": int(label),   # 0=fake, 1=real (your convention)
            "url": url,
            "title": row.get("title", ""),
            "text": text,
        }

        out_path = os.path.join(OUTPUT_DIR, f"{item['id']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False)

        saved += 1

        # polite delay after successful fetch
        time.sleep(0.8)

    print(f"Saved {saved} articles from {os.path.basename(filepath)}")
    return saved


def main():
    total = 0
    total += process_file(FAKE_FILE, label=0)
    total += process_file(REAL_FILE, label=1)
    print(f"Total saved: {total} -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
