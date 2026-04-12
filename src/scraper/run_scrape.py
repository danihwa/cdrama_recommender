"""
Orchestrates the full scraping pipeline:
  1. Scrape all drama URLs from MDL search pages
  2. For each URL, scrape the drama detail page
  3. Append each result to dramas.jsonl (resumable)
  4. Log any failed URLs to failed_urls.txt

Usage:
    uv run src/scraper/run_scrape.py
"""

import json
import time
import os
from src.scraper.drama_urls import get_all_drama_urls
from src.scraper.drama_info import get_drama_info

OUTPUT_FILE = "data/raw/dramas.jsonl"
FAILED_FILE = "data/failed_urls.txt"
DELAY_SECONDS = 1.5  # polite delay between requests


def load_already_scraped(output_file: str) -> set[str]:
    """Read the output file and return all URLs already scraped.

    Used to skip completed entries on resume. Keyed on `mdl_url` — every
    scraped record must include this field or it won't be recognised as done.
    Returns a set for O(1) membership checks when filtering the URL list.
    """
    if not os.path.exists(output_file):
        return set()

    already_scraped = set()
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                drama = json.loads(line)
                if drama.get("mdl_url"):
                    already_scraped.add(drama["mdl_url"])
            except json.JSONDecodeError:
                pass  # skip malformed lines

    print(f"Found {len(already_scraped)} already scraped dramas in {output_file}")
    return already_scraped


def append_drama(output_file: str, drama: dict) -> None:
    """Append a single drama record to the NDJSON output file.

    Writes in append mode so existing records are never overwritten.
    `ensure_ascii=False` preserves CJK characters in titles and descriptions.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(drama, ensure_ascii=False) + "\n")


def log_failed(failed_file: str, url: str, reason: str) -> None:
    """Append a failed URL and its error reason to the failure log.

    Tab-separated so the file can be re-processed later as a simple table.
    Failures are isolated per URL — one bad page won't stop the whole run.
    """
    os.makedirs(os.path.dirname(failed_file), exist_ok=True)
    with open(failed_file, "a", encoding="utf-8") as f:
        f.write(f"{url}\t{reason}\n")


def run():
    """Run the full scraping pipeline end-to-end.

    Collects all drama URLs from MDL search pages, skips any already present
    in the output file, then scrapes each remaining URL and saves it immediately.
    This makes the run resumable — safe to restart after a crash or interrupt.
    """
    print("Step 1: Collecting drama URLs from search pages...")
    all_urls = get_all_drama_urls(max_pages=136)
    print(f"Found {len(all_urls)} drama URLs total\n")

    already_scraped = load_already_scraped(OUTPUT_FILE)
    urls_to_scrape = [url for url in all_urls if url not in already_scraped]
    print(f"Skipping {len(already_scraped)} already scraped")
    print(f"Scraping {len(urls_to_scrape)} remaining dramas\n")

    success_count = 0
    fail_count = 0

    for i, url in enumerate(urls_to_scrape, start=1):
        print(f"[{i}/{len(urls_to_scrape)}] Scraping: {url}", end=" ... ")

        try:
            drama = get_drama_info(url)
            append_drama(OUTPUT_FILE, drama)
            success_count += 1
            print(f"{drama.get('title', '???')}")

        except Exception as e:
            fail_count += 1
            reason = str(e)
            log_failed(FAILED_FILE, url, reason)
            print(f"FAILED — {reason}")

        time.sleep(DELAY_SECONDS)

    print(f"\n{'='*50}")
    print(f"Scraped successfully : {success_count}")
    print(f"Failed (logged)      : {fail_count}")
    print(f"Output file         : {OUTPUT_FILE}")
    if fail_count > 0:
        print(f"Failed URLs logged  : {FAILED_FILE}")
    print(f"{'='*50}")


if __name__ == "__main__":
    run()
