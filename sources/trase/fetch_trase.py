"""
Trase Dataset Fetcher
Downloads all datasets from trase.earth/open-data.

Trase provides free supply chain and deforestation data for 10 countries
and multiple commodities (soy, beef, palm oil, cocoa, etc.) from 1997-2025.

Source: https://trase.earth/open-data
No API key or registration required.

Output:
  data/trase/{dataset_slug}/{filename}.csv

Usage:
  python sources/trase/fetch_trase.py

Options:
  --dry-run     List all datasets and URLs without downloading
  --country     Filter by country slug, e.g. --country brazil
  --commodity   Filter by commodity slug, e.g. --commodity soy
"""

import argparse
import json
import os
import sys
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://trase.earth"
OUTPUT_DIR = os.path.join("data", "trase")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; climate-tools/1.0)"
}


def get_all_datasets() -> list[dict]:
    """Fetch the open-data listing page and return all dataset metadata."""
    url = f"{BASE_URL}/open-data"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        print(f"ERROR: HTTP {resp.status_code} fetching {url}")
        sys.exit(1)

    soup = BeautifulSoup(resp.text, "html.parser")
    tag = soup.find("script", {"id": "__NEXT_DATA__"})
    if not tag:
        print("ERROR: Could not find page data. Trase may have updated their site.")
        sys.exit(1)

    data = json.loads(tag.string)
    hits = data.get("props", {}).get("pageProps", {}).get("datasetHits", [])
    return [h["item"] for h in hits if "item" in h]


def get_dataset_files(slug: str) -> list[dict]:
    """Fetch a dataset detail page and return its file list."""
    url = f"{BASE_URL}/open-data/datasets/{slug}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    tag = soup.find("script", {"id": "__NEXT_DATA__"})
    if not tag:
        return []

    data = json.loads(tag.string)
    return data.get("props", {}).get("pageProps", {}).get("datasetMetadata", {}).get("files", [])


def download_file(url: str, dest_path: str) -> bool:
    """Download a file, skip if it already exists."""
    if os.path.exists(dest_path):
        return False

    resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
    if resp.status_code != 200:
        print(f"    FAILED (HTTP {resp.status_code})")
        return False

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_kb = os.path.getsize(dest_path) // 1024
    print(f"    saved ({size_kb} KB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download Trase datasets")
    parser.add_argument("--dry-run", action="store_true", help="List datasets without downloading")
    parser.add_argument("--country", help="Filter by country slug, e.g. brazil")
    parser.add_argument("--commodity", help="Filter by commodity slug, e.g. soy")
    args = parser.parse_args()

    print("Fetching dataset listing...")
    datasets = get_all_datasets()
    print(f"Found {len(datasets)} datasets.")

    # Apply filters
    if args.country:
        datasets = [d for d in datasets if d.get("country_slug") == args.country]
        print(f"Filtered to {len(datasets)} datasets for country: {args.country}")
    if args.commodity:
        datasets = [d for d in datasets if d.get("commodity_slug") == args.commodity]
        print(f"Filtered to {len(datasets)} datasets for commodity: {args.commodity}")

    print()

    downloaded = 0
    skipped = 0
    failed = 0

    for i, dataset in enumerate(datasets):
        slug = dataset.get("idSlug")
        title = dataset.get("title", slug)
        print(f"[{i+1}/{len(datasets)}] {title}")

        files = get_dataset_files(slug)
        csv_files = [f for f in files if f.get("file_extension") == "csv"]

        if not csv_files:
            print("  No CSV files found.")
            failed += 1
            time.sleep(0.3)
            continue

        for f in csv_files:
            url = f.get("url")
            filename = f.get("filename")
            size_kb = f.get("file_size_bytes", 0) // 1024

            if not url or not filename:
                continue

            dest = os.path.join(OUTPUT_DIR, slug, filename)
            print(f"  {filename} ({size_kb} KB)")

            if args.dry_run:
                print(f"    -> {url}")
                continue

            if download_file(url, dest):
                downloaded += 1
            else:
                print(f"    already exists, skipping.")
                skipped += 1

        time.sleep(0.3)

    if not args.dry_run:
        print(f"\nDone. Downloaded: {downloaded}, Skipped (existing): {skipped}, Failed: {failed}")
        print(f"Files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
