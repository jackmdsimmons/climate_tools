"""
SBTi Dataset Fetcher
Downloads the Science Based Targets initiative (SBTi) company and target datasets.

Source: https://sciencebasedtargets.org/target-dashboard
Updated: every Thursday by SBTi

Output:
  data/sbti_companies_YYYY-MM-DD.xlsx   -- one row per company
  data/sbti_targets_YYYY-MM-DD.xlsx     -- one row per target (more granular)

Usage:
  python sources/sbti/fetch_sbti.py
"""

import os
import sys
from datetime import date
import requests

# ── Config ────────────────────────────────────────────────────────────────────

URLS = {
    "companies": "https://files.sciencebasedtargets.org/production/files/companies-excel.xlsx",
    "targets":   "https://sciencebasedtargets.org/resources/files/targets-excel.xlsx",
}

OUTPUT_DIR = "data"

# ─────────────────────────────────────────────────────────────────────────────


def fetch(label: str, url: str, output_dir: str) -> str:
    today = date.today().isoformat()
    filename = f"sbti_{label}_{today}.xlsx"
    path = os.path.join(output_dir, filename)

    if os.path.exists(path):
        print(f"  {filename} already exists, skipping download.")
        return path

    print(f"  Downloading {label} dataset...", end=" ", flush=True)
    response = requests.get(url, timeout=60)

    if response.status_code != 200:
        print(f"FAILED (HTTP {response.status_code})")
        sys.exit(1)

    with open(path, "wb") as f:
        f.write(response.content)

    size_kb = len(response.content) // 1024
    print(f"done ({size_kb} KB) -> {path}")
    return path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching SBTi datasets...")
    for label, url in URLS.items():
        fetch(label, url, OUTPUT_DIR)

    print("\nDone. Files saved to the data/ directory.")
    print("Note: SBTi updates these files every Thursday.")


if __name__ == "__main__":
    main()
