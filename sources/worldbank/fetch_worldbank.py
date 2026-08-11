"""
World Bank Indicators Fetcher
Downloads sovereign indicator series (emissions, GDP) as raw JSON.

Provides both halves of a sovereign carbon intensity: a territorial emissions
numerator and a GDP denominator, from one API with no authentication.

Source: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
Updated: irregularly; each response carries a "lastupdated" date, which the
         adapter records as the observation vintage.

Output:
  data/worldbank/<indicator>_YYYY-MM-DD.json

Usage:
  python sources/worldbank/fetch_worldbank.py
  python sources/worldbank/fetch_worldbank.py --indicator NY.GDP.MKTP.PP.CD
  python sources/worldbank/fetch_worldbank.py --start 2015 --end 2023

Raw responses are saved unmodified. Converting them into contract records is a
separate, pure step:

  from climate_attribution.data.adapters import parse_worldbank
  observations = parse_worldbank(payload, measure="gdp", unit="USD", basis="nominal")
"""

import argparse
import json
import os
import sys
from datetime import date

import requests

# ── Config ────────────────────────────────────────────────────────────────────

# Indicators worth having for a sovereign intensity. The emissions series was
# renamed when the World Bank moved to AR5 global warming potentials; the older
# EN.ATM.CO2E.KT covers CO2 only and is in kilotonnes.
INDICATORS = {
    "EN.GHG.CO2.MT.CE.AR5": "CO2 emissions (Mt CO2e, AR5)",
    "NY.GDP.MKTP.CD":       "GDP (current US$)",
    "NY.GDP.MKTP.PP.CD":    "GDP, PPP (current international $)",
}

DEFAULT_INDICATOR = "NY.GDP.MKTP.CD"
OUTPUT_DIR = os.path.join("data", "worldbank")
PER_PAGE = 20000

# ─────────────────────────────────────────────────────────────────────────────

BASE_URL = "https://api.worldbank.org/v2/country/all/indicator/{indicator}"


def fetch(indicator: str, start: int | None, end: int | None, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().isoformat()
    path = os.path.join(output_dir, f"{indicator}_{today}.json")

    if os.path.exists(path):
        print(f"  Already downloaded today: {path}")
        return path

    params = {"format": "json", "per_page": PER_PAGE}
    if start is not None and end is not None:
        params["date"] = f"{start}:{end}"

    print(f"  Downloading {indicator}...", end=" ", flush=True)
    response = requests.get(BASE_URL.format(indicator=indicator), params=params, timeout=120)

    if response.status_code != 200:
        print(f"FAILED (HTTP {response.status_code})")
        sys.exit(1)

    payload = response.json()

    # The API reports errors with HTTP 200 and a message object, so check the body.
    if not isinstance(payload, list) or len(payload) < 2:
        print("FAILED")
        print(f"  Unexpected response: {json.dumps(payload)[:400]}")
        sys.exit(1)

    header, rows = payload[0], payload[1] or []
    with open(path, "w") as f:
        json.dump(payload, f)

    print(f"done ({len(rows)} observations, updated {header.get('lastupdated')}) -> {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download World Bank indicator data.")
    parser.add_argument(
        "--indicator", default=DEFAULT_INDICATOR,
        help=f"indicator code (default: {DEFAULT_INDICATOR})",
    )
    parser.add_argument("--start", type=int, help="first year, e.g. 2015")
    parser.add_argument("--end", type=int, help="last year, e.g. 2023")
    parser.add_argument(
        "--all", action="store_true",
        help=f"fetch all {len(INDICATORS)} known indicators",
    )
    parser.add_argument("--list", action="store_true", help="list known indicators")
    args = parser.parse_args()

    if args.list:
        for code, label in INDICATORS.items():
            print(f"  {code:<24}{label}")
        return

    if (args.start is None) != (args.end is None):
        parser.error("--start and --end must be given together")

    wanted = list(INDICATORS) if args.all else [args.indicator]

    print("Fetching World Bank indicators...")
    for indicator in wanted:
        fetch(indicator, args.start, args.end, OUTPUT_DIR)

    print("\nDone. Files saved to the data/worldbank/ directory.")


if __name__ == "__main__":
    main()
