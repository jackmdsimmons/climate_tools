"""
Global Coal Exit List (GCEL) Fetcher
Logs in to coalexit.org and downloads the full GCEL company dataset.

The GCEL covers ~3,000 companies across the entire thermal coal value chain
including miners, power producers, developers, traders, and service companies.
Updated annually in Q4.

Source: https://www.coalexit.org
Free registration required: https://www.coalexit.org/user/register/gcel

Setup:
  1. Register for a free account at https://www.coalexit.org/user/register/gcel
  2. Add your credentials below (USERNAME and PASSWORD)
  3. Run: python sources/gcel/fetch_gcel.py

Output:
  data/gcel/gcel_YYYY-MM-DD.xlsx
"""

import asyncio
import os
import sys
from datetime import date
from playwright.async_api import async_playwright

# ── Config ────────────────────────────────────────────────────────────────────

USERNAME = "your_username_or_email_here"
PASSWORD = "your_password_here"

# Set to True to run without a visible browser window
HEADLESS = False

OUTPUT_DIR = os.path.join("data", "gcel")

# ─────────────────────────────────────────────────────────────────────────────

LOGIN_URL = "https://www.coalexit.org/user/login"
DOWNLOADS_URL = "https://www.coalexit.org/downloads"


async def fetch_gcel():
    if USERNAME == "your_username_or_email_here":
        print("ERROR: Set your USERNAME and PASSWORD in the script before running.")
        print("Register free at https://www.coalexit.org/user/register/gcel")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = date.today().isoformat()
    output_path = os.path.join(OUTPUT_DIR, f"gcel_{today}.xlsx")

    if os.path.exists(output_path):
        print(f"Already downloaded today: {output_path}")
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        # Log in
        print("Logging in to coalexit.org...")
        await page.goto(LOGIN_URL)

        # Drupal standard field names
        await page.fill("input[name='name']", USERNAME)
        await page.fill("input[name='pass']", PASSWORD)
        await page.click("input[type='submit']")
        await page.wait_for_load_state("networkidle")

        if "login" in page.url or "user/login" in page.url:
            print("ERROR: Login failed. Check your USERNAME and PASSWORD.")
            await browser.close()
            sys.exit(1)

        print("Logged in. Navigating to downloads...")
        await page.goto(DOWNLOADS_URL)
        await page.wait_for_load_state("networkidle")

        # Find the GCEL Excel download link
        excel_link = await page.query_selector("a[href*='gcel'][href*='.xlsx'], a[href*='GCEL'][href*='.xlsx']")

        if not excel_link:
            # Fallback: find any xlsx link on the page
            excel_link = await page.query_selector("a[href$='.xlsx']")

        if not excel_link:
            print("ERROR: Could not find GCEL Excel download link.")
            print("The page structure may have changed — please check https://www.coalexit.org/downloads")
            await browser.close()
            sys.exit(1)

        href = await excel_link.get_attribute("href")
        print(f"Found download link: {href}")

        # Trigger download
        print("Downloading GCEL dataset...")
        async with page.expect_download() as download_info:
            await excel_link.click()
        download = await download_info.value

        await download.save_as(output_path)
        size_kb = os.path.getsize(output_path) // 1024
        print(f"Saved to {output_path} ({size_kb} KB)")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(fetch_gcel())
