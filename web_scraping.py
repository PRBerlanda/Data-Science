# -*- coding: utf-8 -*-
"""
WEB SCRAPING (COLLECTING DATA FROM WEB PAGES): WORKING TOOL
MBA in Data Science and Analytics (USP/Esalq). Working version generated with
Claude from the "scraping_colab" notebook (Data Collection and Web Scraping).

GOAL
- Turn HTML pages into tables (CSV/Excel) without rewriting code:
  * "tables" MODE: captures every <table> on the page(s) (pd.read_html).
  * "selectors" MODE: for "card" layouts (products, news, cities), provide
    the CSS selector of each item and of each field.
- Walks through several pages (list of URLs, a template with {page} or a
  "next" link).

BUILT-IN GOOD PRACTICES
- Respects robots.txt, waits between requests, identifies itself with a
  User-Agent, retries on temporary failures and converts numbers stored as text.
- Use only on sites whose terms allow scraping, without sensitive personal
  data and without overloading the server.

Installation (once, in the Anaconda Prompt):
    pip install pandas requests beautifulsoup4 lxml openpyxl
"""

#%% CONFIGURATION (the only place to edit)

MODE = "selectors"              # "selectors" or "tables"

# --- Pages to visit (use ONE of the three options) ------------------------
URLS = ["https://quotes.toscrape.com/"]         # fixed list of pages
URL_TEMPLATE = None             # e.g. "https://site.com/list?page={page}"
PAGES = range(1, 4)             # used with URL_TEMPLATE
NEXT_SELECTOR = "li.next a"     # CSS of the "next page" link (None = do not follow)
MAX_PAGES = 5

# --- "selectors" MODE: one item = one table row ---------------------------
ITEM_SELECTOR = "div.quote"
FIELDS = {                      # column name: (CSS selector inside the item, what to extract)
    "quote": ("span.text", "text"),
    "author": ("small.author", "text"),
    "author_link": ("a[href*='author']", "href"),
    "tags": ("a.tag", "list"),  # "list" = all elements, separated by ;
}

# --- "tables" MODE --------------------------------------------------------
TABLE_INDEX = "all"             # "all" or the table number on the page (0, 1, ...)
TABLE_CONTAINS = None           # optional: only tables that contain this text

# --- Behavior -------------------------------------------------------------
WAIT_SECONDS = 1.0              # pause between requests (courtesy to the server)
RESPECT_ROBOTS = True
RETRIES = 3
TIMEOUT = 20
USER_AGENT = "Mozilla/5.0 (academic-scraping; contact: your-email@example.com)"
CONVERT_NUMBERS = True          # "R$ 1.234,56" → 1234.56; "25°C" → 25

OUTPUT_FOLDER = "results"
FILE_NAME = "scraped_data"


#%% Packages

from pathlib import Path
from io import StringIO
from urllib import robotparser
from urllib.parse import urljoin, urlparse
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


#%% Utility functions

SCRIPT_DIR = (Path(__file__).resolve().parent
              if "__file__" in globals() else Path.cwd())
OUTPUT = (SCRIPT_DIR / OUTPUT_FOLDER).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
REPORT = []
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})
_robots = {}


def log(text=""):
    print(text)
    REPORT.append(str(text))


def allowed(url):
    """Check the domain's robots.txt (result cached)."""
    if not RESPECT_ROBOTS or not url.lower().startswith(("http://", "https://")):
        return True
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    if base not in _robots:
        rp = robotparser.RobotFileParser()
        try:
            response = session.get(base + "/robots.txt", timeout=TIMEOUT)
            rp.parse(response.text.splitlines() if response.ok else [])
        except requests.RequestException:
            rp.parse([])
        _robots[base] = rp
    return _robots[base].can_fetch(USER_AGENT, url)


def download(url):
    """GET with retries and increasing waits on temporary errors.
    Also accepts an HTML file saved on the computer (path or file://)."""
    if not url.lower().startswith(("http://", "https://")):
        local_path = Path(url.replace("file://", ""))
        if not local_path.is_absolute():
            local_path = SCRIPT_DIR / local_path
        return local_path.read_text(encoding="utf-8", errors="replace")
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            r.encoding = r.apparent_encoding or r.encoding
            return r.text
        except requests.RequestException as error:
            if attempt == RETRIES:
                raise
            time.sleep(WAIT_SECONDS * 2 ** attempt)


def text_to_number(series):
    """
    Convert the column into numbers when >= 90% of the values are numeric.
    Detects the format from the whole column: "1.234,56" / "44.411.238"
    (Brazilian) or "1,234.56" (English). Removes R$, %, °C, spaces, etc.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    t = series.astype(str).str.strip().str.replace(r"[^\d,.\-]", "", regex=True)
    brazilian = (t.str.contains(r",\d{1,2}$").any()
                 or t.str.match(r"^-?\d{1,3}(\.\d{3}){2,}$").any())
    if brazilian:
        t = t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        t = t.str.replace(",", "", regex=False)
    num = pd.to_numeric(t, errors="coerce")
    valid = series.notna() & (series.astype(str).str.strip() != "")
    return num if valid.any() and num[valid].notna().mean() >= 0.9 else series


def extract_items(soup, url):
    """Selectors MODE: one row per item, one column per field."""
    rows = []
    for item in soup.select(ITEM_SELECTOR):
        row = {}
        for name, (selector, kind) in FIELDS.items():
            elements = item.select(selector) if selector else [item]
            if not elements:
                row[name] = None
            elif kind == "list":
                row[name] = "; ".join(e.get_text(strip=True) for e in elements)
            elif kind == "text":
                row[name] = elements[0].get_text(" ", strip=True)
            else:  # attribute (href, src, title, data-...)
                value = elements[0].get(kind)
                if kind in ("href", "src") and value:
                    value = urljoin(url, value)
                row[name] = value
        row["source_url"] = url
        rows.append(row)
    return rows


def extract_tables(html, url):
    """Tables MODE: every <table> on the page as a DataFrame."""
    try:
        tables = pd.read_html(StringIO(html), match=TABLE_CONTAINS or ".+")
    except ValueError:
        return []
    if TABLE_INDEX != "all":
        tables = [tables[int(TABLE_INDEX)]] if int(TABLE_INDEX) < len(tables) else []
    for i, t in enumerate(tables):
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [" | ".join(str(n) for n in c if "Unnamed" not in str(n))
                         for c in t.columns]
        t["source_url"], t["table_on_page"] = url, i
    return tables


#%% 1. Build the page queue

if URL_TEMPLATE:
    queue = [URL_TEMPLATE.format(page=p) for p in PAGES]
else:
    queue = list(URLS)
visited, rows, tables, errors = [], [], [], []


#%% 2. Collect

while queue and len(visited) < MAX_PAGES:
    url = queue.pop(0)
    if url in visited:
        continue
    if not allowed(url):
        log(f"- Blocked by robots.txt (not collected): {url}")
        visited.append(url)
        continue
    try:
        html = download(url)
    except Exception as error:
        errors.append((url, str(error)))
        visited.append(url)
        continue
    visited.append(url)
    soup = BeautifulSoup(html, "html.parser")
    if MODE == "selectors":
        new_rows = extract_items(soup, url)
        rows.extend(new_rows)
        print(f"  {url}: {len(new_rows)} items")
    else:
        new_tables = extract_tables(html, url)
        tables.extend(new_tables)
        print(f"  {url}: {len(new_tables)} tables")
    if NEXT_SELECTOR and not URL_TEMPLATE:
        next_link = soup.select_one(NEXT_SELECTOR)
        if next_link and next_link.get("href"):
            queue.append(urljoin(url, next_link["href"]))
    time.sleep(WAIT_SECONDS)


#%% 3. Organize, clean and save

log("## Collection")
log(f"- Mode: {MODE} | pages visited: {len(visited)} | errors: {len(errors)}")
for url, error in errors:
    log(f"  - Error at {url}: {error}")

if MODE == "selectors":
    result = pd.DataFrame(rows)
    if result.empty:
        log("- No items found: check ITEM_SELECTOR in the browser "
            "(right click → Inspect → copy selector).")
    empty_fields = [c for c in FIELDS if c in result and result[c].isna().all()]
    if empty_fields:
        log(f"- Fields always empty (selector probably wrong): {empty_fields}")
    if CONVERT_NUMBERS and not result.empty:
        for c in FIELDS:
            if c in result and FIELDS[c][1] == "text":
                result[c] = text_to_number(result[c])
    dfs = {FILE_NAME: result}
else:
    dfs = {f"{FILE_NAME}_{i + 1}": t for i, t in enumerate(tables)}
    if CONVERT_NUMBERS:
        for t in dfs.values():
            for c in t.columns:
                if c not in ("source_url", "table_on_page"):
                    t[c] = text_to_number(t[c])

for name, df in dfs.items():
    if df.empty:
        continue
    df.to_csv(OUTPUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    log(f"- `{name}.csv`: {df.shape[0]} rows x {df.shape[1]} columns")
if dfs and any(not d.empty for d in dfs.values()):
    with pd.ExcelWriter(OUTPUT / f"{FILE_NAME}.xlsx") as writer:
        for name, df in list(dfs.items())[:50]:
            if not df.empty:
                df.to_excel(writer, sheet_name=name[-31:], index=False)
    first = next(d for d in dfs.values() if not d.empty)
    log("\n### Sample")
    log(first.head(5).to_string(max_colwidth=50))

(OUTPUT / "report.md").write_text("# Report: Web Scraping\n\n" +
                                  "\n".join(REPORT) + "\n", encoding="utf-8")
print(f"\nFiles in: {OUTPUT}")
