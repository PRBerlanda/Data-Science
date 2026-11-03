# 14 · Web Scraping

A ready-to-use tool that turns **web pages into tables** (CSV and Excel) without rewriting code. It captures HTML tables directly, or extracts fields from "card" layouts (products, news, listings) using CSS selectors, and walks through several pages while following good scraping practices.

> Part of the [Data-Science](https://github.com/PRBerlanda/Data-Science/tree/main) repository. Each branch holds one model.

---

## What this tool does

Web scraping downloads the HTML of a page and extracts the parts you need. The tool has two modes:

| Mode | Use it when | How it works |
|---|---|---|
| `"tables"` | The data is in `<table>` elements. | Every table on the page is read with `pandas.read_html`. |
| `"selectors"` | The data is in repeated blocks ("cards"). | You give the CSS selector of one item and of each field inside it. Each item becomes one row. |

It can visit a fixed list of URLs, a URL template with a page number, or follow the "next page" link.

**Built-in good practices:** respects `robots.txt`, waits between requests, identifies itself with a User-Agent, retries temporary failures (429/5xx) with increasing waits, and converts numbers stored as text (currency, thousand separators, units).

> Use it only on sites whose terms allow scraping, never collect sensitive personal data, and keep a polite pace so you do not overload the server.

## When to use it

- Collecting public tables (statistics, rankings, price lists) for analysis.
- Building a dataset from listing pages (products, quotes, job posts, news headlines).
- Re-running the same collection periodically without touching the code.

**Prefer an API** when the site offers one: it is more stable and usually allowed explicitly. Pages that render content only with JavaScript need a browser-automation tool instead.

## How it works

1. **Build the page queue** from `URLS`, or from `URL_TEMPLATE` + `PAGES`.
2. For each page:
   - check `robots.txt`;
   - download with retries (local HTML files are also accepted, which is useful for testing);
   - extract items (selectors mode) or tables (tables mode);
   - optionally follow the "next page" link;
   - wait before the next request.
3. **Clean and save**: warn about selectors that matched nothing, convert numeric text, and save every table as CSV plus an Excel workbook.

## Inputs

There is no input dataset: the input is the list of pages and the extraction rules.

**Configuration** (the `CONFIGURATION` cell):

| Parameter | What it controls |
|---|---|
| `MODE` | `"selectors"` or `"tables"`. |
| `URLS` | Fixed list of pages (URLs or local HTML files). |
| `URL_TEMPLATE`, `PAGES` | A template such as `"https://site.com/list?page={page}"` and the page numbers. |
| `NEXT_SELECTOR`, `MAX_PAGES` | CSS of the "next page" link (`None` = do not follow) and the page limit. |
| `ITEM_SELECTOR` | CSS selector of each repeated item (selectors mode). |
| `FIELDS` | `{column: (CSS selector inside the item, what to extract)}`, where *what* is `"text"`, `"list"` (all matches joined by `;`) or an attribute such as `"href"` or `"src"`. |
| `TABLE_INDEX`, `TABLE_CONTAINS` | Which table(s) to keep in tables mode. |
| `WAIT_SECONDS`, `RESPECT_ROBOTS`, `RETRIES`, `TIMEOUT`, `USER_AGENT` | Politeness and robustness settings. Put your contact in the User-Agent. |
| `CONVERT_NUMBERS` | Convert numbers stored as text. |
| `OUTPUT_FOLDER`, `FILE_NAME` | Where results go and the base file name. |

**Finding a CSS selector:** in the browser, right-click the element → *Inspect* → right-click the HTML node → *Copy → Copy selector*. Then simplify it to a class, such as `div.quote`.

## Outputs

Saved in `results/`:

| File | Content |
|---|---|
| `report.md` | Mode, pages visited, errors, files created and a sample of the data. |
| `<FILE_NAME>.csv` | The collected data (one CSV per table in tables mode). |
| `<FILE_NAME>.xlsx` | The same data in an Excel workbook (one sheet per table). |

Each row also carries the `source_url` it came from (and `table_on_page` in tables mode).

## How to read the results

- **Pages visited vs. errors**: errors list the URLs that failed and why (timeout, HTTP status).
- **"Blocked by robots.txt"**: the site asks crawlers not to visit that path, and the tool respects it.
- **"No items found"** or **"Fields always empty"**: the selector does not match the page. Inspect the page again and adjust `ITEM_SELECTOR` or `FIELDS`.
- **Numeric conversion**: a text column becomes numeric only if at least 90% of its values convert cleanly. Otherwise it stays as text.
- Sites change their layout over time. If a working configuration suddenly returns nothing, the selectors probably need updating.

## How to run

```bash
pip install -r requirements.txt
```

1. Edit the `CONFIGURATION` cell: mode, pages and fields.
2. Run the script in Spyder (F5) or with `python web_scraping.py`.
3. Open `results/report.md` and the CSV/Excel files.

## Example configuration

The default configuration collects quotes, authors, author links and tags from **quotes.toscrape.com**, a sandbox site built for scraping practice, following the "next page" link for up to five pages.

## Credits

Built from the "scraping_colab" notebook of the Data Collection and Web Scraping module (MBA in Data Science and Analytics, USP/Esalq).
