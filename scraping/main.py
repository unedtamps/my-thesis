import asyncio
import csv
import datetime
import json
import os
import re

from playwright.async_api import Playwright, async_playwright

csv_file_path = "dataset/arxiv_all_attempt.csv"
file_exists = os.path.exists(csv_file_path)


def format_date(d) -> str:
    return d.strftime("%Y-%m-%d")


async def scrape_item(item):

    # title
    title_loc = await item.locator(".title").text_content()
    title = title_loc.strip()

    # url
    url = await item.locator(".list-title > a").get_attribute("href")

    # authors
    authors_locators = await item.locator(".authors a").all()
    authors = await asyncio.gather(*(a.text_content() for a in authors_locators))
    authors_str = ",".join(authors)

    tags_locators = await item.locator(".tags.is-inline-block span").all()
    tags_data = await asyncio.gather(
        *[
            asyncio.gather(a.text_content(), a.get_attribute("data-tooltip"))
            for a in tags_locators
        ]
    )
    tags_json_schema = [
        {"tag": tag.strip(), "name": tooltip.strip() if tooltip else None}
        for tag, tooltip in tags_data
    ]

    # abstract
    abstract_loc = await item.locator(".abstract-full").first.text_content()
    abstract = re.sub(r"\s*△ Less\s*$", "", abstract_loc).strip()

    # date
    meta_p = None
    all_meta_ps = await item.locator("p.is-size-7").all()
    for p in all_meta_ps:
        first_span = p.locator("span").first
        span_text = (await first_span.text_content()) or ""
        if span_text.strip() == "Submitted":
            meta_p = p
            break

    submitted_date, announced_date = None, None
    if meta_p is not None and await meta_p.count() > 0:
        submitted_span = meta_p.locator("span", has_text="Submitted")
        submitted_date = await submitted_span.evaluate(
            "el => el.nextSibling.textContent.trim().replace(/^;?\\s*/, '').replace(/;$/, '')"
        )
        announced_span = meta_p.locator("span", has_text="originally announced")
        announced_date = await announced_span.evaluate(
            "el => el.nextSibling.textContent.trim().replace(/^;?\\s*/, '').replace(/.$/, '')"
        )

    return {
        "title": title,
        "url": url,
        "authors": authors_str,
        "tags": tags_json_schema,
        "abstract": abstract,
        "submitted_date": submitted_date,
        "announced_date": announced_date,
    }


async def run(playwright: Playwright):
    count = 0
    chromium = playwright.chromium
    browser = await chromium.launch(headless=True)
    context = await browser.new_context()
    page = await context.new_page()
    start_date = datetime.datetime(2025, 11, 15)
    end_date = datetime.datetime(2025, 11, 16)
    writer = None
    with open(csv_file_path, "a", newline="", encoding="utf-8") as f:
        while True:
            start_date = start_date + datetime.timedelta(days=+1)
            end_date = end_date + datetime.timedelta(days=+1)

            start_date_str = format_date(start_date)
            end_date_str = format_date(end_date)

            if start_date > (datetime.datetime.now() + datetime.timedelta(days=+1)):
                print("Up until this day")
                break

            await page.goto(
                f"https://arxiv.org/search/advanced?advanced=1&terms-0-operator=AND&terms-0-term=&terms-0-field=title&classification-physics_archives=all&classification-include_cross_list=include&date-year=&date-filter_by=date_range&date-from_date={start_date_str}&date-to_date={end_date_str}&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first",
                timeout=120_000,
                wait_until="domcontentloaded",
            )

            while True:
                all_items = await page.locator(".arxiv-result").all()
                results = await asyncio.gather(*(scrape_item(i) for i in all_items))
                if len(results) > 0:
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=results[0].keys())
                        if not file_exists:
                            writer.writeheader()
                    for row in results:
                        writer.writerow(
                            {
                                k: (
                                    json.dumps(v, ensure_ascii=False)
                                    if isinstance(v, (list, dict))
                                    else v
                                )
                                for k, v in row.items()
                            }
                        )
                    count += len(results)
                    print(f"{count} Rows Saved")
                else:
                    print(f"Data Not Found in {start_date_str} to {end_date_str}")
                    break
                next_button = page.locator(".pagination-next").first
                if not await next_button.is_visible():
                    print(f"Done scraping from {start_date_str} to {end_date_str}")
                    break
                await next_button.click()
                await page.wait_for_load_state("networkidle")


async def main():
    async with async_playwright() as playwright:
        await run(playwright)


asyncio.run(main())