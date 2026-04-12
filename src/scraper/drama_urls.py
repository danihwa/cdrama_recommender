import time

from src.scraper._http import get_page

# Query parameters decoded:
#   adv=titles  → advanced title search
#   ty=68       → type: TV drama (excludes movies and specials)
#   co=2        → country: China
#   rt=7.5,10   → rating range: 7.5–10
#   st=3        → status: completed (not airing)
#   so=top      → sort by: top-rated
BASE_URL = "https://mydramalist.com/search?adv=titles&ty=68&co=2&rt=7.5,10&st=3&so=top&page={page}"


def get_drama_urls_from_page(page: int) -> list[str]:
    """Scrape one search results page and return a list of drama URLs."""
    url = BASE_URL.format(page=page)
    soup = get_page(url)

    urls = []
    for a in soup.select("h6.title a"):
        href = a.get("href")
        if href:
            urls.append(f"https://mydramalist.com{href}")
    return urls


def get_all_drama_urls(max_pages: int = 136) -> list[str]:
    """Scrape all search pages and return all drama URLs.

    max_pages=136 reflects the approximate page count of Chinese dramas
    rated 7.5+ on MDL at time of writing. The early-exit guards against
    max_pages overshooting the actual page count.
    """
    all_urls = []
    for page in range(1, max_pages + 1):
        print(f"Scraping page {page}/{max_pages}...")
        urls = get_drama_urls_from_page(page)
        if not urls:
            print(f"No URLs found on page {page}, stopping.")
            break
        all_urls.extend(urls)
        time.sleep(1)
    return all_urls


if __name__ == "__main__":
    urls = get_all_drama_urls(max_pages=136)
    print(f"\nTotal drama URLs found: {len(urls)}")