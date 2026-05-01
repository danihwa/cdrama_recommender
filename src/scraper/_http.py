import requests
from bs4 import BeautifulSoup

# Bot-style User-Agent following the Googlebot/Bingbot convention: looks
# browser-ish enough to pass edge filters, but the `compatible;` token
# identifies this as an automated client. If you re-run this scraper, add
# your own `+mailto:you@example.com` token so the site owner can reach you.
# Paired with a 1.5s delay in run_scrape.py.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; cdrama-recommender/1.0)"
}

def get_page(url: str) -> BeautifulSoup:
    """
    Fetches the content of a webpage and returns a parsed BeautifulSoup object. Raises HTTPError on non-2xx responses.

    Args:
        url (str): The URL of the webpage to fetch.

    Returns:
        BeautifulSoup: A BeautifulSoup object containing the parsed HTML content.
    """
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()  
    return BeautifulSoup(response.content, "html.parser")