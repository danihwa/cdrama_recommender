import requests
from bs4 import BeautifulSoup

# MDL and many other websites block requests that don't look like a real browser
# Sending a realistic User-Agent header can help bypass these blocks
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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