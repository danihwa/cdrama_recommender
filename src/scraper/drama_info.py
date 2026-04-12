from bs4 import Tag
import re

from src.scraper._http import get_page

def _text(tag: Tag) -> str:
    """Returns stripped text content of a tag, or None if the tag is absent."""
    if tag:
        return tag.get_text(strip=True)
    return None


def get_drama_info(url: str) -> dict:
    """
    Scrapes a single drama page and returns a dictionary of fields.
    Most fields are extracted by finding a bold label (e.g. <b>Genres:</b>)
    and then reading the sibling elements(s) that follow it in the HTML.
    """
    soup = get_page(url)

    # URLs look like /9025-nirvana-in-fire -> we can extract the drama ID
    match = re.search(r"/(\d+)-", url)
    if match:
        mdl_id = int(match.group(1)) 
    else:
        mdl_id = None

    title = _text(soup.find("h1", class_="film-title"))

    # Native title is a plain text link that follows the <b>Native Title:</b> label
    native_title_b = soup.find("b", string="Native Title:")
    if native_title_b:
        native_title = _text(native_title_b.find_next_sibling("a"))    
    else:
        native_title = None

    # Synopsis lives in a dedicated div; prefer the schema.org <span> inside it
    # so we get just the description text without any surrounding UI chrome.
    synopsis_div = soup.find("div", class_="show-synopsis")
    if synopsis_div:
        synopsis_tag = synopsis_div.find("span", itemprop="description") or synopsis_div
        synopsis = _text(synopsis_tag)
    else:
        synopsis = None    

    # Episode count is a bare text node that immediately follows the label
    episodes = None
    episodes_b = soup.find("b", string="Episodes:")
    if episodes_b:
        raw_episodes = episodes_b.find_next_sibling(string=True).strip()
        if raw_episodes and raw_episodes.isdigit():
            episodes = int(raw_episodes)

    # Subtitle format is "Native Title ‧ Type ‧ Year" — year is always the last token.
    year = None
    subtitle = soup.find("div", class_="film-subtitle")
    if subtitle:
        m = re.search(r"\b(19|20)\d{2}\b", subtitle.get_text())
        if m:
            year = int(m.group()) 

    # All genre links are siblings of the <b>Genres:</b> label
    genres = []
    genres_section = soup.find("b", string="Genres:")
    if genres_section:
        for a in genres_section.find_next_siblings("a"):
            genres.append(_text(a))

    # Tags are <span> elements inside a dedicated <li>; each span holds one <a> link.
    tags_li = soup.find("li", class_="show-tags")
    if tags_li:
        tags = [
            _text(span.find("a", class_="text-primary"))
            for span in tags_li.find_all("span")
            if span.find("a", class_="text-primary")
        ]
    else:
        tags = [] 

    # Score follows the same label + sibling pattern as Episodes and Watchers.
    score = None
    score_b = soup.find("b", string="Score:")
    if score_b:
        raw_score = score_b.find_next_sibling(string=True).strip()
        if raw_score:
            try:
                score = float(raw_score)
            except ValueError:
                score = None

    # Watchers count follows the same label + sibling pattern as Episodes.
    watchers = None
    watchers_b = soup.find("b", string="Watchers:")
    if watchers_b:
        raw_watchers = watchers_b.find_next_sibling(string=True).strip()
        if raw_watchers:
            watchers = int(raw_watchers.replace(",", ""))

    return {
        "mdl_id": mdl_id,
        "mdl_url": url,
        "title": title,
        "native_title": native_title,
        "synopsis": synopsis,
        "episodes": episodes,
        "year": year,
        "genres": genres,
        "tags": tags,
        "mdl_score": score,
        "watchers": watchers,
    }


if __name__ == "__main__":
    url = "https://mydramalist.com/9025-nirvana-in-fire"
    print(f"Scraping: {url}\n")
    drama = get_drama_info(url)
    for key, value in drama.items():
        print(f"{key}: {value}")