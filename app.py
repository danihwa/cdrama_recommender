"""Streamlit chat UI for the cdrama recommender.

Run with: uv run streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from openai import OpenAI

from src.database.connection import get_db_connection
from src.env import load_secrets
from src.recommender.pipeline import run_rag


st.set_page_config(
    page_title="C-Drama Recommender",
    page_icon="🎬",
    layout="wide",
)


@st.cache_resource
def get_clients():
    load_secrets()
    return get_db_connection(), OpenAI()


@st.cache_data
def load_drama_titles() -> list[str]:
    """Fetch all drama titles from Supabase, sorted alphabetically."""
    supabase = get_db_connection()
    response = supabase.table("cdramas").select("title").order("title").execute()
    return [row["title"] for row in response.data]


supabase, openai = get_clients()


drama_titles = load_drama_titles()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "candidates" not in st.session_state:
    st.session_state.candidates = {}
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = ""


HISTORY_WINDOW = 6

GENRE_OPTIONS = [
    "romance",
    "historical",
    "fantasy",
    "wuxia",
    "action",
    "comedy",
    "suspense",
    "mystery",
    "family",
    "workplace",
    "crime",
    "political",
    "thriller",
    "youth",
    "time travel",
]

EXAMPLE_QUERIES = [
    (
        "Like Nirvana in Fire, but shorter",
        "Something like Nirvana in Fire but shorter",
    ),
    (
        "Cozy romance from 2022",
        "A cozy romance from 2022 rated above 8",
    ),
    (
        "Avoid fantasy",
        "A slow-burn historical romance without fantasy or wuxia",
    ),
]


def build_filter_hint(
    min_score: float,
    min_year: int,
    include_genres: list[str],
    exclude_genres: list[str],
    exclude_titles: list[str],
) -> str:
    hints: list[str] = []
    if min_score > 0:
        hints.append(
            f"Only recommend dramas with an MDL rating of at least {min_score}."
        )
    if min_year > 1900:
        hints.append(f"Only recommend dramas from {min_year} or later.")
    if include_genres:
        hints.append(
            "Include only dramas in these genres: " + ", ".join(include_genres) + "."
        )
    if exclude_genres:
        hints.append("Avoid dramas in these genres: " + ", ".join(exclude_genres) + ".")
    if exclude_titles:
        hints.append(
            "Exclude these titles from the recommendations: "
            + ", ".join(exclude_titles)
            + "."
        )
    return " ".join(hints)


def render_candidate(candidate: dict) -> None:
    title = candidate.get("title", "Unknown title")
    url = candidate.get("mdl_url")
    year = candidate.get("year", "?")
    score = candidate.get("mdl_score", "?")
    similarity = candidate.get("similarity", 0.0)
    ensemble = candidate.get("ensemble_score", 0.0)
    watchers = candidate.get("watchers", "?")
    genres = candidate.get("genres") or []
    synopsis = candidate.get("synopsis", "")

    if url:
        st.markdown(f"### [{title}]({url})")
    else:
        st.markdown(f"### {title}")

    st.write(
        f"**{year}** · MDL score **{score}** · sim **{similarity:.2f}** · "
        f"ensemble **{ensemble:.2f}** · watchers **{watchers}**"
    )

    if genres:
        st.caption("Genres: " + ", ".join(genres))
    if synopsis:
        st.write(synopsis)
    st.markdown("---")


header_col, tips_col = st.columns([3, 1])
with header_col:
    st.title("🎬 C-Drama Recommender")
    st.markdown(
        "Find your next favorite Chinese drama with smart search filters, "
        "plot-aware recommendation, and ranked candidate previews."
    )
    st.info(
        "Try a reference title, a mood description, or structured filters like "
        "`romance from 2022 rated above 8`."
    )

def _use_example(query: str) -> None:
    st.session_state.prompt_input = query


with tips_col:
    st.subheader("Prompt examples")
    for label, query in EXAMPLE_QUERIES:
        st.button(label, key=label, on_click=_use_example, args=(query,))

    st.markdown("---")
    st.subheader("Tips")
    st.markdown(
        "- Use **`like <title>`** to find similar dramas.\n"
        "- Add **year**, **rating**, or **genre** for more control.\n"
        "- Say **`no fantasy`** or **`exclude wuxia`** to avoid genres.\n"
        "- Mention dramas you already saw to exclude them."
    )

with st.sidebar:
    st.header("Filters")
    min_score = st.slider(
        "Minimum rating",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
    )
    min_year = st.slider(
        "Earliest year",
        min_value=1900,
        max_value=2024,
        value=1900,
        step=1,
    )
    include_genres = st.multiselect(
        "Include genres",
        GENRE_OPTIONS,
        [],
        key="include_genres",
    )
    exclude_genres = st.multiselect(
        "Exclude genres",
        [g for g in GENRE_OPTIONS if g not in include_genres],
        [],
        key="exclude_genres",
    )
    exclude_titles_raw = st.text_input(
        "Exclude titles (comma-separated)",
        key="exclude_titles",
    )
    exclude_titles = [
        title.strip() for title in exclude_titles_raw.split(",") if title.strip()
    ]

    def _add_exclude_from_catalog() -> None:
        selected = st.session_state.selected_exclude_title
        if not selected:
            return
        current = st.session_state.get("exclude_titles", "")
        items = [t.strip() for t in current.split(",") if t.strip()]
        if selected not in items:
            items.append(selected)
            st.session_state.exclude_titles = ", ".join(items)
        st.session_state.selected_exclude_title = ""

    st.selectbox(
        "Exclude a title from the catalog",
        options=[""] + drama_titles,
        key="selected_exclude_title",
        on_change=_add_exclude_from_catalog,
    )

    if (
        min_score > 0
        or min_year > 1900
        or include_genres
        or exclude_genres
        or exclude_titles
    ):
        st.markdown("---")
        st.markdown("**Active filters:**")
        st.write(
            build_filter_hint(
                min_score, min_year, include_genres, exclude_genres, exclude_titles
            )
        )

    def _clear_chat() -> None:
        st.session_state.messages = []
        st.session_state.candidates = {}
        st.session_state.prompt_input = ""

    st.button("Clear chat", use_container_width=True, on_click=_clear_chat)


main_col, preview_col = st.columns([3, 1])
with main_col:
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            cands = st.session_state.candidates.get(idx)
            if cands:
                with st.expander(f"Top {len(cands)} candidates"):
                    for candidate in cands:
                        render_candidate(candidate)

    prompt = st.text_input(
        "What are you in the mood for?",
        key="prompt_input",
    )
    search = st.button("Search")
    if search:
        raw_prompt = prompt.strip()
        filter_hint = build_filter_hint(
            min_score,
            min_year,
            include_genres,
            exclude_genres,
            exclude_titles,
        )
        if not raw_prompt and not filter_hint:
            st.warning("Enter a query or select at least one filter.")
        else:
            user_prompt = raw_prompt or "Recommend dramas matching my selected filters."
            request_text = (
                f"{user_prompt}\n\n{filter_hint}" if filter_hint else user_prompt
            )

            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1][-HISTORY_WINDOW:]
            ]

            with st.chat_message("assistant"):
                with st.spinner("Searching through the dramas..."):
                    try:
                        response, top = run_rag(request_text, supabase, openai, history)
                    except Exception as e:
                        response, top = f"Something went wrong: `{e}`", []
                st.markdown(response)

            assistant_idx = len(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            if top:
                st.session_state.candidates[assistant_idx] = top
            st.rerun()

with preview_col:
    st.subheader("Latest candidate preview")
    if st.session_state.messages:
        last_idx = len(st.session_state.messages) - 1
        if st.session_state.messages[last_idx]["role"] == "assistant":
            cands = st.session_state.candidates.get(last_idx, [])
            if cands:
                for candidate in cands[:5]:
                    render_candidate(candidate)
            else:
                st.write("No candidate preview yet. Submit a query to see matches.")
        else:
            st.write("Submit a query to see the latest recommendations.")
    else:
        st.write("Start by asking for a drama recommendation.")
