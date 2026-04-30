"""
tests/evals/llm_judge.py
------------------------
LLM-as-judge eval suite for the cdrama recommender.

Checks relevance, grounding, and hallucinations on real queries.
Results are saved as JSON in tests/evals/results/.

Guardrails (refused mode) are tested in tests/evals/test_parse_user_query.py.

Usage:
    uv run python tests/evals/llm_judge.py

Cost estimate: ~$0.01 per run (gpt-4o-mini for both pipeline + judge)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from supabase import Client

from src.database.connection import get_db_connection
from src.env import load_secrets
from src.recommender.pipeline import run_rag

RESULTS_DIR = Path("tests/evals/results")

JUDGE_SYSTEM_PROMPT = """\
You are an impartial evaluator for a Chinese drama recommendation system.
You will be given a user query, the list of candidate dramas the system had access to,
and the system's response. Evaluate the response on the criteria provided.

Always respond with valid JSON only. No preamble, no markdown fences.
"""

QUALITY_JUDGE_PROMPT = """\
Evaluate this recommendation response.

USER QUERY: {query}

CANDIDATE DRAMAS (titles the system was allowed to recommend):
{candidate_titles}

SYSTEM RESPONSE:
{response}

Score each dimension from 1 to 5 and explain briefly.

Respond with this exact JSON structure:
{{
  "relevance": {{
    "score": <1-5>,
    "reason": "<one sentence>"
  }},
  "grounding": {{
    "score": <1-5>,
    "reason": "<one sentence — did it only recommend from the candidate list?>"
  }},
  "tone": {{
    "score": <1-5>,
    "reason": "<one sentence — warm, enthusiastic, appropriate for a recommender?>"
  }},
  "hallucinated_titles": ["<any title mentioned that is NOT in candidate_titles>"]
}}

Scoring guide:
  relevance  5=perfectly matches request  1=completely off
  grounding  5=all titles from candidates  1=invented titles
  tone       5=warm and helpful  1=cold or robotic
"""

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

QUALITY_CASES = [
    {
        "id": "political_intrigue_reference",
        "query": "Something like Nirvana in Fire with political intrigue",
    },
    {
        "id": "romance_high_score",
        "query": "Romance dramas from 2022 onwards, rating at least 8.5",
    },
    {
        "id": "exclude_and_find_next",
        "query": "I just finished The Story of Ming Lan — what's next?",
    },
    {
        "id": "wuxia_female_lead",
        "query": "Wuxia fantasy, strong female lead, no older than 2019",
    },
    {
        "id": "compound_exclude_score",
        "query": (
            "Recommend me something similar to How Dare You!? "
            "I already saw Dream Within a Dream. "
            "The drama should be rated above 8"
        ),
    },
    {
        "id": "mystery_thriller",
        "query": "Something like The Bad Kids, dark mystery thriller",
    },
    {
        "id": "light_comedy",
        "query": "Something like Go Ahead, family drama with heartwarming moments",
    },
]


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------

def judge_quality(query: str, response: str, candidates: list[dict], client: OpenAI) -> dict:
    candidate_titles = [d["title"] for d in candidates]
    prompt = QUALITY_JUDGE_PROMPT.format(
        query=query,
        candidate_titles="\n".join(f"- {t}" for t in candidate_titles),
        response=response,
    )
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(result.choices[0].message.content)


# ---------------------------------------------------------------------------
# Running suites
# ---------------------------------------------------------------------------

def run_quality_suite(openai: OpenAI, supabase: Client) -> list[dict]:
    print("\n── Quality Evals ───────────────────────────────────────")
    results = []

    for case in QUALITY_CASES:
        print(f"  [{case['id']}]...", end=" ", flush=True)
        try:
            response, candidates = run_rag(case["query"], supabase, openai)

            if not candidates:
                print("no results returned, skipping judge")
                results.append({"id": case["id"], "query": case["query"], "skipped": True})
                continue

            scores = judge_quality(case["query"], response, candidates, openai)
            rel = scores.get("relevance", {}).get("score", "?")
            grd = scores.get("grounding", {}).get("score", "?")
            hal = len(scores.get("hallucinated_titles", []))
            print(f"relevance={rel}/5  grounding={grd}/5  hallucinations={hal}")
            results.append({
                "id": case["id"],
                "query": case["query"],
                "response": response,
                "candidate_titles": [d["title"] for d in candidates],
                "scores": scores,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"id": case["id"], "query": case["query"], "error": str(e)})

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    load_secrets()
    openai = OpenAI()
    supabase = get_db_connection()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = run_quality_suite(openai, supabase)

    # summary
    print("\n── Summary ─────────────────────────────────────────────")
    judged = [r for r in results if r.get("scores")]
    if judged:
        avg_rel = sum(r["scores"]["relevance"]["score"] for r in judged) / len(judged)
        avg_grd = sum(r["scores"]["grounding"]["score"] for r in judged) / len(judged)
        avg_tone = sum(r["scores"]["tone"]["score"] for r in judged) / len(judged)
        total_hal = sum(len(r["scores"].get("hallucinated_titles", [])) for r in judged)
        print(
            f"  relevance={avg_rel:.1f}/5  grounding={avg_grd:.1f}/5  "
            f"tone={avg_tone:.1f}/5  hallucinations={total_hal}"
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"run_{ts}.json"
    out_path.write_text(json.dumps(
        {"run_at": datetime.now(timezone.utc).isoformat(), "results": results},
        indent=2,
        ensure_ascii=False,
    ))
    print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    main()
