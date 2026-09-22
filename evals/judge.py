"""Optional LLM-as-judge for reasoning quality.

This is the one rubric item that isn't exact-match checkable: "does the
explanation follow from the data?" Everything with a right answer (eligibility,
grounding, tool order) is graded in code in scorers.py, because a judge there
would be slower, costlier, and less reliable than a comparison.
"""

from __future__ import annotations

import json
import os

import anthropic

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "description": "1 (reasoning doesn't follow) to 5 (fully follows)"},
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "rationale": {"type": "string"},
    },
    "required": ["score", "verdict", "rationale"],
    "additionalProperties": False,
}

JUDGE_PROMPT = """You are grading a flight-search assistant's explanation. Judge only whether the explanation logically follows from the data the assistant retrieved. Factual accuracy of individual numbers is checked elsewhere; focus on reasoning.

A passing explanation:
- justifies the recommendation (or the absence of one) using the retrieved offers and the eligibility results
- applies the ranking rule consistently: lowest price, then fewest stops, then shortest duration, then earliest departure; or, if the user asked for direct flights, fewest stops first, then lowest price
- doesn't claim a trade-off, comparison or conclusion that the data doesn't support
- if nothing was found or a search failed, says so plainly instead of implying options exist

<user_request>
{request}
</user_request>

<retrieved_offers>
{offers}
</retrieved_offers>

<eligibility_results>
{eligibility}
</eligibility_results>

<search_errors>
{errors}
</search_errors>

<assistant_final_answer>
{answer}
</assistant_final_answer>"""


def judge_reasoning(run: dict, client: anthropic.Anthropic | None = None) -> dict:
    ans = run["final_answer"]
    if ans is None:
        return {"pass": None, "details": ["N/A: no final answer"]}
    client = client or anthropic.Anthropic()
    prompt = JUDGE_PROMPT.format(
        request=run["request"],
        offers=json.dumps(list(run["offers"].values()), indent=1)[:60000],
        eligibility=json.dumps(run["eligibility_log"], indent=1),
        errors=json.dumps([s["error"] for s in run["searches"] if s["error"]]),
        answer=json.dumps(ans, indent=1),
    )
    response = client.messages.create(
        model=os.environ.get("JUDGE_MODEL", "claude-opus-5"),
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason != "end_turn":
        return {"pass": None, "details": [f"judge stopped with {response.stop_reason}"]}
    data = json.loads(next(b.text for b in response.content if b.type == "text"))
    return {"pass": data["verdict"] == "pass", "score": data["score"], "details": [data["rationale"]]}
