"""The trip planning agent: a manual Claude tool-use loop.

A manual loop (rather than the SDK tool runner) is deliberate here: we want to
own exactly what gets recorded at each step, because the recording *is* the
thing being evaluated.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic

from .tools import TOOL_DEFS, RunContext
from .tracing import Tracer

DEFAULT_MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def model_params(model: str) -> dict:
    """Request settings that differ by model family.

    Haiku 4.5 predates adaptive thinking and takes a fixed thinking budget. Server-side
    refusal fallbacks are enabled for the Opus/Fable tier, where they're recommended."""
    if model.startswith("claude-haiku-4-5"):
        return {"thinking": {"type": "enabled", "budget_tokens": 4000}}
    params = {"thinking": {"type": "adaptive", "display": "summarized"}}
    if model.startswith(("claude-opus-5", "claude-fable-5")):
        params.update(betas=[FALLBACK_BETA], fallbacks="default")
    return params

SYSTEM_PROMPT = """You are a trip planning assistant that finds round-trip flights within a user's budget and dates. Today's date is {today}.

How budgets work: the budget is the maximum round-trip price per person in USD, inclusive (a fare equal to the budget fits).

Working rules:
- Only state flight facts (prices, times, airlines, flight numbers, stops) that appear in search_flights results. If a search fails or returns nothing, say so; never fill the gap with a plausible-looking flight.
- A stop limit the user REQUIRES ("nonstop only", "no more than one stop") is a hard constraint: pass it as max_stops to both search_flights and check_eligibility. A mere preference ("direct if possible") is not: leave max_stops null and apply it when ranking.
- Never relax the user's constraints on your own. If nothing fits, report that nothing fits, and pass the user's real budget and dates to check_eligibility.
- For a city with several airports, search all of its commercial airports in one call (comma-separated codes) and say which airports you searched.
- If the dates are invalid (return before departure, in the past) or too vague to search, don't guess: submit status needs_clarification and explain what you need.
- If searches fail (API error, rate limit), you may retry once; if it still fails, submit status cannot_complete.
- Before simulate_book, run check_eligibility on the offer. Book only if the user asked you to book and the offer passed. Book at most one offer.
- When several offers are eligible, recommend by: lowest price, then fewest stops, then shortest total duration, then earliest departure. If the user says they prefer direct or nonstop flights, rank by fewest stops first, then lowest price, then duration, then departure; a connecting flight is still acceptable when no eligible nonstop exists. Say which rule you used.
- Avoid redundant calls: don't repeat a search with identical arguments; check all candidate offers in one check_eligibility call.
- Finish by calling submit_answer exactly once. List every offer that passed eligibility in eligible_offer_ids."""


def _dump(blocks) -> list[dict]:
    # Thinking-block signatures are opaque verification blobs; they add noise to traces.
    return [b.model_dump(exclude_none=True, exclude={"signature"}) for b in blocks]


def _context(system: str, messages: list[dict]) -> list[dict]:
    """The full context the model sees this turn, role-labeled so Langfuse renders it as a chat."""
    out = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if isinstance(content, list):  # assistant turns hold SDK blocks; tool results are plain dicts
            content = [b.model_dump(exclude_none=True, exclude={"signature"}) if hasattr(b, "model_dump") else b
                       for b in content]
        out.append({"role": m["role"], "content": content})
    return out


def run_tools(tool_uses, ctx: RunContext, tracer: Tracer, parallel_search: bool = True) -> dict:
    """Execute one turn's tool calls. Returns {tool_use_id: (content, is_error, start, duration_ms)}.

    Searches are read-only and dominated by network wait, so when Claude requests several
    in one turn they're fetched concurrently (registered first, in order, so ids stay
    deterministic). Everything else runs one at a time in the order Claude asked."""
    out = {}
    searches = [tu for tu in tool_uses if tu.name == "search_flights"] if parallel_search else []
    if len(searches) > 1:
        prepared = []
        for tu in searches:
            t0 = time.perf_counter()
            try:
                record, error = ctx.prepare_search(**tu.input)
            except TypeError as e:
                record, error = None, {"error": f"bad arguments: {e}"}
            if error:
                out[tu.id] = (json.dumps(error), True, t0, round((time.perf_counter() - t0) * 1000, 1))
            else:
                prepared.append((tu, record))
        parent = tracer.current_context()

        def fetch(item):
            tu, record = item
            with tracer.attached(parent), tracer.observation(as_type="tool", name=tu.name, input=tu.input) as span:
                t0 = time.perf_counter()
                result, is_error = ctx.fetch_search(record)
                content = json.dumps(result)
                duration_ms = round((time.perf_counter() - t0) * 1000, 1)
                span.update(output=content, level="ERROR" if is_error else "DEFAULT", metadata={"parallel": True})
            return tu.id, (content, is_error, t0, duration_ms)

        with ThreadPoolExecutor(max_workers=min(4, len(prepared)) or 1) as pool:
            out.update(dict(pool.map(fetch, prepared)))
    for tu in tool_uses:
        if tu.id in out:
            continue
        with tracer.observation(as_type="tool", name=tu.name, input=tu.input) as span:
            t0 = time.perf_counter()
            content, is_error = ctx.execute(tu.name, tu.input)
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            span.update(output=content, level="ERROR" if is_error else "DEFAULT")
        out[tu.id] = (content, is_error, t0, duration_ms)
    return out


def run_agent(
    request: str,
    ctx: RunContext,
    tracer: Tracer | None = None,
    *,
    model: str | None = None,
    max_turns: int = 12,
    client: anthropic.Anthropic | None = None,
    parallel_search: bool = True,
) -> dict:
    """Run one conversation. Tool calls land on ctx; returns run metadata."""
    tracer = tracer or Tracer(enabled=False)
    client = client or anthropic.Anthropic()
    model = model or os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
    system = SYSTEM_PROMPT.format(today=ctx.today.isoformat())
    messages: list[dict] = [{"role": "user", "content": request}]
    usage = {"input_tokens": 0, "output_tokens": 0}
    stop = "max_turns"
    run_t0 = time.perf_counter()

    def offset_ms(t: float) -> float:  # ms since the run started, for timeline views
        return round((t - run_t0) * 1000, 1)

    for turn in range(max_turns):
        with tracer.observation(as_type="generation", name="decide-next-step", model=model,
                                input=_context(system, messages), metadata={"turn": turn}) as gen:
            llm_t0 = time.perf_counter()
            try:
                response = client.beta.messages.create(
                    model=model,
                    max_tokens=16000,
                    system=system,
                    tools=TOOL_DEFS,
                    messages=messages,
                    **model_params(model),
                )
            except anthropic.APIStatusError as e:
                gen.update(level="ERROR", status_message=f"{e.status_code}: {e.message}")
                ctx.events.append({"type": "llm_error", "turn": turn, "status": e.status_code, "message": e.message})
                stop = "llm_error"
                break
            except anthropic.APIConnectionError as e:
                gen.update(level="ERROR", status_message=str(e))
                ctx.events.append({"type": "llm_error", "turn": turn, "message": str(e)})
                stop = "llm_error"
                break
            gen.update(output=_dump(response.content), usage_details={
                "input": response.usage.input_tokens, "output": response.usage.output_tokens})
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
        ctx.events.append({"type": "llm_call", "turn": turn, "start_ms": offset_ms(llm_t0),
                           "duration_ms": round((time.perf_counter() - llm_t0) * 1000, 1),
                           "input_tokens": response.usage.input_tokens,
                           "output_tokens": response.usage.output_tokens})

        # Record the agent's visible reasoning and text so the trajectory is complete.
        for block in response.content:
            if block.type == "thinking" and block.thinking:
                ctx.events.append({"type": "thinking", "turn": turn, "text": block.thinking})
            elif block.type == "text" and block.text:
                ctx.events.append({"type": "text", "turn": turn, "text": block.text})

        if response.stop_reason == "refusal":
            ctx.events.append({"type": "refusal", "turn": turn,
                               "details": response.stop_details.model_dump() if response.stop_details else None})
            stop = "refusal"
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            stop = response.stop_reason or "end_turn"
            break

        outcomes = run_tools(tool_uses, ctx, tracer, parallel_search)
        results = []
        for tu in tool_uses:
            content, is_error, t0, duration_ms = outcomes[tu.id]
            ctx.events.append({"type": "tool_call", "turn": turn, "name": tu.name, "input": tu.input,
                               "output": content, "is_error": is_error, "duration_ms": duration_ms,
                               "start_ms": offset_ms(t0), "parallel": parallel_search and tu.name == "search_flights"
                               and sum(t.name == "search_flights" for t in tool_uses) > 1})
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": content,
                            "is_error": is_error})

        if ctx.final_answer is not None:
            stop = "submitted"
            break
        messages.append({"role": "user", "content": results})

    return {"model": model, "stop": stop, "turns": turn + 1, "usage": usage}
