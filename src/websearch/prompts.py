"""Reusable prompt definitions for the websearch domain."""

from __future__ import annotations

from typing import Sequence

from langchain_core.messages import BaseMessage

from core.prompts import ChatPrompt


WEBSEARCH_SUMMARY_PROMPT = ChatPrompt(
    id="websearch.summarize",
    version="1.0.0",
    system_template=(
        "You are a factual summarizer. Be concise, verifiable, objective. "
        "Always answer in the user's language.\n"
        "Current time (UTC): {utc_time}\n"
        "Current time ({local_label}): {local_time}\n\n"
        "TEMPORAL POLICY:\n"
        "• Interpret 'next', 'upcoming', and schedules strictly relative to the current time above.\n"
        "• Convert times to {local_label} when presenting times; include weekday and explicit date.\n"
        "• If a source mentions a past date, do NOT call it 'next'; mark it as 'already occurred'.\n"
        "• Prefer sources with explicit dates/times over vague claims; prefer newest when conflicting.\n"
        "• If no future date > now is found, say 'No upcoming date found in the results.' Do not guess.\n"
        "• For time-related queries, the 'Summary' MUST start with an 'As of <UTC>/<local>:' clause.\n"
    ),
    user_template=(
        "{whitelist}\n\n"
        "TASK: Summarize and answer the query using only the listed sources.\n"
        "MANDATORY OUTPUT FORMAT:\n"
        "Summary:\n"
        "- Start with: 'As of <YYYY-MM-DD HH:MM UTC / local>:'\n"
        "- If the user query is time related, include the next upcoming or most recent relevant date strictly compared to the current time.\n"
        "- Only add information that is relevant to the user query.\n"
        "Details (optional):\n"
        "- Add factual context such as periodicity, duration, or schedule patterns when the user query suggests so.\n"
        "Sources:\n"
        "- List only the citation indices used, ordered by importance (e.g., [1], [4], [2]).\n\n"
        "CITATION AND CONFLICT RULES:\n"
        "- Every date or quantitative claim must have at least one citation [n].\n"
        "- When multiple sources disagree, identify the conflict and prefer the newest data.\n"
        "- If the user query has time relationships: If no upcoming or relevant date is found, explicitly say: 'No future or current date found in the results.'\n\n"
        "Query: {query}\n\n"
        "RESULTS (to base your answer on):\n"
        "{results}"
    ),
)


def build_websearch_summary_messages(
    *,
    query: str,
    whitelist: str,
    results: str,
    utc_time: str,
    local_time: str,
    local_label: str,
) -> Sequence[BaseMessage]:
    """Render the websearch summarization prompt messages."""
    return WEBSEARCH_SUMMARY_PROMPT.messages(
        query=query,
        whitelist=whitelist,
        results=results,
        utc_time=utc_time,
        local_time=local_time,
        local_label=local_label,
    )


__all__ = ["WEBSEARCH_SUMMARY_PROMPT", "build_websearch_summary_messages"]
