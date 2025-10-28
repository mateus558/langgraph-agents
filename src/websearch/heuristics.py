"""Category heuristics for query classification.

This module provides heuristic-based and LLM-based category detection
for search queries, along with category normalization and validation.
"""

import re
from collections.abc import Iterable

from pydantic import BaseModel, Field, field_validator

from .constants import ALLOWED_CATEGORIES, HEURISTIC_PATTERNS


def heuristic_categories(query: str) -> list[str]:
    """Determine categories using regex patterns.

    Applies regex patterns to the query and scores categories based on
    pattern matches. Returns categories sorted by score (highest first).

    Args:
        query: The search query string.

    Returns:
        List of category names sorted by relevance score.
        Returns ["general"] if no patterns match.
    """
    q = query.lower()
    hits: dict[str, int] = {}

    for pattern, cats, base in HEURISTIC_PATTERNS:
        if re.search(pattern, q):
            for c in cats:
                hits[c] = hits.get(c, 0) + base

    if not hits:
        return ["general"]

    return [c for c, _ in sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))]


def sanitize_categories(cats: Iterable[str]) -> list[str]:
    """Normalize and validate categories.

    - Converts to lowercase
    - Maps aliases to canonical names (e.g., "social" -> "social media")
    - Filters to only allowed categories
    - Removes duplicates while preserving order
    - Returns ["general"] if all categories are invalid

    Args:
        cats: Iterable of category names.

    Returns:
        List of normalized, deduplicated, valid category names.
    """
    norm = []
    for c in cats:
        c0 = c.strip().lower()

        # Normalize aliases
        if c0 in {"social", "social-media"}:
            c0 = "social media"
        if c0 in {"tech", "technology"}:
            c0 = "it"
        if c0 in {"economic", "finance"}:
            c0 = "economics"
        if c0 in {"q&a", "questions", "question", "answers"}:
            c0 = "qa"
        if c0 in {"photo", "pics", "pictures"}:
            c0 = "images"
        if c0 in {"video"}:
            c0 = "videos"
        if c0 in {"maps"}:
            c0 = "map"
        if c0 in {"academic", "papers", "scholar"}:
            c0 = "science"

        if c0 in ALLOWED_CATEGORIES:
            norm.append(c0)

    if not norm:
        norm = ["general"]

    # Remove duplicates while preserving order
    seen = set()
    out = []
    for c in norm:
        if c not in seen:
            out.append(c)
            seen.add(c)

    return out


class CategoryResponse(BaseModel):
    """Schema for categorization response."""
    categories: list[str] = Field(..., description="List of chosen categories")

    @field_validator("categories", mode="before")
    def _sanitize(cls, v):
        """Validate and sanitize categories from LLM output."""
        v2 = sanitize_categories(v)
        if not v2:
            return ["general"]
        return v2[:3]  # Limit to top 3 categories
