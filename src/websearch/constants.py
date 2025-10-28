"""Constants for web search categorization and patterns.

This module defines allowed categories and pattern matching rules for
categorizing search queries using heuristics.
"""

from typing import Iterable, List, Tuple

# Supported categories (adjust according to your instance)
ALLOWED_CATEGORIES: Tuple[str, ...] = (
    "general", "news", "images", "videos", "map", "music", "files",
    "it", "science", "social media", "shopping", "qa", "apps",
    "economics", "sports", "education"
)

# ---------------------------
# Deterministic heuristic (HINT ONLY)
# Format: (pattern, categories, score_weight)
# ---------------------------
HEURISTIC_PATTERNS: List[Tuple[str, Iterable[str], int]] = [
    (r"\b(news|breaking|latest|today|now|headlines?|headline|election|leak)\b", ["news"], 3),
    (r"\b(image|wallpaper|photo|photos?|png|jpg|jpeg|logo|icon|sprite)\b", ["images"], 3),
    (r"\b(video|trailer|watch|youtube|mp4|mkv)\b", ["videos"], 3),
    (r"\b(map|maps|near me|nearby|route|directions?|nominatim|osm)\b", ["map"], 3),
    (r"\b(mp3|flac|lyrics|music|song|playlist|spotify|soundcloud)\b", ["music"], 3),
    (r"\b(pdf|torrent|magnet|zip|rar|7z|index of|filetype:|download)\b", ["files"], 3),
    (r"\b(github|stack ?overflow|stackexchange|gist|pip|npm|pypi|maven|gradle|kotlin|java|python|golang|rust|error|exception|stackoverflow)\b", ["it"], 3),
    (r"\b(arxiv|doi:|preprint|paper|research|study|scientific article|scholar|semantic scholar)\b", ["science"], 3),
    (r"\b(reddit|twitter|x\.com|instagram|tiktok|facebook|mastodon|lemmy)\b", ["social media"], 3),
    (r"\b(buy|price|prices|review|compare|deal|promo|coupon|aliexpress|amazon|ebay|marketplace)\b", ["shopping"], 3),
    (r"\b(stack ?overflow|superuser|askubuntu|serverfault|quora)\b", ["qa"], 2),
    (r"\b(apk|fdroid|app store|play store|apkmirror|apkpure)\b", ["apps"], 2),
    (r"\b(inflation|gdp|econom(y|ics)|finance|stocks?|ticker|ibovespa|nasdaq|usd/brl|interest rate)\b", ["economics"], 2),
    (r"\b(football|soccer|nba|nfl|ufc|f1|formula 1|premier league|score|sports?)\b", ["sports"], 2),
    (r"\b(course|tutorial|how to|learn|mooc|udemy|coursera|khan academy|lesson)\b", ["education"], 1),
]
