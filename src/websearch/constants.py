"""Constants and patterns for WebSearch agent.

This module defines the allowed search categories and heuristic patterns
used for query classification.
"""

from typing import Tuple, List, Iterable

# ---------------------------
# Categorias suportadas (ajuste conforme sua instância)
# ---------------------------
ALLOWED_CATEGORIES: Tuple[str, ...] = (
    "general", "news", "images", "videos", "map", "music", "files",
    "it", "science", "social media", "shopping", "qa", "apps",
    "economics", "sports", "education"
)

# ---------------------------
# Heurística determinística (APENAS HINT)
# Formato: (pattern, categories, score_weight)
# ---------------------------
HEURISTIC_PATTERNS: List[Tuple[str, Iterable[str], int]] = [
    (r"\b(news|notícia|breaking|últimas|hoje|agora|manchetes|headline|election|eleição|vazou)\b", ["news"], 3),
    (r"\b(image|wallpaper|foto|photos?|png|jpg|jpeg|logo|ícone|icon|sprite)\b", ["images"], 3),
    (r"\b(video|trailer|watch|assistir|youtube|mp4|mkv)\b", ["videos"], 3),
    (r"\b(map|maps|near me|perto de mim|rota|como chegar|nominatim|osm)\b", ["map"], 3),
    (r"\b(mp3|flac|lyrics|letra|música|song|playlist|spotify|soundcloud)\b", ["music"], 3),
    (r"\b(pdf|torrent|magnet|zip|rar|7z|index of|filetype:|download)\b", ["files"], 3),
    (r"\b(github|stack ?overflow|stackexchange|gist|pip|npm|pypi|maven|gradle|kotlin|java|python|golang|rust|error|exception|stackoverflow)\b", ["it"], 3),
    (r"\b(arxiv|doi:|preprint|paper|research|estudo|artigo científico|scholar|semantic scholar)\b", ["science"], 3),
    (r"\b(reddit|twitter|x\.com|instagram|tiktok|facebook|mastodon|lemmy)\b", ["social media"], 3),
    (r"\b(comprar|buy|preço|price|review|compar(ação|e)|deal|promo|cupom|coupon|aliexpress|amazon|ebay|mercado livre)\b", ["shopping"], 3),
    (r"\b(stack ?overflow|superuser|askubuntu|serverfault|quora)\b", ["qa"], 2),
    (r"\b(apk|fdroid|app store|play store|apkmirror|apkpure)\b", ["apps"], 2),
    (r"\b(cdi|selic|inflação|inflation|gdp|econom(y|ia)|finance|stocks?|ticker|ibovespa|nasdaq|dólar|usd/brl)\b", ["economics"], 2),
    (r"\b(futebol|football|soccer|nba|nfl|ufc|f1|formula 1|libertadores|brasileirão|premier league|score)\b", ["sports"], 2),
    (r"\b(curso|tutorial|how to|learn|mooc|udemy|coursera|khan academy|aula)\b", ["education"], 1),
]
