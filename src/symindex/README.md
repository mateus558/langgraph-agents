# symindex

`symindex` is the semantic symbol indexing layer used by langgraph-agents. It turns a
Python source file into a graph of semantic units (modules, classes, functions,
fields, properties, imports, type aliases, etc.) and provides helper queries for
analyzing those units.

## Why

- Offer a structured representation that is richer than flat symbol lists.
- Enable future enrichers (docstring sections, overrides, exports) without
  changing downstream APIs.
- Keep the implementation stdlib-only so it can run in restricted environments.

## How it works

1. **Parse** – `SemanticParser` walks the AST and emits semantic units plus
   `contains` edges.
2. **Index** – `SemanticIndex` wraps those units, builds lookup tables, and
   exposes query helpers such as `find_by_qualname` and `list_public_api`.

The package lives at `src/utils/symindex`. See `docs/symindex.md` for a detailed
conceptual overview and diagrams.

## Quick start

```python
from pathlib import Path

from symindex import build_semantic_index

source = Path("example.py").read_text(encoding="utf-8")
index = build_semantic_index(src=source, module="example", filepath="example.py")

print([unit.qualname for unit in index.list_public_api()])
```

## Future work

- Add enrichers for roles (dataclass, pydantic, enum) and docstring sections.
- Discover additional edge types (overrides, imports, decorators).
- Provide serialization helpers for storing semantic graphs.
