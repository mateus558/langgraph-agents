"""High-level semantic index API wrapping the symindex parsing pipeline.

``SemanticIndex`` stores the module-level semantic graph and exposes helper
queries.  ``build_semantic_index`` is the main entry point for consumers that
need to analyse Python source code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .model import ClassUnit, Edge, FunctionUnit, ModuleUnit, SemanticUnit
from .parser import SemanticParser


class SymindexParseError(Exception):
    pass


@dataclass
class SemanticIndex:
    module: ModuleUnit
    units: List[SemanticUnit]
    edges: List[Edge]
    errors: List[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_qualname: Dict[str, List[SemanticUnit]] = {}
        for unit in self.units:
            self._by_qualname.setdefault(unit.qualname, []).append(unit)
        self._edge_types: Dict[str, List[Edge]] = {}
        for edge in self.edges:
            self._edge_types.setdefault(edge.type, []).append(edge)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------
    def find_by_qualname(self, qualname: str) -> List[SemanticUnit]:
        return list(self._by_qualname.get(qualname, []))

    def list_public_api(self, module: Optional[str] = None) -> List[SemanticUnit]:
        module_name = module or self.module.qualname
        return [
            unit
            for unit in self.units
            if unit.module == module_name
            and unit.kind in {"class", "function", "async_function", "property", "field"}
            and unit.visibility == "public"
            and unit.parent_qualname is None
        ]

    def list_data_models(self, module: Optional[str] = None) -> List[ClassUnit]:
        module_name = module or self.module.qualname
        result: List[ClassUnit] = []
        for unit in self.units:
            if isinstance(unit, ClassUnit) and unit.module == module_name:
                if unit.flags.get("is_dataclass") or unit.flags.get("is_pydantic") or unit.flags.get("is_enum"):
                    result.append(unit)
        return result

    def list_overrides(self, class_qualname: str) -> List[Tuple[FunctionUnit, str]]:
        # Overrides not yet implemented in phase 1
        return []

    def graph_exports(self, module: Optional[str] = None) -> List[Edge]:
        return list(self._edge_types.get("exports", []))

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "module": self.module.to_dict(),
            "units": [unit.to_dict() for unit in self.units],
            "edges": [edge.to_dict() for edge in self.edges],
            "errors": list(self.errors),
        }

def build_semantic_index(
    *, src: str, module: str, filepath: Optional[str], tolerate_errors: bool = False
) -> SemanticIndex:
    parser = SemanticParser(src=src, module=module, filepath=filepath)
    if tolerate_errors:
        artifacts = parser.parse(tolerate_errors=True)
        return SemanticIndex(
            module=artifacts.module,
            units=artifacts.units,
            edges=artifacts.edges,
            errors=[
                {
                    "message": err.message,
                    "lineno": err.lineno,
                    "col_offset": err.col_offset,
                    "text": err.text,
                }
                for err in artifacts.errors
            ],
        )
    try:
        artifacts = parser.parse(tolerate_errors=False)
    except SyntaxError as e:
        # Provide a clearer, stable exception type upstream
        text = getattr(e, "text", "").rstrip("\n") if getattr(e, "text", None) else ""
        loc = f" (line {getattr(e, 'lineno', '?')}, col {getattr(e, 'offset', '?')})"
        raise SymindexParseError(f"Syntax error parsing module '{module}': {e}{loc}\n{text}") from e

    return SemanticIndex(
        module=artifacts.module,
        units=artifacts.units,
        edges=artifacts.edges,
        errors=[],
    )
