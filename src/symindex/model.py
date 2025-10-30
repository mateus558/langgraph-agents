"""Data models backing the semantic symbol index (symindex).

This module defines light dataclasses that describe semantic units (modules,
classes, functions, fields, properties, imports, and type aliases) as well as
graph edges between those units.  The structures provide the canonical
representation used by higher-level query helpers and future enrichers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal

Visibility = Literal["public", "private"]


@dataclass
class Parameter:
    name: str
    kind: Literal["posonly", "poskw", "vararg", "kwonly", "varkw"]
    annotation: Optional[str] = None
    default: Optional[str] = None


@dataclass
class SemanticUnit:
    kind: str
    name: str
    qualname: str
    module: str
    filepath: Optional[str]
    line_range: Tuple[int, int]
    docstring: Optional[str] = None
    visibility: Visibility = "public"
    exported: Optional[bool] = None
    source_snippet: Optional[str] = None
    id_sha: Optional[str] = None
    parent_qualname: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "qualname": self.qualname,
            "module": self.module,
            "filepath": self.filepath,
            "line_range": list(self.line_range),
            "docstring": self.docstring,
            "visibility": self.visibility,
            "exported": self.exported,
            "source_snippet": self.source_snippet,
            "id_sha": self.id_sha,
            "parent_qualname": self.parent_qualname,
        }


@dataclass
class ModuleUnit(SemanticUnit):
    imports: List[str] = field(default_factory=list)
    from_imports: List[str] = field(default_factory=list)
    exports: Optional[List[str]] = None
    top_level: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "imports": list(self.imports),
            "from_imports": list(self.from_imports),
            "exports": list(self.exports) if self.exports is not None else None,
            "top_level": list(self.top_level),
        })
        return data


@dataclass
class ClassUnit(SemanticUnit):
    bases: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    metaclass: Optional[str] = None
    type_params: List[str] = field(default_factory=list)
    abstract: bool = False
    slots_enabled: bool = False
    flags: Dict[str, bool] = field(default_factory=dict)
    fields: List[FieldUnit] = field(default_factory=list)
    properties: List[PropertyUnit] = field(default_factory=list)
    methods: List[FunctionUnit] = field(default_factory=list)
    nested_classes: List["ClassUnit"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "bases": list(self.bases),
            "decorators": list(self.decorators),
            "metaclass": self.metaclass,
            "type_params": list(self.type_params),
            "abstract": self.abstract,
            "slots_enabled": self.slots_enabled,
            "flags": dict(self.flags),
            "fields": [field.to_dict() for field in self.fields],
            "properties": [prop.to_dict() for prop in self.properties],
            "methods": [method.to_dict() for method in self.methods],
            "nested_classes": [nested.to_dict() for nested in self.nested_classes],
        })
        return data


@dataclass
class FunctionUnit(SemanticUnit):
    parameters: List[Parameter] = field(default_factory=list)
    return_annotation: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    raises: List[str] = field(default_factory=list)
    async_def: bool = False
    is_method: bool = False
    is_staticmethod: bool = False
    is_classmethod: bool = False

    def signature_string(self) -> str:
        parts: List[str] = []
        for param in self.parameters:
            frag = param.name
            if param.kind == "vararg":
                frag = f"*{frag}"
            elif param.kind == "varkw":
                frag = f"**{frag}"
            if param.annotation:
                frag = f"{frag}: {param.annotation}"
            if param.default:
                frag = f"{frag}={param.default}"
            parts.append(frag)
        ret = f" -> {self.return_annotation}" if self.return_annotation else ""
        return f"{self.name}({', '.join(parts)}){ret}"

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "parameters": [param.__dict__ for param in self.parameters],
            "return_annotation": self.return_annotation,
            "decorators": list(self.decorators),
            "roles": list(self.roles),
            "raises": list(self.raises),
            "async_def": self.async_def,
            "is_method": self.is_method,
            "is_staticmethod": self.is_staticmethod,
            "is_classmethod": self.is_classmethod,
        })
        return data


@dataclass
class FieldUnit(SemanticUnit):
    annotation: Optional[str] = None
    default: Optional[str] = None
    is_classvar: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "annotation": self.annotation,
            "default": self.default,
            "is_classvar": self.is_classvar,
        })
        return data


@dataclass
class PropertyUnit(SemanticUnit):
    type_annotation: Optional[str] = None
    accessor_roles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "type_annotation": self.type_annotation,
            "accessor_roles": list(self.accessor_roles),
        })
        return data


@dataclass
class ImportUnit(SemanticUnit):
    module_path: str = ""
    alias: Optional[str] = None
    import_from: Optional[str] = None
    level: int = 0
    type_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "module_path": self.module_path,
            "alias": self.alias,
            "import_from": self.import_from,
            "level": self.level,
            "type_only": self.type_only,
        })
        return data


@dataclass
class TypeAliasUnit(SemanticUnit):
    target: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({"target": self.target})
        return data


@dataclass
class Edge:
    type: str
    src_id: Optional[str]
    dst_id: Optional[str]
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "data": self.data,
        }
