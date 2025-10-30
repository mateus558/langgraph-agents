"""AST-to-semantic-unit parser for the Python symbol semantic index.

The ``SemanticParser`` mirrors the legacy ``PySymbolExtractor`` traversal but
emits ``SemanticUnit`` instances and ``Edge`` relationships.  It is
dependency-free (pure ``ast``) and designed to be the first stage in the
semantic indexing pipeline before additional resolvers or enrichers run.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from .model import (
    ClassUnit,
    Edge,
    FieldUnit,
    FunctionUnit,
    ModuleUnit,
    Parameter,
    PropertyUnit,
    SemanticUnit,
    Visibility,
)


@dataclass
class ParseError:
    message: str
    lineno: int
    col_offset: int
    text: str | None = None


@dataclass
class ParseArtifacts:
    module: ModuleUnit
    units: List[SemanticUnit]
    edges: List[Edge]
    errors: List[ParseError] = field(default_factory=list)


class SemanticParser(ast.NodeVisitor):
    """AST visitor that builds semantic units for a single Python module."""

    def __init__(self, src: str, module: str, filepath: Optional[str]) -> None:
        self.src = src
        self.module = module
        self.filepath = filepath

        self._exports: Optional[List[str]] = None
        self._imports: List[str] = []
        self._from_imports: List[str] = []
        self._module_doc: Optional[str] = None

        self._units: List[SemanticUnit] = []
        self._edges: List[Edge] = []
        self._class_stack: List[ClassUnit] = []
        self._qualname_stack: List[str] = []
        self._module_children: List[SemanticUnit] = []

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _sha256(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    def _segment(self, node: Optional[ast.AST]) -> Optional[str]:
        if node is None:
            return None
        try:
            seg = ast.get_source_segment(self.src, node)
            return seg.strip() if seg is not None else None
        except Exception:
            return None

    def _safe_unparse(self, node: Optional[ast.AST]) -> Optional[str]:
        if node is None:
            return None
        try:
            return ast.unparse(node)
        except Exception:
            return None

    def _annotation_text(self, node: Optional[ast.AST]) -> Optional[str]:
        return self._segment(node) or self._safe_unparse(node)

    def _line_range(self, node: ast.AST) -> Tuple[int, int]:
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start)
        return (start, end)

    def _visibility(self, name: str) -> Visibility:
        return "private" if name.startswith("_") else "public"

    def _qualname(self, name: str) -> str:
        if self._qualname_stack:
            return ".".join([*self._qualname_stack, name])
        return name

    def _exported(self, name: str) -> Optional[bool]:
        if self._exports is None:
            return None
        return name in self._exports

    def _decorator_texts(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> List[str]:
        decos: List[str] = []
        for deco in node.decorator_list:
            decos.append(self._segment(deco) or self._safe_unparse(deco) or "<decorator>")
        return decos

    # ------------------------------------------------------------------
    # AST visitor overrides
    # ------------------------------------------------------------------
    def visit_Module(self, node: ast.Module) -> None:
        self._module_doc = ast.get_docstring(node)
        for child in node.body:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        try:
                            val = ast.literal_eval(child.value)
                            if isinstance(val, (list, tuple)):
                                self._exports = [str(v) for v in val]
                        except Exception:
                            pass
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    asname = f" as {alias.asname}" if alias.asname else ""
                    self._imports.append(f"import {alias.name}{asname}")
            elif isinstance(child, ast.ImportFrom):
                module = child.module or ""
                names = ", ".join([
                    alias.name + (f" as {alias.asname}" if alias.asname else "")
                    for alias in child.names
                ])
                dots = "." * child.level if getattr(child, "level", 0) else ""
                self._from_imports.append(f"from {dots}{module} import {names}")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_unit = self._build_class_unit(node)
        self._register_unit(class_unit, parent=self._current_parent())
        self._qualname_stack.append(node.name)
        self._class_stack.append(class_unit)

        for item in node.body:
            if isinstance(item, ast.AnnAssign):
                field_unit = self._build_field_unit(item)
                if field_unit:
                    self._register_unit(field_unit, parent=class_unit)
                    class_unit.fields.append(field_unit)
            elif isinstance(item, ast.Assign):
                for field_unit in self._build_field_units_from_assign(item):
                    self._register_unit(field_unit, parent=class_unit)
                    class_unit.fields.append(field_unit)
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_unit, property_unit = self._build_function_unit(item, in_class=True)
                if property_unit is not None:
                    self._register_unit(property_unit, parent=class_unit)
                    class_unit.properties.append(property_unit)
                if fn_unit is not None:
                    self._register_unit(fn_unit, parent=class_unit)
                    if property_unit is None or "property_accessor" not in fn_unit.roles:
                        class_unit.methods.append(fn_unit)
            elif isinstance(item, ast.ClassDef):
                # Nested class handled via visitor
                self.visit_ClassDef(item)
            else:
                self.generic_visit(item)

        self._qualname_stack.pop()
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._class_stack:
            return
        fn_unit, _ = self._build_function_unit(node, in_class=False)
        if fn_unit is not None:
            self._register_unit(fn_unit, parent=self._current_parent())
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self._class_stack:
            return
        fn_unit, _ = self._build_function_unit(node, in_class=False)
        if fn_unit is not None:
            self._register_unit(fn_unit, parent=self._current_parent())
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _build_class_unit(self, node: ast.ClassDef) -> ClassUnit:
        bases = [self._segment(base) or self._safe_unparse(base) or "<base>" for base in node.bases]
        decorators = self._decorator_texts(node)
        metaclass = None
        for kw in node.keywords:
            if kw.arg == "metaclass":
                metaclass = self._segment(kw.value) or self._safe_unparse(kw.value)
        abstract = any(
            (b or "").endswith("ABC") or ("abc.ABC" in (b or "")) for b in bases
        ) or any("abstract" in (d or "") for d in decorators)
        slots_enabled = any(
            isinstance(stmt, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__slots__" for target in stmt.targets)
            for stmt in node.body
        )
        qualname = self._qualname(node.name)
        class_unit = ClassUnit(
            kind="class",
            name=node.name,
            qualname=qualname,
            module=self.module,
            filepath=self.filepath,
            line_range=self._line_range(node),
            docstring=ast.get_docstring(node),
            visibility=self._visibility(node.name),
            exported=self._exported(node.name),
            source_snippet=self._segment(node),
            bases=bases,
            decorators=decorators,
            metaclass=metaclass,
            abstract=abstract,
            slots_enabled=slots_enabled,
            parent_qualname=self._current_parent_qualname(),
        )
        header_parts: List[str] = []
        if class_unit.bases:
            header_parts.append(", ".join(class_unit.bases))
        if class_unit.metaclass:
            header_parts.append(f"metaclass={class_unit.metaclass}")
        inside = ", ".join(header_parts)
        header = f"{self.module}.{class_unit.qualname}:{class_unit.name}({inside})" if inside else f"{self.module}.{class_unit.qualname}:{class_unit.name}"
        class_unit.id_sha = self._sha256(header)
        return class_unit

    def _build_field_unit(self, node: ast.AnnAssign) -> Optional[FieldUnit]:
        if not isinstance(node.target, ast.Name):
            return None
        name = node.target.id
        qualname = self._qualname(name)
        annotation = self._annotation_text(node.annotation)
        default = self._segment(node.value) or self._safe_unparse(node.value)
        field_unit = FieldUnit(
            kind="field",
            name=name,
            qualname=qualname,
            module=self.module,
            filepath=self.filepath,
            line_range=self._line_range(node),
            docstring=None,
            visibility=self._visibility(name),
            exported=self._exported(name),
            source_snippet=self._segment(node),
            annotation=annotation,
            default=default,
            is_classvar=True,
            parent_qualname=self._current_parent_qualname(),
        )
        return field_unit

    def _build_field_units_from_assign(self, node: ast.Assign) -> List[FieldUnit]:
        units: List[FieldUnit] = []
        default_text = self._segment(node.value) or self._safe_unparse(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                units.append(
                    FieldUnit(
                        kind="field",
                        name=name,
                        qualname=self._qualname(name),
                        module=self.module,
                        filepath=self.filepath,
                        line_range=self._line_range(node),
                        docstring=None,
                        visibility=self._visibility(name),
                        exported=self._exported(name),
                        source_snippet=self._segment(node),
                        annotation=None,
                        default=default_text,
                        is_classvar=True,
                        parent_qualname=self._current_parent_qualname(),
                    )
                )
        return units

    def _build_function_unit(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        *,
        in_class: bool,
    ) -> Tuple[Optional[FunctionUnit], Optional[PropertyUnit]]:
        name = node.name
        qualname = self._qualname(name)
        decorators = self._decorator_texts(node)

        params: List[Parameter] = []
        args = node.args
        for arg in getattr(args, "posonlyargs", []):
            params.append(Parameter(name=arg.arg, kind="posonly", annotation=self._annotation_text(arg.annotation)))
        for arg in args.args:
            params.append(Parameter(name=arg.arg, kind="poskw", annotation=self._annotation_text(arg.annotation)))
        if args.vararg is not None:
            params.append(Parameter(name=args.vararg.arg, kind="vararg", annotation=self._annotation_text(args.vararg.annotation)))
        for arg in args.kwonlyargs:
            params.append(Parameter(name=arg.arg, kind="kwonly", annotation=self._annotation_text(arg.annotation)))
        if args.kwarg is not None:
            params.append(Parameter(name=args.kwarg.arg, kind="varkw", annotation=self._annotation_text(args.kwarg.annotation)))

        defaults = list(args.defaults)
        if defaults:
            positional = [p for p in params if p.kind in ("posonly", "poskw")]
            for param, default_node in zip(positional[-len(defaults):], defaults):
                param.default = self._segment(default_node) or self._safe_unparse(default_node)
        for param, default_node in zip([p for p in params if p.kind == "kwonly"], args.kw_defaults):
            if default_node is not None:
                param.default = self._segment(default_node) or self._safe_unparse(default_node)

        return_ann = self._annotation_text(node.returns)

        roles: List[str] = []
        decorator_texts = [d.strip() for d in decorators]
        is_staticmethod = any("staticmethod" in d for d in decorator_texts)
        is_classmethod = any("classmethod" in d for d in decorator_texts)
        is_property = any(d.endswith("property") or d == "property" for d in decorator_texts)
        if is_property:
            roles.append("property_accessor")
            roles.append("property_getter")
        if name == "__init__" and in_class:
            roles.append("constructor")

        fn_unit = FunctionUnit(
            kind="async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
            name=name,
            qualname=qualname,
            module=self.module,
            filepath=self.filepath,
            line_range=self._line_range(node),
            docstring=ast.get_docstring(node),
            visibility=self._visibility(name),
            exported=self._exported(name),
            source_snippet=self._segment(node),
            parameters=params,
            return_annotation=return_ann,
            decorators=decorators,
            roles=roles,
            async_def=isinstance(node, ast.AsyncFunctionDef),
            is_method=in_class,
            is_staticmethod=is_staticmethod,
            is_classmethod=is_classmethod,
            parent_qualname=self._current_parent_qualname(),
        )
        signature = fn_unit.signature_string()
        header = f"{self.module}.{fn_unit.qualname}:{signature}"
        fn_unit.id_sha = self._sha256(header)

        property_unit: Optional[PropertyUnit] = None
        if in_class and is_property:
            property_unit = PropertyUnit(
                kind="property",
                name=name,
                qualname=qualname,
                module=self.module,
                filepath=self.filepath,
                line_range=self._line_range(node),
                docstring=fn_unit.docstring,
                visibility=self._visibility(name),
                exported=self._exported(name),
                source_snippet=fn_unit.source_snippet,
                type_annotation=return_ann,
                accessor_roles=["fget"],
                parent_qualname=self._current_parent_qualname(),
            )
            property_unit.id_sha = self._sha256(f"{self.module}.{property_unit.qualname}:property")
        return fn_unit, property_unit

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def _current_parent(self) -> Optional[SemanticUnit]:
        return self._class_stack[-1] if self._class_stack else None

    def _current_parent_qualname(self) -> Optional[str]:
        parent = self._current_parent()
        return parent.qualname if parent else None

    def _register_unit(self, unit: SemanticUnit, parent: Optional[SemanticUnit]) -> None:
        if isinstance(unit, ModuleUnit):
            self._units.insert(0, unit)
        else:
            self._units.append(unit)
        if parent is None:
            if not isinstance(unit, ModuleUnit):
                self._module_children.append(unit)
        else:
            self._edges.append(Edge(type="contains", src_id=parent.id_sha, dst_id=unit.id_sha))
            if isinstance(parent, ClassUnit) and isinstance(unit, ClassUnit):
                parent.nested_classes.append(unit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def parse(self, *, tolerate_errors: bool = False) -> ParseArtifacts:
        try:
            tree = ast.parse(self.src)
        except SyntaxError as e:
            if not tolerate_errors:
                raise
            # Build a minimal module unit so callers still get structure + error report
            end_lineno = len(self.src.splitlines()) or 1
            module_unit = ModuleUnit(
                kind="module",
                name=self.module.split(".")[-1],
                qualname=self.module,
                module=self.module,
                filepath=self.filepath,
                line_range=(1, max(end_lineno, 1)),
                docstring=None,
                visibility="public",
                exported=None,
                source_snippet=None,
                imports=[],
                from_imports=[],
                exports=None,
                top_level=[],
            )
            module_unit.id_sha = hashlib.sha256(f"module:{self.module}:".encode("utf-8")).hexdigest()
            err = ParseError(
                message=str(e),
                lineno=getattr(e, "lineno", 0) or 0,
                col_offset=getattr(e, "offset", 0) or 0,
                text=getattr(e, "text", None),
            )
            return ParseArtifacts(module=module_unit, units=[], edges=[], errors=[err])

        self.visit(tree)
        module_unit = self._build_module_unit(tree)
        self._register_unit(module_unit, parent=None)
        for child in self._module_children:
            self._edges.append(Edge(type="contains", src_id=module_unit.id_sha, dst_id=child.id_sha))
        return ParseArtifacts(module=module_unit, units=self._units, edges=self._edges, errors=[])

    def _build_module_unit(self, tree: ast.AST) -> ModuleUnit:
        end_lineno = getattr(tree, "end_lineno", None)
        if end_lineno is None:
            end_lineno = len(self.src.splitlines()) or 1
        module_unit = ModuleUnit(
            kind="module",
            name=self.module.split(".")[-1],
            qualname=self.module,
            module=self.module,
            filepath=self.filepath,
            line_range=(1, max(end_lineno, 1)),
            docstring=self._module_doc,
            visibility="public",
            exported=None,
            source_snippet=None,
            imports=self._imports,
            from_imports=self._from_imports,
            exports=self._exports,
            top_level=[child.qualname for child in self._module_children],
        )
        module_unit.id_sha = self._sha256(
            f"module:{self.module}:{','.join(module_unit.top_level)}"
        )
        return module_unit
