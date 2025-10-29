"""
Python AST-based symbol schemas + extractor for documentation pipelines.

- Discriminated schemas: ModuleSymbol, ClassSymbol, FunctionSymbol, AttributeSymbol
- Rich metadata (FQN, bases, decorators, parameters, annotations, docstrings, visibility)
- Deterministic IDs via SHA256 of canonical headers
- Extracts from source text (keeps exact annotation/default text when possible)
- No external dependencies (pure stdlib)

Usage (example):
    text = Path("your_module.py").read_text(encoding="utf-8")
    extractor = PySymbolExtractor(src=text, filepath="your_module.py", module="your_module")
    result = extractor.extract()
    print(result.module)
    for cls in result.classes:
        print(cls.qualname, cls.bases)
        for m in cls.methods:
            print(" ", m.signature_string)

This file is designed for integration with a LangGraph doc-gen flow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, List, Tuple, Dict, Union, Literal
import ast
import hashlib

# ------------------------------------------------------------
# Enums & small helpers
# ------------------------------------------------------------
ParamKind = Literal[
    "posonly", "poskw", "vararg", "kwonly", "varkw"
]

SymbolKind = Literal[
    "module", "class", "function", "async_function", "attribute"
]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _safe_unparse(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _segment(src: Optional[str], node: Optional[ast.AST]) -> Optional[str]:
    if src is None or node is None:
        return None
    try:
        seg = ast.get_source_segment(src, node)
        return seg.strip() if seg is not None else None
    except Exception:
        return None


def _ann_text(ann: Optional[ast.AST], src: Optional[str]) -> Optional[str]:
    # Prefer exact source segment, fall back to unparse
    return _segment(src, ann) or _safe_unparse(ann)


# ------------------------------------------------------------
# Schemas (base + discriminated subtypes)
# ------------------------------------------------------------
@dataclass
class SymbolBase:
    kind: SymbolKind
    name: str
    qualname: str
    module: str
    filepath: Optional[str]
    line_range: Tuple[int, int]
    docstring: Optional[str] = None
    visibility: Literal["public", "private"] = "public"
    exported: Optional[bool] = None  # based on __all__ if present
    source_snippet: Optional[str] = None
    id_sha: Optional[str] = None  # stable content hash


@dataclass
class AttributeSymbol(SymbolBase):
    annotation: Optional[str] = None
    default: Optional[str] = None
    is_classvar: bool = False
    is_property: bool = False


@dataclass
class Parameter:
    name: str
    kind: ParamKind
    annotation: Optional[str] = None
    default: Optional[str] = None


@dataclass
class FunctionSymbol(SymbolBase):
    parameters: List[Parameter] = field(default_factory=list)
    return_annotation: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    raises: List[str] = field(default_factory=list)  # heuristic/optional
    async_def: bool = False
    is_method: bool = False
    is_staticmethod: bool = False
    is_classmethod: bool = False

    @property
    def signature_string(self) -> str:
        parts: List[str] = []
        for p in self.parameters:
            frag = p.name
            if p.kind == "vararg":
                frag = "*" + frag
            elif p.kind == "varkw":
                frag = "**" + frag
            if p.annotation:
                frag += f": {p.annotation}"
            if p.default:
                frag += f"={p.default}"
            parts.append(frag)
        ret = f" -> {self.return_annotation}" if self.return_annotation else ""
        return f"{self.name}({', '.join(parts)}){ret}"


@dataclass
class ClassSymbol(SymbolBase):
    bases: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    metaclass: Optional[str] = None
    type_params: List[str] = field(default_factory=list)  # PEP 695 or TypeVars
    attributes: List[AttributeSymbol] = field(default_factory=list)
    methods: List[FunctionSymbol] = field(default_factory=list)
    abstract: bool = False
    slots_enabled: bool = False

    @property
    def header_string(self) -> str:
        parts = []
        if self.bases:
            parts.append(", ".join(self.bases))
        if self.metaclass:
            parts.append(f"metaclass={self.metaclass}")
        inside = ", ".join(parts)
        return f"{self.name}({inside})" if inside else self.name


@dataclass
class ModuleSymbol(SymbolBase):
    imports: List[str] = field(default_factory=list)
    from_imports: List[str] = field(default_factory=list)
    exports: Optional[List[str]] = None  # __all__
    top_level: List[str] = field(default_factory=list)  # qualnames of children


@dataclass
class ExtractResult:
    module: ModuleSymbol
    classes: List[ClassSymbol]
    functions: List[FunctionSymbol]
    attributes: List[AttributeSymbol]


# ------------------------------------------------------------
# Extractor
# ------------------------------------------------------------
class PySymbolExtractor(ast.NodeVisitor):
    """Extracts symbols from Python source into rich schemas."""

    def __init__(self, src: str, filepath: Optional[str], module: str):
        self.src = src
        self.filepath = filepath
        self.module = module
        self.class_stack: List[str] = []  # for qualname building
        self.exports: Optional[List[str]] = None

        self.classes: List[ClassSymbol] = []
        self.functions: List[FunctionSymbol] = []
        self.attributes: List[AttributeSymbol] = []

        self._module_doc: Optional[str] = None
        self._imports: List[str] = []
        self._from_imports: List[str] = []
        self._top_level: List[str] = []

    # ---------- Helpers ----------
    def _vis(self, name: str) -> Literal["public", "private"]:
        return "private" if name.startswith("_") else "public"

    def _qualname(self, name: str) -> str:
        return ".".join([*self.class_stack, name]) if self.class_stack else name

    def _exported(self, name: str) -> Optional[bool]:
        if self.exports is None:
            return None
        return name in self.exports

    def _line_range(self, node: ast.AST) -> Tuple[int, int]:
        return (
            getattr(node, "lineno", 1),
            getattr(node, "end_lineno", getattr(node, "lineno", 1)),
        )

    def _decorator_texts(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> List[str]:
        decos: List[str] = []
        for d in node.decorator_list:
            decos.append(_segment(self.src, d) or _safe_unparse(d) or "<decorator>")
        return decos

    def _id_for_function(self, f: FunctionSymbol) -> str:
        header = f"{self.module}.{f.qualname}:{f.signature_string}"
        return _sha256(header)

    def _id_for_class(self, c: ClassSymbol) -> str:
        header = f"{self.module}.{c.qualname}:{c.header_string}"
        return _sha256(header)

    # ---------- Module ----------
    def visit_Module(self, node: ast.Module):
        self._module_doc = ast.get_docstring(node)
        # gather __all__ and imports
        for n in node.body:
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id == "__all__":
                        try:
                            val = ast.literal_eval(n.value)
                            if isinstance(val, (list, tuple)):
                                self.exports = [str(x) for x in val]
                        except Exception:
                            pass
            elif isinstance(n, ast.Import):
                # import a, b as c
                for a in n.names:
                    asname = f" as {a.asname}" if a.asname else ""
                    self._imports.append(f"import {a.name}{asname}")
            elif isinstance(n, ast.ImportFrom):
                mod = n.module or ""
                names = ", ".join([a.name + (f" as {a.asname}" if a.asname else "") for a in n.names])
                dots = "." * n.level if getattr(n, "level", 0) else ""
                self._from_imports.append(f"from {dots}{mod} import {names}")

        self.generic_visit(node)

    # ---------- Classes ----------
    def visit_ClassDef(self, node: ast.ClassDef):
        name = node.name
        qualname = self._qualname(name)
        lr = self._line_range(node)
        bases = [(_segment(self.src, b) or _safe_unparse(b) or "<base>") for b in node.bases]
        decos = self._decorator_texts(node)
        meta = None
        for kw in node.keywords:
            if kw.arg == "metaclass":
                meta = _segment(self.src, kw.value) or _safe_unparse(kw.value)

        # Heuristics
        abstract = any("abc.ABC" in (b or "") or "ABC" == (b or "") for b in bases) or any(
            d.endswith("abstractmethod") or d.endswith("@abstractmethod") for d in decos
        )
        slots_enabled = any(isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__slots__" for t in n.targets
        ) for n in node.body)

        cls = ClassSymbol(
            kind="class",
            name=name,
            qualname=qualname,
            module=self.module,
            filepath=self.filepath,
            line_range=lr,
            docstring=ast.get_docstring(node),
            visibility=self._vis(name),
            exported=self._exported(name),
            source_snippet=_segment(self.src, node),
            bases=bases,
            decorators=decos,
            metaclass=meta,
            abstract=abstract,
            slots_enabled=slots_enabled,
        )

        # Enter class scope
        self.class_stack.append(name)

        # Collect class-level attributes & methods
        for n in node.body:
            if isinstance(n, ast.AnnAssign):
                target = n.target
                if isinstance(target, ast.Name):
                    ann = _ann_text(n.annotation, self.src)
                    default = _segment(self.src, n.value) or _safe_unparse(n.value)
                    attr = AttributeSymbol(
                        kind="attribute",
                        name=target.id,
                        qualname=self._qualname(target.id),
                        module=self.module,
                        filepath=self.filepath,
                        line_range=self._line_range(n),
                        docstring=None,
                        visibility=self._vis(target.id),
                        exported=self._exported(target.id),
                        annotation=ann,
                        default=default,
                        is_classvar=True,
                        is_property=False,
                        source_snippet=_segment(self.src, n),
                    )
                    cls.attributes.append(attr)
                    self.attributes.append(attr)
            elif isinstance(n, ast.Assign):
                # unannotated class attribute
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        default = _segment(self.src, n.value) or _safe_unparse(n.value)
                        attr = AttributeSymbol(
                            kind="attribute",
                            name=t.id,
                            qualname=self._qualname(t.id),
                            module=self.module,
                            filepath=self.filepath,
                            line_range=self._line_range(n),
                            docstring=None,
                            visibility=self._vis(t.id),
                            exported=self._exported(t.id),
                            annotation=None,
                            default=default,
                            is_classvar=True,
                            is_property=False,
                            source_snippet=_segment(self.src, n),
                        )
                        cls.attributes.append(attr)
                        self.attributes.append(attr)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = self._function_from_node(n, in_class=True)
                # Convert @property into attribute symbol as well
                if any(d.strip().endswith("property") or d.strip() == "property" for d in fn.decorators):
                    prop = AttributeSymbol(
                        kind="attribute",
                        name=fn.name,
                        qualname=self._qualname(fn.name),
                        module=self.module,
                        filepath=self.filepath,
                        line_range=fn.line_range,
                        docstring=fn.docstring,
                        visibility=self._vis(fn.name),
                        exported=self._exported(fn.name),
                        annotation=fn.return_annotation,
                        default=None,
                        is_classvar=False,
                        is_property=True,
                        source_snippet=fn.source_snippet,
                    )
                    cls.attributes.append(prop)
                    self.attributes.append(prop)
                else:
                    cls.methods.append(fn)
                    self.functions.append(fn)

        # Exit class scope
        self.class_stack.pop()

        cls.id_sha = self._id_for_class(cls)
        self.classes.append(cls)
        self._top_level.append(cls.qualname) if not self.class_stack else None

    # ---------- Functions ----------
    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not self.class_stack:  # top-level only; methods handled in ClassDef
            fn = self._function_from_node(node, in_class=False)
            fn.id_sha = self._id_for_function(fn)
            self.functions.append(fn)
            self._top_level.append(fn.qualname)
        # continue to allow nested defs if needed
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if not self.class_stack:
            fn = self._function_from_node(node, in_class=False)
            fn.id_sha = self._id_for_function(fn)
            self.functions.append(fn)
            self._top_level.append(fn.qualname)
        self.generic_visit(node)

    # ---------- Build function symbol ----------
    def _function_from_node(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], *, in_class: bool) -> FunctionSymbol:
        name = node.name
        qualname = self._qualname(name)
        lr = self._line_range(node)
        decos = self._decorator_texts(node)

        is_static = any("staticmethod" in (d or "") for d in decos)
        is_classm = any("classmethod" in (d or "") for d in decos)

        params: List[Parameter] = []
        A = node.args

        # posonly (Py 3.8+)
        for a in getattr(A, "posonlyargs", []):
            params.append(Parameter(name=a.arg, kind="posonly", annotation=_ann_text(a.annotation, self.src)))
        # pos/kw
        for a in A.args:
            params.append(Parameter(name=a.arg, kind="poskw", annotation=_ann_text(a.annotation, self.src)))
        # vararg *args
        if A.vararg is not None:
            params.append(Parameter(name=A.vararg.arg, kind="vararg", annotation=_ann_text(A.vararg.annotation, self.src)))
        # kwonly
        for a in A.kwonlyargs:
            params.append(Parameter(name=a.arg, kind="kwonly", annotation=_ann_text(a.annotation, self.src)))
        # varkw **kwargs
        if A.kwarg is not None:
            params.append(Parameter(name=A.kwarg.arg, kind="varkw", annotation=_ann_text(A.kwarg.annotation, self.src)))

        # defaults: align from the right with positional args
        defaults = list(A.defaults)
        if defaults:
            # only for the last N positional params (posonly + poskw)
            pos_params = [p for p in params if p.kind in ("posonly", "poskw")]
            for p, d in zip(pos_params[-len(defaults):], defaults):
                p.default = _segment(self.src, d) or _safe_unparse(d)
        # kw_defaults align 1:1 with kwonlyargs
        for p, d in zip([p for p in params if p.kind == "kwonly"], A.kw_defaults):
            if d is not None:
                p.default = _segment(self.src, d) or _safe_unparse(d)

        ret_ann = _ann_text(node.returns, self.src)

        fn = FunctionSymbol(
            kind="async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
            name=name,
            qualname=qualname,
            module=self.module,
            filepath=self.filepath,
            line_range=lr,
            docstring=ast.get_docstring(node),
            visibility=self._vis(name),
            exported=self._exported(name),
            source_snippet=_segment(self.src, node),
            parameters=params,
            return_annotation=ret_ann,
            decorators=decos,
            async_def=isinstance(node, ast.AsyncFunctionDef),
            is_method=in_class,
            is_staticmethod=is_static,
            is_classmethod=is_classm,
        )
        return fn

    # ---------- Finalization ----------
    def extract(self) -> ExtractResult:
        tree = ast.parse(self.src)
        self.visit(tree)
        # Build module symbol at the end
        mod = ModuleSymbol(
            kind="module",
            name=self.module.split(".")[-1],
            qualname=self.module,
            module=self.module,
            filepath=self.filepath,
            line_range=(1, max(getattr(tree, "end_lineno", 1), 1)),
            docstring=self._module_doc,
            visibility="public",
            exported=None,
            source_snippet=None,
            imports=self._imports,
            from_imports=self._from_imports,
            exports=self.exports,
            top_level=self._top_level,
        )
        mod.id_sha = _sha256(f"module:{self.module}:{','.join(mod.top_level)}")
        return ExtractResult(module=mod, classes=self.classes, functions=self.functions, attributes=self.attributes)

# ------------------------------------------------------------
# CLI / main for quick testing
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json
    from dataclasses import asdict, is_dataclass
    from pathlib import Path

    def _to_dict(x):
        if is_dataclass(x) and not isinstance(x, type):
            return {k: _to_dict(v) for k, v in asdict(x).items()}
        if isinstance(x, (list, tuple)):
            return [_to_dict(i) for i in x]
        if isinstance(x, dict):
            return {k: _to_dict(v) for k, v in x.items()}
        return x

    ap = argparse.ArgumentParser(description="AST Doc Symbol Extractor (quick test)")
    ap.add_argument("path", help="Python source file to parse")
    ap.add_argument("--module", default=None, help="Module name (defaults to stem of path)")
    ap.add_argument("--json", action="store_true", help="Print full JSON output")
    args = ap.parse_args()

    p = Path(args.path)
    text = p.read_text(encoding="utf-8", errors="replace")
    module = args.module or p.stem

    extractor = PySymbolExtractor(src=text, filepath=str(p), module=module)
    res = extractor.extract()

    if args.json:
        print(json.dumps({
            "module": _to_dict(res.module),
            "classes": _to_dict(res.classes),
            "functions": _to_dict(res.functions),
            "attributes": _to_dict(res.attributes),
        }, ensure_ascii=False, indent=2))
    else:
        print(f"Module: {res.module.qualname}")
        if res.classes:
            print("\nClasses:")
            for c in res.classes:
                print(f"  - {c.qualname}: {c.header_string}  [lines {c.line_range[0]}-{c.line_range[1]}]")
                if c.attributes:
                    for a in c.attributes:
                        print(f"      • attr {a.qualname}: ann={a.annotation!r} default={a.default!r}")
                if c.methods:
                    for m in c.methods:
                        kind = "async def" if m.async_def else "def"
                        role = "static " if m.is_staticmethod else ("class " if m.is_classmethod else "")
                        print(f"      • {role}{kind} {m.qualname} :: {m.signature_string}")
        if res.functions:
            print("\nTop-level functions:")
            for f in res.functions:
                kind = "async def" if f.async_def else "def"
                print(f"  - {kind} {f.qualname} :: {f.signature_string}  [lines {f.line_range[0]}-{f.line_range[1]}]")
        if res.attributes:
            print("\nTop-level attributes:")
            for a in res.attributes:
                # filter only those not nested (qualname has no dot)
                if "." not in a.qualname:
                    print(f"  - {a.qualname}: ann={a.annotation!r} default={a.default!r}")

