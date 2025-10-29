"""
Java 17+ Symbol Extractor in Python using tree-sitter-java.

- Matches the same schema used by the Python AST extractor:
  SymbolBase → ModuleSymbol | ClassSymbol | FunctionSymbol | AttributeSymbol
- Supports: package, classes/interfaces/enums/records (incl. sealed/permits),
  extends/implements, fields, methods & constructors, annotations, modifiers, line ranges.
- Emits ExtractResult so you can re-use downstream code.

Install deps:
    pip install tree-sitter tree-sitter-java

Usage:
    python java17_extractor.py --src ./src/main/java --json out.json

Notes:
- Tree-sitter gives precise ranges and raw text segments. We keep textual fidelity.
- Javadoc is not parsed here (tree-sitter doesn’t retain comments by default);
  if you need it, pair with a tokenizer or tree-sitter with comment capture.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Optional, List, Tuple, Dict, Literal, Union, Any
from pathlib import Path
import argparse
import json
import hashlib
import logging

logger = logging.getLogger(__name__)

from tree_sitter import Parser
try:  # tree_sitter >=0.22 prefers assigning tree_sitter.Language instances
    from tree_sitter import Language  # type: ignore
except ImportError:  # pragma: no cover - older bindings do not expose Language
    Language = None  # type: ignore
import tree_sitter_java as ts_java

# ------------------------------------------------------------
# Shared schema (aligned with Python AST extractor)
# ------------------------------------------------------------
ParamKind = Literal["posonly", "poskw", "vararg", "kwonly", "varkw"]  # kept for shape parity
SymbolKind = Literal["module", "class", "function", "async_function", "attribute"]


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


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
    exported: Optional[bool] = None
    source_snippet: Optional[str] = None
    id_sha: Optional[str] = None


@dataclass
class AttributeSymbol(SymbolBase):
    annotation: Optional[str] = None  # field type
    default: Optional[str] = None     # initializer
    is_classvar: bool = True
    is_property: bool = False


@dataclass
class Parameter:
    name: str
    kind: ParamKind  # for Java, use "poskw" and set vararg=True via name decoration
    annotation: Optional[str] = None  # param type
    default: Optional[str] = None


@dataclass
class FunctionSymbol(SymbolBase):
    parameters: List[Parameter] = field(default_factory=list)
    return_annotation: Optional[str] = None  # return type or None for ctor
    decorators: List[str] = field(default_factory=list)  # annotations
    raises: List[str] = field(default_factory=list)
    async_def: bool = False
    is_method: bool = True
    is_staticmethod: bool = False
    is_classmethod: bool = False

    @property
    def signature_string(self) -> str:
        parts = []
        for p in self.parameters:
            frag = p.name
            if p.annotation:
                frag = f"{p.annotation} {frag}"
            if p.default:
                frag += f"={p.default}"
            parts.append(frag)
        ret = f" -> {self.return_annotation}" if self.return_annotation else ""
        return f"{self.name}({', '.join(parts)}){ret}"


@dataclass
class ClassSymbol(SymbolBase):
    bases: List[str] = field(default_factory=list)          # extends
    decorators: List[str] = field(default_factory=list)     # annotations
    metaclass: Optional[str] = None                         # N/A for Java
    type_params: List[str] = field(default_factory=list)
    attributes: List[AttributeSymbol] = field(default_factory=list)
    methods: List[FunctionSymbol] = field(default_factory=list)
    abstract: bool = False
    slots_enabled: bool = False                             # N/A for Java
    implements: List[str] = field(default_factory=list)
    record_components: List[AttributeSymbol] = field(default_factory=list)
    sealed: bool = False
    permits: List[str] = field(default_factory=list)

    @property
    def header_string(self) -> str:
        parts = []
        if self.bases:
            parts.append("extends " + ", ".join(self.bases))
        if self.implements:
            parts.append("implements " + ", ".join(self.implements))
        inside = (" " + " ".join(parts)) if parts else ""
        return f"{self.name}{inside}"


@dataclass
class ModuleSymbol(SymbolBase):
    imports: List[str] = field(default_factory=list)
    from_imports: List[str] = field(default_factory=list)
    exports: Optional[List[str]] = None
    top_level: List[str] = field(default_factory=list)


@dataclass
class ExtractResult:
    module: ModuleSymbol
    classes: List[ClassSymbol]
    functions: List[FunctionSymbol]
    attributes: List[AttributeSymbol]


# ------------------------------------------------------------
# Tree-sitter helpers
# ------------------------------------------------------------
def _load_language():
    """Return a tree-sitter language compatible with both old/new bindings."""
    raw = ts_java.language()
    if 'Language' in globals() and Language is not None:
        try:
            return Language(raw)  # tree_sitter >=0.22 expects Language objects
        except (TypeError, ValueError):
            pass
    return raw


LANG = _load_language()


def _init_parser(lang) -> Parser:
    """Initialize a tree-sitter Parser robustly across API variants.

    Strategy:
    1) Prefer Parser(); set_language(lang)
    2) Try assigning parser.language = lang (new API)
    3) If that fails, try Parser(lang)
    4) Validate by parsing a tiny snippet; otherwise raise a helpful error
    """
    # 1) Preferred: explicit set_language
    errors = []

    def _validate(p: Parser) -> bool:
        try:
            t = p.parse(b"class A{}")
            return t is not None and getattr(t, "root_node", None) is not None
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"parse: {exc}")
            return False

    # 1) Preferred: explicit set_language when available (older API)
    parser = Parser()
    if hasattr(parser, "set_language"):
        try:
            parser.set_language(lang)  # type: ignore[attr-defined]
            if _validate(parser):
                return parser
        except Exception as exc:
            errors.append(f"set_language: {exc}")

    # 2) Newer API: assign via property
    try:
        parser = Parser()
        parser.language = lang  # type: ignore[attr-defined]
        if _validate(parser):
            return parser
    except Exception as exc:
        errors.append(f"language property: {exc}")

    # 3) Fallback: constructor accepting language (legacy bindings)
    try:
        parser = Parser(lang)  # type: ignore[call-arg]
        if _validate(parser):
            return parser
    except Exception as exc:
        errors.append(f"ctor: {exc}")

    raise RuntimeError(
        "Failed to initialize tree-sitter parser: " + "; ".join(errors) if errors else "unknown error"
    )


PARSER = _init_parser(LANG)

# Tree-sitter node field utils
class TSNode:
    def __init__(self, node, src: bytes):
        self.node = node
        self.src = src

    @property
    def type(self):
        return self.node.type

    @property
    def text(self) -> str:
        s, e = self.node.start_byte, self.node.end_byte
        return self.src[s:e].decode("utf-8", errors="replace")

    @property
    def line_range(self) -> Tuple[int, int]:
        (sr, _), (er, _) = self.node.start_point, self.node.end_point
        # Convert 0-based rows to 1-based lines
        return (sr + 1, er + 1 if er >= sr else sr + 1)

    def children(self):
        for i in range(self.node.child_count):
            yield TSNode(self.node.child(i), self.src)

    def walk(self):
        # DFS
        stack = [self.node]
        while stack:
            n = stack.pop()
            yield TSNode(n, self.src)
            for i in reversed(range(n.child_count)):
                stack.append(n.child(i))

    def named_children(self):
        for i in range(self.node.named_child_count):
            yield TSNode(self.node.named_child(i), self.src)

    def find_children(self, t: str):
        # immediate named children only
        return [c for c in self.named_children() if c.type == t]

    def find_descendants(self, t: str):
        # recursive search anywhere under this node
        return [TSNode(n.node, self.src) for n in self.walk() if n.type == t]


# ------------------------------------------------------------
# Core extraction
# ------------------------------------------------------------
MODIFIER_KEYWORDS = {
    "public",
    "private",
    "protected",
    "static",
    "abstract",
    "final",
    "sealed",
    "non-sealed",
    "default",
    "strictfp",
}
TYPE_NODE_NAMES = {
    "type",
    "integral_type",
    "floating_point_type",
    "boolean_type",
    "void_type",
    "array_type",
    "generic_type",
    "scoped_type_identifier",
    "type_identifier",
    "class_or_interface_type",
    "unannotated_type",
    "unann_type",
    "primitive_type",
}

CLASS_DECL_TYPES = {
    "class_declaration",
    "interface_declaration",
    "enum_declaration",
    "record_declaration",
}

BODY_NODE_TYPES = {
    "class_body",
    "interface_body",
    "enum_body",
    "record_body",
}

BODY_WRAPPER_TYPES = {
    "class_body_declaration",
    "interface_body_declaration",
    "enum_body_declaration",
    "record_body_declaration",
}


def _visibility(mods: List[str]) -> str:
    if "public" in mods:
        return "public"
    if "private" in mods:
        return "private"
    return "public" if not mods else "public" if "protected" in mods else "public" if "sealed" in mods else "public"


def _mods(node: TSNode) -> List[str]:
    mods = []
    for c in node.find_children("modifiers"):
        tokens: List[str] = []
        named = list(c.named_children())
        if named:
            for m in named:
                t = m.type
                if t in ("marker_annotation", "annotation"):
                    continue
                text = m.text.strip()
                if text:
                    tokens.append(text)
        else:
            tokens = c.text.strip().split()
        for tok in tokens:
            if tok in MODIFIER_KEYWORDS:
                mods.append(tok)
    return mods


def _annotations(node: TSNode) -> List[str]:
    out = []
    for c in node.find_children("modifiers"):
        for m in c.named_children():
            if m.type in ("marker_annotation", "annotation"):
                out.append(m.text.strip())
    return out


def _name_from(node: TSNode) -> Optional[str]:
    for ch in node.named_children():
        if ch.type == "identifier":
            return ch.text
    return None


def _package_name(root: TSNode) -> str:
    for child in root.named_children():
        if child.type == "package_declaration":
            # form: package a.b.c ;
            parts = [c.text for c in child.named_children() if c.type in ("scoped_identifier", "identifier")]
            if parts:
                return parts[0]
    return ""


def _find_lists(node: TSNode, type_name: str) -> List[TSNode]:
    return [c for c in node.walk() if c.type == type_name]


def parse_java(src: str) -> TSNode:
    b = src.encode("utf-8")
    tree = PARSER.parse(b)
    return TSNode(tree.root_node, b)


def extract_from_file(path: Path) -> ExtractResult:
    text = path.read_text(encoding="utf-8", errors="replace")
    root = parse_java(text)
    pkg = _package_name(root)
    module_name = pkg if pkg else "<default>"

    classes: List[ClassSymbol] = []
    functions: List[FunctionSymbol] = []  # Java rarely has top-level functions
    attributes: List[AttributeSymbol] = []

    # Collect type declarations
    for n in root.named_children():
        if n.type in CLASS_DECL_TYPES:
            classes.extend(build_class_symbols(n, pkg, path, text))

    mod = ModuleSymbol(
        kind="module",
        name=module_name.split(".")[-1],
        qualname=module_name,
        module=module_name,
        filepath=str(path),
        line_range=root.line_range,
        visibility="public",
        exported=None,
        imports=[],
        from_imports=[],
        top_level=[c.qualname for c in classes],
    )
    mod.id_sha = _sha256(f"module:{mod.qualname}:{','.join(mod.top_level)}")

    return ExtractResult(module=mod, classes=classes, functions=functions, attributes=attributes)


def _class_body_node(n: TSNode) -> Optional[TSNode]:
    for body_type in BODY_NODE_TYPES:
        body = n.find_children(body_type)
        if body:
            return body[0]
    return None


def _iter_body_members(body: TSNode):
    for child in body.named_children():
        if child.type in BODY_WRAPPER_TYPES:
            yield from _iter_body_members(child)
        else:
            yield child


def _build_field_symbol(field_node: TSNode, owner: ClassSymbol) -> Optional[AttributeSymbol]:
    fmods = _mods(field_node)
    ftype = _first_type_text(field_node)
    fname = None
    init_text = None
    for ch in field_node.named_children():
        if ch.type == "variable_declarator":
            for sub in ch.named_children():
                if sub.type == "identifier":
                    fname = sub.text
                elif sub.type == "variable_initializer":
                    init_text = sub.text
    if not fname:
        return None
    visibility = "public"
    if "private" in fmods:
        visibility = "private"
    elif "public" in fmods:
        visibility = "public"
    attr = AttributeSymbol(
        kind="attribute",
        name=fname,
        qualname=f"{owner.qualname}.{fname}",
        module=owner.module,
        filepath=owner.filepath,
        line_range=field_node.line_range,
        docstring=None,
        visibility=visibility,
        exported=None,
        source_snippet=field_node.text,
        annotation=ftype,
        default=init_text,
        is_classvar=True,
        is_property=False,
    )
    return attr


def build_class_symbols(
    n: TSNode,
    pkg: str,
    path: Path,
    full_text: str,
    parent_qual: Optional[str] = None,
) -> List[ClassSymbol]:
    mods = _mods(n)
    ann = _annotations(n)
    name = _name_from(n) or "<anonymous>"
    if parent_qual:
        qual = f"{parent_qual}.{name}"
    else:
        qual = f"{pkg}.{name}" if pkg else name

    # type parameters
    tparams = []
    for ch in n.named_children():
        if ch.type == "type_parameters":
            tparams = [c.text for c in ch.named_children() if c.type in ("type_parameter", "identifier")]

    # extends / implements / permits
    bases, impls, permits = [], [], []
    for ch in n.named_children():
        if ch.type == "superclass":
            for t in ch.named_children():
                if t.type in ("type_identifier", "scoped_type_identifier", "type"):
                    bases.append(t.text)
        elif ch.type == "super_interfaces":
            impls = [t.text for t in ch.find_children("type_list") for t in t.named_children()]
        elif ch.type == "permits":
            permits = [x.text for x in ch.find_children("type_list") for x in x.named_children()]

    sealed = "sealed" in mods

    cls = ClassSymbol(
        kind="class",
        name=name,
        qualname=qual,
        module=pkg or "<default>",
        filepath=str(path),
        line_range=n.line_range,
        docstring=None,
        visibility="public" if "public" in mods else ("private" if "private" in mods else "public"),
        exported=None,
        source_snippet=n.text,
        decorators=ann,
        type_params=tparams,
        bases=bases,
        implements=impls,
        sealed=sealed,
        permits=permits,
    )

    body = _class_body_node(n)
    nested_classes: List[ClassSymbol] = []
    if body:
        for member in _iter_body_members(body):
            if member.type == "field_declaration":
                attr = _build_field_symbol(member, cls)
                if attr:
                    cls.attributes.append(attr)
            elif member.type == "method_declaration":
                cls.methods.append(build_method_symbol(member, cls, is_ctor=False))
            elif member.type == "constructor_declaration":
                cls.methods.append(build_method_symbol(member, cls, is_ctor=True))
            elif member.type in CLASS_DECL_TYPES:
                nested_classes.extend(build_class_symbols(member, pkg, path, full_text, cls.qualname))

    if n.type == "record_declaration":
        for ch in n.named_children():
            if ch.type == "record_header":
                for comp in ch.find_children("formal_parameter"):
                    cname = _first_child_text(comp, "identifier")
                    ctype = _first_type_text(comp)
                    if cname:
                        rc = AttributeSymbol(
                            kind="attribute",
                            name=cname,
                            qualname=f"{qual}.{cname}",
                            module=cls.module,
                            filepath=cls.filepath,
                            line_range=comp.line_range,
                            annotation=ctype,
                            default=None,
                        )
                        cls.record_components.append(rc)

    cls.id_sha = _sha256(f"{cls.module}.{cls.qualname}:{cls.header_string}")
    return [cls] + nested_classes


def build_method_symbol(n: TSNode, owner: ClassSymbol, is_ctor: bool) -> FunctionSymbol:
    mods = _mods(n)
    ann = _annotations(n)

    # name & return type
    name = _name_from(n) or (owner.name if is_ctor else "<anonymous>")
    ret_type = None if is_ctor else _first_type_text(n)

    # parameters
    params: List[Parameter] = []
    for ch in n.named_children():
        if ch.type == "formal_parameters":
            for fp in ch.find_children("formal_parameter"):
                ptype = _first_type_text(fp)
                pname = _first_child_text(fp, "identifier") or "_"
                params.append(Parameter(name=pname, kind="poskw", annotation=ptype))
            for vp in ch.find_children("spread_parameter"):
                ptype = _first_type_text(vp)
                pname = _first_child_text(vp, "identifier") or "_"
                params.append(Parameter(name=pname, kind="poskw", annotation=(ptype + "..." if ptype else "...")))

    # throws
    throws: List[str] = []
    for ch in n.named_children():
        if ch.type == "throws":
            for t in ch.find_children("type_list"):
                throws.extend([tt.text for tt in t.named_children()])

    qual = f"{owner.qualname}.{name}"
    lr = n.line_range

    fn = FunctionSymbol(
        kind="function",
        name=name,
        qualname=qual,
        module=owner.module,
        filepath=owner.filepath,
        line_range=lr,
        docstring=None,
        visibility="public" if "public" in mods else ("private" if "private" in mods else "public"),
        exported=None,
        source_snippet=n.text,
        parameters=params,
        return_annotation=ret_type,
        decorators=ann,
        raises=throws,
        async_def=False,
        is_method=True,
        is_staticmethod=("static" in mods),
        is_classmethod=False,
    )
    fn.id_sha = _sha256(f"{fn.module}.{fn.qualname}:{fn.signature_string}")
    return fn


def _first_child_text(n: TSNode, type_name: str) -> Optional[str]:
    for ch in n.named_children():
        if ch.type == type_name:
            return ch.text
    return None


def _first_type_text(n: TSNode) -> Optional[str]:
    """Return the textual representation of the first type-like child."""
    for ch in n.named_children():
        if ch.type in TYPE_NODE_NAMES:
            return ch.text
    for ch in n.named_children():
        out = _first_type_text(ch)
        if out:
            return out
    return None


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser(description="Java 17+ Symbol Extractor (tree-sitter)")
    ap.add_argument("--src", required=False, help="Root dir with .java files")
    ap.add_argument("--json", default=None, help="Output JSON file (stdout if omitted)")
    ap.add_argument("--selftest", action="store_true", help="Run built-in tests and exit")
    args = ap.parse_args()

    if args.selftest:
        run_self_tests()
        return

    if not args.src:
        raise SystemExit("--src is required unless --selftest is provided")

    root = Path(args.src).resolve()
    files = [p for p in root.rglob("*.java") if p.is_file()]

    all_classes: List[ClassSymbol] = []
    all_functions: List[FunctionSymbol] = []
    all_attributes: List[AttributeSymbol] = []

    module_symbol: Optional[ModuleSymbol] = None

    for p in files:
        try:
            res = extract_from_file(p)
            # Merge into project-level sets
            all_classes.extend(res.classes)
            all_functions.extend(res.functions)
            all_attributes.extend(res.attributes)
            if module_symbol is None:
                module_symbol = res.module
        except Exception as e:
            logger.warning("[warn] %s: %s", p, e)

    # Fallback module symbol if none parsed
    if module_symbol is None:
        module_symbol = ModuleSymbol(
            kind="module",
            name="<default>",
            qualname="<default>",
            module="<default>",
            filepath=str(root),
            line_range=(1, 1),
            visibility="public",
            exports=None,
            imports=[],
            from_imports=[],
            top_level=[c.qualname for c in all_classes],
        )
        module_symbol.id_sha = _sha256("module:<default>")

    project = ExtractResult(
        module=module_symbol,
        classes=all_classes,
        functions=all_functions,
        attributes=all_attributes,
    )

    out = json.dumps({
        "module": _to_dict(project.module),
        "classes": _to_dict(project.classes),
        "functions": _to_dict(project.functions),
        "attributes": _to_dict(project.attributes),
    }, ensure_ascii=False, indent=2)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(out, encoding="utf-8")
        logger.info("Wrote JSON -> %s", args.json)
    else:
        logger.info(out)


# ---------------------- Self tests ----------------------

def run_self_tests():
    logger.info("[selftest] Running Java 17+ extractor tests...")
    tmp = Path("_tmp_test.java")

    cases = {
        "class_basic": """
            package a.b;
            public class C extends Base implements X, Y {
                private int n = 1;
                public static String s;
                public C() {}
                public int add(int x, int y) throws Exception { return x + y; }
            }
        """,
        "record_sealed": """
            package demo;
            public sealed interface Shape permits Circle {}
            public record Circle(double r) {}
        """,
        "interface_default": """
            package p;
            public interface I { default int v() { return 1; } }
        """,
    }

    for name, code in cases.items():
        tmp.write_text(code, encoding="utf-8")
        res = extract_from_file(tmp)
        assert res.classes, f"{name}: expected classes"
        # Smoke checks per case
        if name == "class_basic":
            cls = next(c for c in res.classes if c.name == "C")
            assert "Base" in cls.bases, "extends Base missing"
            # fields
            field_names = {a.name for a in cls.attributes}
            assert {"n", "s"}.issubset(field_names), f"expected fields n and s, got {field_names}"
            # constructor
            ctors = [m for m in cls.methods if m.name == "C" and m.return_annotation is None]
            assert ctors, "constructor C() missing"
            # method add
            assert any(m.name == "add" for m in cls.methods), "method add missing"
            addm = next(m for m in cls.methods if m.name == "add")
            assert addm.return_annotation and "int" in addm.return_annotation, "return type int missing"
        if name == "record_sealed":
            names = [c.name for c in res.classes]
            assert "Shape" in names and "Circle" in names, "sealed/record types missing"
        if name == "interface_default":
            iface = next(c for c in res.classes if c.name == "I")
            assert any(m.is_staticmethod is False for m in iface.methods), "default method not found"
        print(f"[selftest] {name}: OK")

    try:
        tmp.unlink()
    except Exception:
        pass
    print("[selftest] All tests passed.")


def extract_multiple(paths: List[Path]) -> List[ExtractResult]:
    """Helper for tests: run the extractor on a list of files."""
    results = []
    for path in paths:
        results.append(extract_from_file(path))
    return results


if __name__ == "__main__":
    main()
