"""Feature coverage tests for the Java symbol extractor."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pytest  # type: ignore

from src.utils.java_symbol_extractor import extract_from_file

DATA_DIR = Path(__file__).parent.parent / "src" / "utils" / "testdata"


@pytest.mark.parametrize(
    "filename",
    [
        "SimpleClass.java",
        "Generics.java",
        "Records.java",
        "Nested.java",
        "Interfaces.java",
    ],
)
def test_extracts_without_error(filename: str):
    path = DATA_DIR / filename
    result = extract_from_file(path)
    assert result.classes, f"{filename}: expected at least one class"


def test_simple_class_members():
    res = extract_from_file(DATA_DIR / "SimpleClass.java")
    simple = next(c for c in res.classes if c.name == "SimpleClass")
    field_names = {a.name for a in simple.attributes}
    method_names = {m.name for m in simple.methods}
    assert {"counter", "NAME"}.issubset(field_names)
    assert "increment" in method_names


def test_generics_static_method():
    res = extract_from_file(DATA_DIR / "Generics.java")
    box = next(c for c in res.classes if c.name == "Box")
    static_methods = [m for m in box.methods if m.is_staticmethod]
    assert any(m.name == "first" for m in static_methods)
    first = next(m for m in static_methods if m.name == "first")
    assert first.return_annotation == "U"
    assert any(param.annotation == "List<U>" for param in first.parameters)


def test_records_and_sealed():
    res = extract_from_file(DATA_DIR / "Records.java")
    names = {c.name for c in res.classes}
    sealed = next(c for c in res.classes if c.name == "Shape")
    assert {"Shape", "Circle", "Rectangle"}.issubset(names)
    assert sealed.permits == ["Circle", "Rectangle"]


def test_nested_classes():
    res = extract_from_file(DATA_DIR / "Nested.java")
    by_name = {c.qualname: c for c in res.classes}
    assert "demo.nested.Outer" in by_name
    assert "demo.nested.Outer.Inner" in by_name
    assert "demo.nested.Outer.StaticNested" in by_name

    inner = by_name["demo.nested.Outer.Inner"]
    assert any(m.name == "square" for m in inner.methods)

    static_nested = by_name["demo.nested.Outer.StaticNested"]
    double_method = next(m for m in static_nested.methods if m.name == "doubleValue")
    assert double_method.is_staticmethod


def test_interface_default_and_static():
    res = extract_from_file(DATA_DIR / "Interfaces.java")
    iface = next(c for c in res.classes if c.name == "Computable")
    method_by_name = {m.name: m for m in iface.methods}
    assert method_by_name["compute"].return_annotation == "int"
    assert method_by_name["name"].is_staticmethod
