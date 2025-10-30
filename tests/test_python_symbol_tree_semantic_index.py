from __future__ import annotations

from textwrap import dedent

from src.symindex import ClassUnit, FunctionUnit, PropertyUnit, build_semantic_index


def _sample_source() -> str:
    return dedent(
        '''
        """Example module docstring."""

        class Foo:
            bar: int = 1

            @property
            def name(self) -> str:
                return "foo"

            def method(self, value: int = 2) -> int:
                return value

        async def helper(flag: bool) -> None:
            return None

        def top(level: int = 5) -> int:
            return helper(True) or level
        '''
    )


def test_semantic_index_contains_expected_units():
    src = _sample_source()
    index = build_semantic_index(src=src, module="sample", filepath=None)

    # module metadata
    assert index.module.qualname == "sample"
    assert index.module.top_level == ["Foo", "helper", "top"]

    # find class and functions
    foo_units = index.find_by_qualname("Foo")
    assert foo_units and foo_units[0].kind == "class"

    property_units = [u for u in index.units if u.kind == "property"]
    assert any(p.qualname == "Foo.name" for p in property_units)

    public_api = index.list_public_api("sample")
    public_names = {unit.qualname for unit in public_api}
    assert {"Foo", "helper", "top"}.issubset(public_names)


def test_class_units_group_fields_properties_and_methods():
    src = _sample_source()
    index = build_semantic_index(src=src, module="sample", filepath=None)

    foo = next(unit for unit in index.units if isinstance(unit, ClassUnit))

    assert [field.name for field in foo.fields] == ["bar"]
    assert [prop.name for prop in foo.properties] == ["name"]
    assert [method.name for method in foo.methods] == ["method"]

    property_unit = next(unit for unit in index.units if isinstance(unit, PropertyUnit))
    assert property_unit.qualname == "Foo.name"
    assert property_unit.type_annotation == "str"


def test_top_level_functions_preserve_signatures_and_async_flag():
    src = _sample_source()
    index = build_semantic_index(src=src, module="sample", filepath=None)

    helper = next(unit for unit in index.units if isinstance(unit, FunctionUnit) and unit.name == "helper")
    assert helper.async_def is True
    assert helper.parameters[0].name == "flag"
    assert helper.parameters[0].annotation == "bool"

    top_fn = next(unit for unit in index.units if isinstance(unit, FunctionUnit) and unit.name == "top")
    param_defaults = {param.name: param.default for param in top_fn.parameters}
    assert param_defaults == {"level": "5"}
    assert top_fn.return_annotation == "int"
