from __future__ import annotations

from textwrap import dedent

from src.symindex import ClassUnit, FunctionUnit, PropertyUnit, build_semantic_index


def test_basic_module_units():
    src = dedent(
        '''
        """Simple module docstring."""

        def greet(name: str) -> str:
            return f"hi {name}"
        '''
    )

    index = build_semantic_index(src=src, module="basic", filepath=None)

    assert index.module.qualname == "basic"
    assert index.module.top_level == ["greet"]

    greet = next(unit for unit in index.units if isinstance(unit, FunctionUnit))
    assert greet.parameters[0].name == "name"
    assert greet.parameters[0].annotation == "str"
    assert greet.return_annotation == "str"
    assert greet.visibility == "public"


def test_class_with_property_and_method():
    src = dedent(
        '''
        class Widget:
            kind: str = "widget"

            def __init__(self, value: int) -> None:
                self.value = value

            @property
            def size(self) -> int:
                return self.value

            def double(self) -> int:
                return self.value * 2
        '''
    )

    index = build_semantic_index(src=src, module="widget_mod", filepath=None)

    widget = next(unit for unit in index.units if isinstance(unit, ClassUnit))
    assert [field.name for field in widget.fields] == ["kind"]
    assert [prop.name for prop in widget.properties] == ["size"]
    assert [method.name for method in widget.methods] == ["__init__", "double"]

    property_unit = next(unit for unit in index.units if isinstance(unit, PropertyUnit))
    assert property_unit.type_annotation == "int"
    assert property_unit.parent_qualname == "Widget"

    double_method = next(
        unit for unit in index.units if isinstance(unit, FunctionUnit) and unit.name == "double"
    )
    assert [param.name for param in double_method.parameters] == ["self"]
    assert double_method.return_annotation == "int"


def test_nested_structures_and_async_functions():
    src = dedent(
        '''
        class Container:
            class Inner:
                pass

            async def run(self, flag: bool) -> None:
                if flag:
                    await self._inner()

            async def _inner(self) -> None:
                return None

        async def outer(flag: bool = False) -> None:
            return None
        '''
    )

    index = build_semantic_index(src=src, module="nest", filepath=None)

    container = next(unit for unit in index.units if isinstance(unit, ClassUnit) and unit.name == "Container")
    inner = next(unit for unit in index.units if isinstance(unit, ClassUnit) and unit.name == "Inner")
    run_method = next(
        unit for unit in index.units if isinstance(unit, FunctionUnit) and unit.qualname == "Container.run"
    )
    outer_func = next(unit for unit in index.units if isinstance(unit, FunctionUnit) and unit.name == "outer")

    assert run_method.async_def is True
    assert [param.name for param in run_method.parameters] == ["self", "flag"]
    assert run_method.parameters[1].annotation == "bool"

    assert outer_func.async_def is True
    assert {param.name: param.default for param in outer_func.parameters} == {"flag": "False"}

    # Ensure containment edge between Container and Inner exists
    container_contains_inner = any(
        edge.type == "contains" and edge.src_id == container.id_sha and edge.dst_id == inner.id_sha
        for edge in index.edges
    )
    assert container_contains_inner