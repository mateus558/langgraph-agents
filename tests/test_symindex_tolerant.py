from __future__ import annotations

import pytest

from src.utils.symindex import SymindexParseError, build_semantic_index


def test_syntax_error_raises_without_tolerance():
    bad_src = "def broken(:\n    pass\n"

    with pytest.raises(SymindexParseError) as exc:
        build_semantic_index(src=bad_src, module="broken", filepath=None)

    # The error message should include module name and location
    assert "broken" in str(exc.value)
    assert "line" in str(exc.value)


ess_broken_snippet = """
class X:
    def a(self):
        return [1,2
"""


def test_tolerate_errors_returns_minimal_index_with_error():
    index = build_semantic_index(src=ess_broken_snippet, module="broken2", filepath=None, tolerate_errors=True)

    assert index.module.qualname == "broken2"
    assert index.units == []
    assert index.edges == []
    assert index.errors and isinstance(index.errors[0], dict)
    err = index.errors[0]
    assert "message" in err and "lineno" in err
