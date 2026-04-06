"""Tests for utils.py — shared utility functions."""

from utils import safe_serialize


def test_serialize_primitives():
    assert safe_serialize("hello") == "hello"
    assert safe_serialize(42) == 42
    assert safe_serialize(3.14) == 3.14
    assert safe_serialize(True) is True
    assert safe_serialize(None) is None


def test_serialize_dict():
    result = safe_serialize({"a": 1, "b": [2, 3]})
    assert result == {"a": 1, "b": [2, 3]}


def test_serialize_list():
    result = safe_serialize([1, "x", None])
    assert result == [1, "x", None]


def test_serialize_tuple():
    result = safe_serialize((1, 2))
    assert result == [1, 2]


def test_serialize_pydantic():
    from state import VocabEntry

    v = VocabEntry(name="G1", category="net", description="d")
    result = safe_serialize(v)
    assert result["name"] == "G1"


def test_serialize_nested():
    from state import VocabEntry

    data = {
        "items": [VocabEntry(name="G1", category="net", description="d")],
        "count": 1,
    }
    result = safe_serialize(data)
    assert result["items"][0]["name"] == "G1"


def test_serialize_unknown_type():
    """Objects without model_dump fall back to __dict__ or str()."""

    class Opaque:
        __slots__ = ()  # no __dict__

    result = safe_serialize(Opaque())
    assert isinstance(result, str)


def test_serialize_object_with_dict():
    """Objects with __dict__ are serialized as dicts."""

    class Simple:
        def __init__(self):
            self.x = 1
            self._private = 2

    result = safe_serialize(Simple())
    assert result["x"] == 1
    assert "_private" not in result
