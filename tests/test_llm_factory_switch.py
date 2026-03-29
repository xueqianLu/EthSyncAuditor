from __future__ import annotations

import os

import pytest

from agents import llm_factory


class _DummyModel:
    pass


def test_normalize_provider_aliases():
    assert llm_factory._normalize_provider("claude") == "claude"
    assert llm_factory._normalize_provider("anthropic") == "claude"
    assert llm_factory._normalize_provider("gemini") == "gemini"
    assert llm_factory._normalize_provider("google") == "gemini"


@pytest.mark.parametrize("value", ["x", "openai"]) 
def test_normalize_provider_invalid(value: str):
    with pytest.raises(ValueError):
        llm_factory._normalize_provider(value)


def test_create_llm_dispatch_claude(monkeypatch):
    called = {}

    def _fake_create_claude_llm(model: str, temperature: float):
        called["provider"] = "claude"
        called["model"] = model
        called["temperature"] = temperature
        return _DummyModel()

    monkeypatch.setattr(llm_factory, "create_claude_llm", _fake_create_claude_llm)

    obj = llm_factory.create_llm(provider="claude", model="claude-test", temperature=0.2)
    assert isinstance(obj, _DummyModel)
    assert called["provider"] == "claude"
    assert called["model"] == "claude-test"


def test_create_llm_dispatch_gemini(monkeypatch):
    called = {}

    def _fake_create_gemini_llm(model: str, temperature: float):
        called["provider"] = "gemini"
        called["model"] = model
        called["temperature"] = temperature
        return _DummyModel()

    monkeypatch.setattr(llm_factory, "create_gemini_llm", _fake_create_gemini_llm)

    obj = llm_factory.create_llm(provider="gemini", model="gemini-test", temperature=0.1)
    assert isinstance(obj, _DummyModel)
    assert called["provider"] == "gemini"
    assert called["model"] == "gemini-test"


def test_provider_from_env(monkeypatch):
    called = {}

    def _fake_create_gemini_llm(model: str, temperature: float):
        called["model"] = model
        return _DummyModel()

    monkeypatch.setattr(llm_factory, "create_gemini_llm", _fake_create_gemini_llm)
    monkeypatch.setenv("ETHAUDITOR_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("ETHAUDITOR_GEMINI_MODEL", "gemini-env-model")

    obj = llm_factory.create_llm(provider=None, model=None, temperature=0.0)
    assert isinstance(obj, _DummyModel)
    assert called["model"] == "gemini-env-model"
