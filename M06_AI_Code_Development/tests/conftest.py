"""Fail immediately if a contract test accidentally reaches the live API."""
import pytest
import openai


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def blocked(*args, **kwargs):
        pytest.fail("Offline tests must replace API calls with fakes.")

    monkeypatch.setattr(openai.OpenAI, "__init__", blocked)
