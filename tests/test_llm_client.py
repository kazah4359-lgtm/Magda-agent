import os
from unittest.mock import patch

from magda_agent.llm_client import LLMClient


def test_defaults_to_openai_without_base_url():
    with patch.dict(os.environ, {}, clear=True):
        client = LLMClient(api_key="sk-test")
        assert client.model == "gpt-4o"
        assert client.base_url is None


def test_reads_base_url_and_model_from_env():
    env = {
        "OPENAI_API_KEY": "sk-test",
        "OPENAI_BASE_URL": "http://gemini-web2api:8081/v1",
        "OPENAI_MODEL": "gemini-3.5-flash-thinking",
    }
    with patch.dict(os.environ, env, clear=True):
        client = LLMClient()
        assert client.base_url == "http://gemini-web2api:8081/v1"
        assert client.model == "gemini-3.5-flash-thinking"


def test_explicit_args_override_env():
    env = {
        "OPENAI_API_KEY": "sk-env",
        "OPENAI_BASE_URL": "http://env-url/v1",
        "OPENAI_MODEL": "env-model",
    }
    with patch.dict(os.environ, env, clear=True):
        client = LLMClient(api_key="sk-arg", model="arg-model", base_url="http://arg-url/v1")
        assert client.api_key == "sk-arg"
        assert client.model == "arg-model"
        assert client.base_url == "http://arg-url/v1"


def test_base_url_passed_to_async_client():
    env = {"OPENAI_API_KEY": "sk-test", "OPENAI_BASE_URL": "http://gemini-web2api:8081/v1"}
    with patch.dict(os.environ, env, clear=True):
        client = LLMClient()
        # The underlying AsyncOpenAI client should carry the configured base_url.
        assert str(client.client.base_url).rstrip("/") == "http://gemini-web2api:8081/v1"
