"""LLM credentials must not have to live in the CloudFormation template.

Lambda environment variables are plaintext in the stack — anyone who can
`describe-stacks` can read them. db/postgres.py already solved this for the
database DSN by reading an SSM SecureString at run time; llm.py now does the
same for DEEPSEEK_API_KEY and GEMINI_API_KEYS. CloudFormation's
`{{resolve:ssm-secure}}` dynamic reference is not an alternative here: it is not
supported for Lambda environment variables.

Two properties are worth pinning, and neither is about the happy path:

  * env beats SSM, so a developer with a scratch key exported is not silently
    overridden by whatever the deployment is configured with;
  * an SSM failure returns None rather than raising, because "no key" already
    has a defined meaning upstream (DeepSeek unset falls back to Gemini) and
    turning a credential fetch into an exception would make a degraded provider
    into a dead request.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langgraph_agents import llm


@pytest.fixture(autouse=True)
def _clear_secret_cache():
    """_secret_from_ssm is lru_cached for the process; tests change its answer."""
    llm._secret_from_ssm.cache_clear()
    yield
    llm._secret_from_ssm.cache_clear()


def _fake_ssm(value: str):
    client = MagicMock()
    client.get_parameter.return_value = {"Parameter": {"Value": value}}
    boto3 = MagicMock()
    boto3.client.return_value = client
    return patch.dict("sys.modules", {"boto3": boto3}), client


@pytest.mark.unit
def test_env_wins_over_ssm(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    monkeypatch.setenv("DEEPSEEK_API_KEY_PARAM", "/vva/deepseek")

    patcher, client = _fake_ssm("sk-from-ssm")
    with patcher:
        assert llm._resolve_api_key() == "sk-from-env"
    client.get_parameter.assert_not_called()


@pytest.mark.unit
def test_falls_back_to_ssm_when_env_absent(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY_PARAM", "/vva/deepseek")

    patcher, client = _fake_ssm("sk-from-ssm")
    with patcher:
        assert llm._resolve_api_key() == "sk-from-ssm"
    client.get_parameter.assert_called_once()
    assert client.get_parameter.call_args.kwargs["WithDecryption"] is True


@pytest.mark.unit
def test_no_param_means_no_lookup(monkeypatch):
    """Local development must not attempt an AWS call it never asked for."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY_PARAM", raising=False)

    patcher, client = _fake_ssm("unused")
    with patcher:
        assert llm._resolve_api_key() is None
    client.get_parameter.assert_not_called()


@pytest.mark.unit
def test_ssm_failure_degrades_to_none(monkeypatch):
    """A broken SSM call must not raise into the caller.

    DeepSeek unconfigured is a state the code already handles by falling back to
    Gemini. An exception here would convert that into a failed request.
    """
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY_PARAM", "/vva/deepseek")

    boto3 = MagicMock()
    boto3.client.side_effect = RuntimeError("no credentials")
    with patch.dict("sys.modules", {"boto3": boto3}):
        assert llm._resolve_api_key() is None


@pytest.mark.unit
def test_gemini_reads_the_comma_separated_shape_from_ssm(monkeypatch):
    """One parameter holding the whole list, not one parameter per key."""
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS_PARAM", "/vva/gemini")

    patcher, _ = _fake_ssm(" key-one , key-two ")
    with patcher:
        assert llm._resolve_first_gemini_key() == "key-one"


@pytest.mark.unit
def test_secret_is_fetched_once_per_parameter(monkeypatch):
    """Called per LLM construction; each miss would be a network round trip."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY_PARAM", "/vva/deepseek")

    patcher, client = _fake_ssm("sk-cached")
    with patcher:
        for _ in range(5):
            assert llm._resolve_api_key() == "sk-cached"
    client.get_parameter.assert_called_once()
