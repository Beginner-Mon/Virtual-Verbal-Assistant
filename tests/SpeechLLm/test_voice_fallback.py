# -*- coding: utf-8 -*-
"""A missing reference .wav must degrade to the preset voice, never raise.

SpeechLLm is the only place in the system that can answer "does that file
exist?" — the LangGraph service builds `voices/<slug>_<lang>.wav` in another
process and, deployed, on another host. So the check and the fallback both
belong here, and a character with no recording yet still has to be able to
speak.
"""

import logging

import pytest

from src.services.vieneu_client import VieNeuClient, _resolve_ref


class _FakeTTS:
    """Stands in for the GGUF model — loading the real one costs ~30s."""

    def __init__(self):
        self.encoded = []

    def encode_reference(self, path):
        self.encoded.append(path)
        return f"<voice:{path}>"


@pytest.fixture
def client(tmp_path):
    c = VieNeuClient({"output_dir": str(tmp_path / "out"), "voice_path": ""})
    c._tts = _FakeTTS()          # _load_model() short-circuits on this
    return c


@pytest.mark.unit
def test_missing_reference_falls_back_instead_of_raising(client, caplog):
    with caplog.at_level(logging.WARNING, logger="speechllm.vieneu"):
        voice = client._resolve_voice("voices/nobody_recorded_this_vi.wav")

    assert voice is None                       # preset
    assert client._tts.encoded == []           # never attempted to encode
    assert "voice_reference_missing" in caplog.text
    # The absolute path matters: the symptom of this branch is audio in the
    # wrong voice, which by ear is indistinguishable from "right voice, bad
    # model". Without the resolved path in the log there is nothing to search.
    assert str(_resolve_ref("voices/nobody_recorded_this_vi.wav")) in caplog.text


@pytest.mark.unit
def test_existing_reference_is_encoded_and_cached(client, tmp_path, monkeypatch):
    wav = tmp_path / "anne_vi.wav"
    wav.write_bytes(b"RIFF")

    first = client._resolve_voice(str(wav))
    second = client._resolve_voice(str(wav))

    assert first == second
    assert len(client._tts.encoded) == 1, "second call re-encoded instead of using the cache"


@pytest.mark.unit
def test_relative_paths_anchor_to_speechllm_not_the_cwd(monkeypatch, tmp_path):
    """uvicorn's working directory must not change which file is found."""
    monkeypatch.chdir(tmp_path)
    assert _resolve_ref("voices/anne_vi.wav").is_absolute()
    assert _resolve_ref("voices/anne_vi.wav").parts[-2:] == ("voices", "anne_vi.wav")


@pytest.mark.unit
def test_absolute_paths_are_left_alone():
    p = _resolve_ref("C:/somewhere/else/x.wav" if __import__("os").name == "nt"
                     else "/somewhere/else/x.wav")
    assert p.is_absolute()
    assert p.name == "x.wav"


@pytest.mark.unit
def test_no_voice_path_at_all_is_the_preset(client):
    assert client._resolve_voice(None) is None
    assert client._tts.encoded == []
