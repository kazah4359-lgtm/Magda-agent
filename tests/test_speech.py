
import sys
from unittest.mock import MagicMock
import sys
from unittest.mock import MagicMock
import sys
from unittest.mock import MagicMock
import pytest
import os
import sys
import asyncio

# Make sure the module can be imported properly by mocking at module load level
import sys
from unittest.mock import MagicMock

# Create a mock for torch
mock_torch = MagicMock()
mock_torch.tensor.return_value.unsqueeze.return_value = "fake_embeddings"

# Create a mock for huggingface/transformers before importing our module
mock_pipeline_func = MagicMock()
def mock_pipeline_call(*args, **kwargs):
    return {"text": "mocked transcribed text"}

mock_pipeline_func.return_value = mock_pipeline_call

mock_transformers = MagicMock()
mock_transformers.pipeline = mock_pipeline_func
mock_processor = MagicMock()
mock_processor.return_value = MagicMock(return_value={"input_ids": [1, 2, 3]})
mock_transformers.SpeechT5Processor.from_pretrained.return_value = mock_processor
mock_transformers.SpeechT5ForTextToSpeech.from_pretrained.return_value = MagicMock()
mock_transformers.SpeechT5HifiGan.from_pretrained.return_value = MagicMock()

mock_datasets = MagicMock()
mock_datasets.load_dataset.return_value = [{"xvector": [0.1]*512} for _ in range(8000)]

mock_anyascii = MagicMock()
mock_anyascii.anyascii.return_value = "Privet mir"

mock_sf = MagicMock()

mock_pydub = MagicMock()

sys.modules["torch"] = mock_torch
sys.modules["transformers"] = mock_transformers
sys.modules["datasets"] = mock_datasets
sys.modules["anyascii"] = mock_anyascii
sys.modules["soundfile"] = mock_sf
sys.modules["pydub"] = mock_pydub

from magda_agent.speech.processor import SpeechProcessor

@pytest.fixture
def processor():
    return SpeechProcessor()

@pytest.mark.asyncio
async def test_stt(processor, monkeypatch):
    # Mock AudioSegment so it doesn't try to read a real file
    mock_audio = MagicMock()
    import pydub
    monkeypatch.setattr(pydub.AudioSegment, "from_file", MagicMock(return_value=mock_audio))
    monkeypatch.setattr(processor, "_stt_pipe", MagicMock(return_value={"text": "mocked transcribed text"}))
    monkeypatch.setattr(processor, "_ensure_models_loaded", MagicMock())

    # Run STT
    result = await processor.stt("dummy.ogg")
    assert result == "mocked transcribed text"

@pytest.mark.asyncio
async def test_tts(processor, monkeypatch):
    # Mock sf.write and pydub so it doesn't write real files
    import soundfile as sf
    monkeypatch.setattr(sf, "write", MagicMock())
    mock_audio = MagicMock()
    import pydub
    monkeypatch.setattr(pydub.AudioSegment, "from_wav", MagicMock(return_value=mock_audio))

    # Run TTS
    out_path = await processor.tts("Привет мир", "out.ogg")
    assert out_path == "out.ogg"

    # Verify mock was called
    assert mock_audio.export.called
    assert mock_audio.export.call_args[1]["format"] == "ogg"
    assert mock_audio.export.call_args[1]["codec"] == "libopus"
