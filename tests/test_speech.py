import pytest
import os
from unittest.mock import patch, MagicMock
from magda_agent.speech import SpeechProcessor

@pytest.fixture
def mock_speech_processor():
    with patch('magda_agent.speech.pipeline') as mock_pipeline, \
         patch('transformers.SpeechT5HifiGan') as mock_hifigan, \
         patch('datasets.load_dataset') as mock_load_dataset:

        # Mock STT pipeline
        mock_stt = MagicMock()
        mock_stt.return_value = {"text": "hello world"}

        # Mock TTS pipeline
        mock_tts = MagicMock()
        mock_tts.return_value = {"audio": [0.1, 0.2], "sampling_rate": 16000}

        # Determine which pipeline to return based on task
        def pipeline_side_effect(task, model=None):
            if task == "automatic-speech-recognition":
                return mock_stt
            elif task == "text-to-speech":
                return mock_tts
            return MagicMock()

        mock_pipeline.side_effect = pipeline_side_effect

        # Mock vocoder
        mock_vocoder = MagicMock()
        mock_hifigan.from_pretrained.return_value = mock_vocoder

        # Mock speaker embeddings
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = {"xvector": [0.1] * 512}
        mock_load_dataset.return_value = mock_dataset

        processor = SpeechProcessor()
        return processor

def test_stt_success(mock_speech_processor):
    text = mock_speech_processor.stt("dummy_audio.wav")
    assert text == "hello world"
    mock_speech_processor.stt_pipeline.assert_called_once_with("dummy_audio.wav")

def test_stt_not_initialized():
    processor = SpeechProcessor()
    processor.stt_pipeline = None
    text = processor.stt("dummy_audio.wav")
    assert text == "STT model not initialized."

@patch('magda_agent.speech.sf.write')
def test_tts_success(mock_sf_write, mock_speech_processor):
    success = mock_speech_processor.tts("hello", "output.wav")
    assert success is True
    mock_speech_processor.tts_pipeline.assert_called_once()
    mock_sf_write.assert_called_once()
    args, _ = mock_sf_write.call_args
    assert args[0] == "output.wav"
    assert args[1] == [0.1, 0.2]

def test_tts_not_initialized():
    processor = SpeechProcessor()
    processor.tts_pipeline = None
    success = processor.tts("hello", "output.wav")
    assert success is False

@patch('magda_agent.speech.sf.write')
def test_tts_transliteration(mock_sf_write, mock_speech_processor):
    # Test cyrillic transliteration
    success = mock_speech_processor.tts("привет", "output.wav")
    assert success is True
    # Check that anyascii transliterated it
    args, kwargs = mock_speech_processor.tts_pipeline.call_args
    assert args[0] == "privet"

@patch('magda_agent.speech.sf.write')
def test_tts_truncation(mock_sf_write, mock_speech_processor):
    # Test long string truncation
    long_text = "a" * 1000
    success = mock_speech_processor.tts(long_text, "output.wav")
    assert success is True
    # Check that it was truncated to 550 chars
    args, kwargs = mock_speech_processor.tts_pipeline.call_args
    assert len(args[0]) == 550
