import asyncio
import logging
import os
import torch
import anyascii
import soundfile as sf
from pydub import AudioSegment
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

class SpeechProcessor:
    """
    Handles Speech-to-Text (STT) and Text-to-Speech (TTS) conversions.
    """

    def __init__(self):
        """Initialize the SpeechProcessor and its models."""
        self._stt_pipe = None
        self._tts_processor = None
        self._tts_model = None
        self._tts_vocoder = None
        self._speaker_embeddings = None

    def _ensure_models_loaded(self):
        """Load the models lazily on first use."""
        if self._stt_pipe is None:
            logging.info("Loading STT model...")
            self._stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

            logging.info("Loading TTS models...")
            self._tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self._tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self._tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

            # Load xvector containing speaker's voice characteristics from a dataset
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            self._speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    async def stt(self, audio_path: str) -> str:
        """
        Convert speech in audio file to text.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            str: The transcribed text.
        """
        def _stt():
            self._ensure_models_loaded()
            # pydub can load ogg/opus. Convert to wav for whisper
            audio = AudioSegment.from_file(audio_path)
            temp_wav = audio_path + ".wav"
            audio.export(temp_wav, format="wav")
            result = self._stt_pipe(temp_wav)
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            return result["text"]

        return await asyncio.to_thread(_stt)

    async def tts(self, text: str, output_path: str) -> str:
        """
        Convert text to speech and save as an OGG file with OPUS codec.

        Args:
            text (str): The text to synthesize.
            output_path (str): The path to save the output audio.

        Returns:
            str: The output path of the generated audio file.
        """
        def _tts():
            self._ensure_models_loaded()
            # Transliterate and truncate
            clean_text = anyascii.anyascii(text)
            if len(clean_text) > 550:
                clean_text = clean_text[:550]

            inputs = self._tts_processor(text=clean_text, return_tensors="pt")
            speech = self._tts_model.generate_speech(inputs["input_ids"], self._speaker_embeddings, vocoder=self._tts_vocoder)

            # Save as temp WAV
            temp_wav = output_path + ".wav"
            sf.write(temp_wav, speech.numpy(), samplerate=16000)

            # Convert to OGG OPUS using pydub
            audio = AudioSegment.from_wav(temp_wav)
            audio.export(output_path, format="ogg", codec="libopus")

            if os.path.exists(temp_wav):
                os.remove(temp_wav)

            return output_path

        return await asyncio.to_thread(_tts)
