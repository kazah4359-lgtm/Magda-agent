import logging
import anyascii
from transformers import pipeline
import soundfile as sf
import torch

class SpeechProcessor:
    def __init__(self):
        logging.info("Initializing SpeechProcessor ML models...")
        try:
            self.stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
            self.tts_pipeline = pipeline("text-to-speech", model="microsoft/speecht5_tts")
            from transformers import SpeechT5HifiGan
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

            # Load xvector containing speaker's voice characteristics
            # Note: We create a dummy tensor for the speaker embedding here
            # In a real app, this should be downloaded (e.g. from Mathexen/speecht5-cmu-arctic-xvectors)
            from datasets import load_dataset
            embeddings_dataset = load_dataset("Mathexen/speecht5-cmu-arctic-xvectors", split="validation")
            self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

            logging.info("SpeechProcessor models initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize SpeechProcessor models: {e}")
            self.stt_pipeline = None
            self.tts_pipeline = None
            self.vocoder = None
            self.speaker_embeddings = None

    def stt(self, audio_path: str) -> str:
        """Transcribes audio file to text."""
        if not self.stt_pipeline:
            return "STT model not initialized."
        try:
            result = self.stt_pipeline(audio_path)
            return result.get("text", "")
        except Exception as e:
            logging.error(f"STT error: {e}")
            return ""

    def tts(self, text: str, output_path: str) -> bool:
        """Converts text to speech and saves it to output_path.
        Returns True if successful, False otherwise.
        """
        if not self.tts_pipeline or self.vocoder is None or self.speaker_embeddings is None:
            return False

        # Transliterate Cyrillic and other non-ASCII to ASCII (speecht5 only supports ASCII)
        ascii_text = anyascii.anyascii(text)

        # Truncate to ~550 chars to prevent index error
        ascii_text = ascii_text[:550]
        if not ascii_text.strip():
            return False

        try:
            speech = self.tts_pipeline(ascii_text, forward_params={"speaker_embeddings": self.speaker_embeddings, "vocoder": self.vocoder})
            sf.write(output_path, speech["audio"], samplerate=speech["sampling_rate"])
            return True
        except Exception as e:
            logging.error(f"TTS error: {e}")
            return False

# Global instance to avoid reloading on every request
speech_processor = SpeechProcessor()
