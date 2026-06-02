import os
import torch
import torchaudio
import soundfile as sf
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from pydub import AudioSegment
from anyascii import anyascii

class SpeechProcessor:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # STT pipeline (Whisper is robust for Russian and multi-language)
        self.stt_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=self.device
        )

        # TTS pipeline (SpeechT5 from Microsoft)
        self.tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)

        # Load a speaker embedding (using a sample from CMU Arctic)
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)

    def convert_ogg_to_wav(self, ogg_path: str, wav_path: str) -> None:
        """Converts Telegram voice note (OGG/OPUS) to WAV."""
        audio = AudioSegment.from_file(ogg_path, format="ogg")
        # STT models generally expect 16kHz
        audio = audio.set_frame_rate(16000)
        audio.export(wav_path, format="wav")

    def convert_wav_to_ogg(self, wav_path: str, ogg_path: str) -> None:
        """Converts generated WAV to Telegram compatible OGG (OPUS)."""
        audio = AudioSegment.from_file(wav_path, format="wav")
        audio.export(ogg_path, format="ogg", codec="libopus")

    def speech_to_text(self, audio_file_path: str) -> str:
        """Transcribes an audio file to text."""
        result = self.stt_pipeline(audio_file_path)
        return result["text"]

    def text_to_speech(self, text: str, output_path: str) -> None:
        """Synthesizes text to speech and saves it as a WAV file."""
        # SpeechT5 is primarily trained on English and will fail on Cyrillic.
        # We transliterate the text so the model can read it, preserving the flow.
        # SpeechT5 has a maximum input length limit (usually ~600 characters).
        # We truncate the input to prevent IndexError.
        truncated_text = text[:550]
        transliterated_text = anyascii(truncated_text)

        # Process the text
        inputs = self.tts_processor(text=transliterated_text, return_tensors="pt").to(self.device)

        # Generate speech
        speech = self.tts_model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)

        # Save to file
        sf.write(output_path, speech.cpu().numpy(), samplerate=16000)
