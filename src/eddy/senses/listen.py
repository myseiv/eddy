"""Listening: offline speech recognition with Vosk, so it works on a Pi without internet.

Needs:  pip install 'eddy-ai[voice]'  and a Vosk model folder, e.g.
    https://alphacephei.com/vosk/models  ->  vosk-model-small-en-us-0.15  (about 40 MB)
Point Eddy at it with --voice-model or the EDDY_VOSK_MODEL environment variable.
"""
from __future__ import annotations

import json
import os
import queue
from typing import Iterator, Optional


def default_model_path() -> Optional[str]:
    for path in (os.environ.get("EDDY_VOSK_MODEL"), os.path.expanduser("~/.eddy/vosk-model"), "models/vosk-model"):
        if path and os.path.isdir(path):
            return path
    return None


class Listener:
    def __init__(self, model_path: Optional[str] = None, samplerate: int = 16000, device=None,
                 vocabulary: Optional[str] = None):
        """vocabulary: space-separated words to listen for. Default: Eddy's command words.
        Pass "" for open dictation."""
        try:
            import vosk
        except ImportError as exc:
            raise RuntimeError("listening needs: pip install 'eddy-ai[voice]'") from exc
        model_path = model_path or default_model_path()
        if not model_path:
            raise RuntimeError("no Vosk speech model found. Download one from "
                               "https://alphacephei.com/vosk/models and pass --voice-model <folder>")
        vosk.SetLogLevel(-1)
        self._vosk = vosk
        self.samplerate = samplerate
        self.device = device
        self.model = vosk.Model(model_path)
        if vocabulary is None:
            from ..intents import VOCABULARY
            vocabulary = VOCABULARY
        self.grammar = json.dumps([vocabulary, "[unk]"]) if vocabulary else None

    def _recognizer(self, rate: int):
        if self.grammar:
            return self._vosk.KaldiRecognizer(self.model, rate, self.grammar)
        return self._vosk.KaldiRecognizer(self.model, rate)

    def phrases(self) -> Iterator[str]:
        """Yield each finished phrase heard on the microphone, forever."""
        try:
            import sounddevice
        except (ImportError, OSError) as exc:
            raise RuntimeError("the microphone needs: pip install 'eddy-ai[voice]' and "
                               "sudo apt install libportaudio2") from exc
        recognizer = self._recognizer(self.samplerate)
        audio: "queue.Queue[bytes]" = queue.Queue()
        with sounddevice.RawInputStream(samplerate=self.samplerate, blocksize=8000, dtype="int16", channels=1,
                                        device=self.device, callback=lambda data, *_: audio.put(bytes(data))):
            while True:
                if recognizer.AcceptWaveform(audio.get()):
                    text = _clean(json.loads(recognizer.Result()).get("text", ""))
                    if text:
                        yield text

    def transcribe(self, wav_path: str) -> str:
        """Recognise speech in a 16-bit mono WAV file (for testing without a microphone)."""
        import wave
        with wave.open(wav_path, "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("need a 16-bit mono WAV file")
            recognizer = self._recognizer(wav.getframerate())
            heard = []
            while True:
                data = wav.readframes(4000)
                if not data:
                    break
                if recognizer.AcceptWaveform(data):
                    heard.append(json.loads(recognizer.Result()).get("text", ""))
            heard.append(json.loads(recognizer.FinalResult()).get("text", ""))
        return _clean(" ".join(heard))


def _clean(text: str) -> str:
    """Drop the recogniser's marker for sounds that were not one of our words."""
    return " ".join(w for w in text.split() if w != "[unk]")


def strip_wake_word(text: str, wake_word: Optional[str] = "eddy") -> Optional[str]:
    """With a wake word set, only sentences that contain it count; return the rest of the sentence."""
    if not wake_word:
        return text
    words = text.lower().split()
    # Speech recognisers hear "eddy" as several near-misses.
    variants = {wake_word, "eddie", "eddy", "edi", "eddi", "eddies"} if wake_word == "eddy" else {wake_word}
    for i, w in enumerate(words):
        if w in variants:
            rest = " ".join(words[i + 1:]).strip()
            return rest or None
    return None
