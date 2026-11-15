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
    """Find the speech model: the environment variable first, then where the installer puts
    it, then a folder next to the code. None if there is no model anywhere."""
    for path in (os.environ.get("EDDY_VOSK_MODEL"), os.path.expanduser("~/.eddy/vosk-model"), "models/vosk-model"):
        if path and os.path.isdir(path):
            return path
    return None


class Listener:
    """Turns speech into text, from the microphone (phrases) or from a WAV file (transcribe)."""

    def __init__(self, model_path: Optional[str] = None, samplerate: int = 16000, device=None,
                 vocabulary: Optional[str] = None):
        """vocabulary: space-separated words to listen for. Default: Eddy's command words.
        Pass "" for open dictation."""
        # Vosk is imported here, not at the top, so Eddy runs without it until listening is used.
        try:
            import vosk
        except ImportError as exc:
            raise RuntimeError("listening needs: pip install 'eddy-ai[voice]'") from exc
        model_path = model_path or default_model_path()
        if not model_path:
            raise RuntimeError("no Vosk speech model found. Download one from "
                               "https://alphacephei.com/vosk/models and pass --voice-model <folder>")
        vosk.SetLogLevel(-1)        # silence Vosk's own start-up chatter
        self._vosk = vosk
        self.samplerate = samplerate
        self.device = device
        self.model = vosk.Model(model_path)
        if vocabulary is None:
            from ..intents import VOCABULARY
            vocabulary = VOCABULARY
        # The grammar limits the recogniser to our words; "[unk]" lets it mark anything else
        # as unknown instead of forcing it onto the nearest command word.
        self.grammar = json.dumps([vocabulary, "[unk]"]) if vocabulary else None

    def _recognizer(self, rate: int):
        """A fresh recogniser for audio at this sample rate, with the grammar if there is one."""
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
        # The sound card calls the callback from its own thread with each half second of audio;
        # the queue hands those chunks to this thread.
        audio: "queue.Queue[bytes]" = queue.Queue()
        with sounddevice.RawInputStream(samplerate=self.samplerate, blocksize=8000, dtype="int16", channels=1,
                                        device=self.device, callback=lambda data, *_: audio.put(bytes(data))):
            while True:
                # AcceptWaveform returns True when the speaker has paused, i.e. a phrase is finished.
                if recognizer.AcceptWaveform(audio.get()):
                    text = json.loads(recognizer.Result()).get("text", "").strip()
                    if _clean(text):
                        yield text      # unknown sounds stay marked "[unk]"; see heard_command()

    def transcribe(self, wav_path: str) -> str:
        """Recognise speech in a 16-bit mono WAV file (for testing without a microphone)."""
        import wave
        with wave.open(wav_path, "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("need a 16-bit mono WAV file")
            recognizer = self._recognizer(wav.getframerate())
            heard = []
            # Feed the file through in chunks, collecting each finished phrase.
            while True:
                data = wav.readframes(4000)
                if not data:
                    break
                if recognizer.AcceptWaveform(data):
                    heard.append(json.loads(recognizer.Result()).get("text", ""))
            heard.append(json.loads(recognizer.FinalResult()).get("text", ""))     # whatever was left at the end
        return " ".join(" ".join(heard).split())    # unknown sounds stay marked "[unk]"


def _clean(text: str) -> str:
    """Drop the recogniser's marker for sounds that were not one of our words."""
    return " ".join(w for w in text.split() if w != "[unk]")


UNKNOWN = "[unk]"       # the recogniser's marker for a sound that was not one of Eddy's words


def heard_command(phrase: str, wake_word: Optional[str] = "eddy"):
    """Decide what to do with a phrase from the recogniser. Returns (sentence, unclear).

    sentence is None when the phrase was not addressed to Eddy. unclear is True when part of
    the command was a sound Eddy does not know: the words it did catch might be the opposite
    of what was said ("don't ... drive forward"), so such a command must not be carried out.
    A sentence containing "stop" is never unclear: stopping is always safe.
    """
    words = phrase.lower().split()
    if wake_word:
        variants = {wake_word, "eddie", "eddy", "edi", "eddi", "eddies"} if wake_word == "eddy" else {wake_word}
        hits = [i for i, w in enumerate(words) if w in variants]
        if not hits:
            return None, False
        words = words[hits[0] + 1:]
    if not _clean(" ".join(words)):
        return None, False
    if any(w in ("stop", "halt", "freeze") for w in words):
        return "stop", False
    return _clean(" ".join(words)), UNKNOWN in words


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
            return rest or None     # the wake word alone is not a command
    return None                     # no wake word: not addressed to Eddy
