"""
Cartesia Sonic TTS Client
──────────────────────────
Ultra-low-latency text-to-speech (~90ms).
Supports both one-shot synthesis and streaming chunk-by-chunk output.
"""

import asyncio
import base64
import logging
import threading
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)

# Default voice: "Helpful Woman" from Cartesia
DEFAULT_VOICE_ID = "a0e99841-438c-4a64-b679-ae501e7d6091"
CARTESIA_MODEL = "sonic-2"


class CartesiaTTS:
    def __init__(self, api_key: str, voice_id: str = DEFAULT_VOICE_ID):
        self.api_key = api_key
        self.voice_id = voice_id
        self._client = None

    def _get_client(self):
        """Lazy-initialize Cartesia client."""
        if not self._client:
            from cartesia import Cartesia
            self._client = Cartesia(api_key=self.api_key)
        return self._client

    async def synthesize(self, text: str) -> str:
        """
        Synthesize full text to audio.
        Returns base64-encoded MP3 string for browser playback.
        """
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(
            None, self._synthesize_sync, text
        )
        return base64.b64encode(audio_bytes).decode()

    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronous synthesis, runs in thread executor."""
        client = self._get_client()
        chunks = []
        try:
            for chunk in client.tts.sse(
                model_id=CARTESIA_MODEL,
                transcript=text,
                voice={"id": self.voice_id},
                output_format={
                    "container": "mp3",
                    "encoding": "mp3",
                    "sample_rate": 44100,
                },
                stream=True,
            ):
                if hasattr(chunk, "audio") and chunk.audio:
                    chunks.append(chunk.audio)
        except Exception as e:
            logger.error(f"Cartesia synthesis error: {e}")
        return b"".join(chunks)

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """
        Async generator that yields raw PCM audio chunks as they stream.
        Use this for lowest-latency output (first chunk < 90ms).
        """
        client = self._get_client()
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _stream():
            try:
                for chunk in client.tts.sse(
                    model_id=CARTESIA_MODEL,
                    transcript=text,
                    voice={"id": self.voice_id},
                    output_format={
                        "container": "raw",
                        "encoding": "pcm_f32le",
                        "sample_rate": 44100,
                    },
                    stream=True,
                ):
                    if hasattr(chunk, "audio") and chunk.audio:
                        asyncio.run_coroutine_threadsafe(
                            queue.put(chunk.audio), loop
                        )
            except Exception as e:
                logger.error(f"Cartesia stream error: {e}")
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        thread = threading.Thread(target=_stream, daemon=True)
        thread.start()

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
