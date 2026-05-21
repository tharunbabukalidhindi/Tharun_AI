"""
Gemini Live API Client
─────────────────────
Handles real-time speech-to-speech conversation with Gemini 2.0 Flash Live.
Sends mic audio → receives AI audio + text response in real-time.
"""

import asyncio
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an intelligent AI assistant with a warm, expressive personality.
You are represented as an animated avatar — speak naturally and conversationally.
Keep responses concise (1-3 sentences) unless the user asks for detailed explanations.
Be helpful, engaging, and human-like in your responses."""

GEMINI_LIVE_MODEL = "gemini-2.5-flash-native-audio-latest"


class GeminiLiveClient:
    def __init__(self, api_key: str, agent_name: str = "AI Agent"):
        self.api_key = api_key
        self.agent_name = agent_name
        self.is_connected = False
        self._session = None
        self._client = None
        self._ctx = None
        self._receive_task: Optional[asyncio.Task] = None
        self._on_text: Optional[Callable] = None
        self._on_audio: Optional[Callable] = None
        self._on_transcript: Optional[Callable] = None

    async def connect(
        self,
        on_text: Callable,
        on_audio: Callable,
        on_transcript: Callable,
    ):
        """Connect to Gemini Live and start background receive loop."""
        from google import genai
        from google.genai import types

        self._on_text = on_text
        self._on_audio = on_audio
        self._on_transcript = on_transcript

        self._client = genai.Client(api_key=self.api_key)

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=SYSTEM_PROMPT,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Puck"
                    )
                )
            ),
        )

        self._ctx = self._client.aio.live.connect(
            model=GEMINI_LIVE_MODEL,
            config=config,
        )
        self._session = await self._ctx.__aenter__()
        self.is_connected = True

        # Start background receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info("✓ Gemini Live session connected")

    async def _receive_loop(self):
        """Background loop that receives AI responses from Gemini."""
        try:
            while self.is_connected and self._session:
                turn = self._session.receive()
                async for response in turn:
                    await self._handle_response(response)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.is_connected:
                logger.error(f"Gemini receive error: {e}")

    async def _handle_response(self, response):
        """Parse and dispatch a response chunk from Gemini."""
        try:
            # Top-level audio bytes
            if hasattr(response, "data") and response.data:
                await self._on_audio(response.data)

            # Server content (structured parts)
            if hasattr(response, "server_content") and response.server_content:
                sc = response.server_content

                # Model turn parts (text + audio)
                if hasattr(sc, "model_turn") and sc.model_turn:
                    for part in sc.model_turn.parts:
                        # Skip internal thought/thinking process parts
                        if getattr(part, "thought", False):
                            continue
                        if hasattr(part, "text") and part.text:
                            await self._on_text(part.text)
                        if hasattr(part, "inline_data") and part.inline_data:
                            await self._on_audio(part.inline_data.data)

            # Fallback to top-level text only if there is no structured server content
            else:
                if hasattr(response, "text") and response.text:
                    await self._on_text(response.text)

                # Input transcription (what the user said)
                if hasattr(sc, "input_transcription") and sc.input_transcription:
                    t = sc.input_transcription
                    if hasattr(t, "text") and t.text:
                        is_final = getattr(t, "is_final", True)
                        await self._on_transcript(t.text, is_final)

        except Exception as e:
            logger.error(f"Error handling Gemini response: {e}")

    async def send_audio(self, pcm_bytes: bytes):
        """
        Send raw PCM audio chunk to Gemini.
        Expected format: 16kHz, 16-bit, mono (LINEAR16)
        """
        if not self.is_connected or not self._session:
            return
        try:
            from google.genai import types
            await self._session.send(
                input=types.LiveClientRealtimeInput(
                    media_chunks=[
                        types.Blob(
                            data=pcm_bytes,
                            mime_type="audio/pcm;rate=16000"
                        )
                    ]
                )
            )
        except Exception as e:
            logger.error(f"Error sending audio to Gemini: {e}")

    async def send_text(self, text: str):
        """Send a text message to Gemini as user input."""
        if not self.is_connected or not self._session:
            return
        try:
            await self._session.send(input=text, end_of_turn=True)
        except Exception as e:
            logger.error(f"Error sending text to Gemini: {e}")

    async def disconnect(self):
        """Cleanly disconnect from Gemini Live."""
        self.is_connected = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._ctx and self._session:
            try:
                await self._ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self._session = None
        self._ctx = None
        logger.info("Gemini Live session disconnected")
