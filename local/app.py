"""
AI Video Agent — Local Dashboard Server
────────────────────────────────────────
FastAPI server that:
  • Serves the dashboard UI
  • Manages browser WebSocket connections
  • Coordinates Gemini Live ↔ Cartesia TTS ↔ GPU pipeline
  • Tracks real-time service status
"""

import asyncio
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import httpx
import websockets
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── Load env ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env", override=True)

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
CARTESIA_API_KEY    = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_VOICE_ID   = os.getenv("CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091")
LIVEKIT_URL         = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY     = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET  = os.getenv("LIVEKIT_API_SECRET", "")
GPU_WS_URL          = os.getenv("GPU_WS_URL", "")
GPU_API_URL         = os.getenv("GPU_API_URL", "")
APP_HOST            = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT            = int(os.getenv("APP_PORT", "8000"))
AGENT_NAME          = os.getenv("AGENT_NAME", "Mr. Brain")
PERSONA_IMAGE_PATH  = os.getenv("PERSONA_IMAGE_PATH", "../persona/persona.jpg")

# ── Paths ─────────────────────────────────────────────────────────────────────
STATIC_DIR    = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Add local dir to path so imports work
sys.path.insert(0, str(BASE_DIR))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("avatar-agent")

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="AI Video Agent", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ── Service Status ────────────────────────────────────────────────────────────
service_status = {
    "dashboard":      {"status": "online",  "label": "Dashboard",      "side": "local"},
    "gemini":         {"status": "offline", "label": "Gemini Live",    "side": "local"},
    "cartesia":       {"status": "offline", "label": "Cartesia TTS",   "side": "local"},
    "livekit_client": {"status": "offline", "label": "LiveKit Client", "side": "local"},
    "gpu_server":     {"status": "offline", "label": "GPU FastAPI",    "side": "gpu"},
    "musetalk":       {"status": "offline", "label": "MuseTalk",       "side": "gpu"},
    "emage":          {"status": "offline", "label": "EMAGE",          "side": "gpu"},
    "livekit_server": {"status": "offline", "label": "LiveKit Server", "side": "gpu"},
}

# ── Connected Browser Clients ─────────────────────────────────────────────────
connected_clients: list[WebSocket] = []


async def broadcast(message: dict):
    """Broadcast a JSON message to all connected browser clients."""
    data = json.dumps(message)
    dead = []
    for ws in connected_clients:
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients.remove(ws)


async def set_status(service: str, status: str):
    """Update one service's status and broadcast to all clients."""
    if service in service_status:
        service_status[service]["status"] = status
    await broadcast({"type": "status", "services": service_status})


async def check_gpu_health_loop():
    """Periodically queries the GPU server health/telemetry."""
    while True:
        if GPU_API_URL:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get(f"{GPU_API_URL}/status")
                    if response.status_code == 200:
                        await set_status("gpu_server", "online")
                        await set_status("musetalk", "online")
                        await set_status("emage", "online")
                        await set_status("livekit_server", "online")
                    else:
                        await set_status("gpu_server", "offline")
                        await set_status("musetalk", "offline")
                        await set_status("emage", "offline")
                        await set_status("livekit_server", "offline")
            except Exception:
                await set_status("gpu_server", "offline")
                await set_status("musetalk", "offline")
                await set_status("emage", "offline")
                await set_status("livekit_server", "offline")
        await asyncio.sleep(5.0)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {
        "request":       request,
        "agent_name":    AGENT_NAME,
        "has_gemini":    bool(GEMINI_API_KEY),
        "has_cartesia":  bool(CARTESIA_API_KEY),
        "has_livekit":   bool(LIVEKIT_URL),
        "has_gpu":       bool(GPU_WS_URL),
    })


@app.get("/api/status")
async def get_status():
    return JSONResponse({"services": service_status})


@app.get("/api/config")
async def get_config():
    return JSONResponse({
        "agent_name":  AGENT_NAME,
        "has_gemini":  bool(GEMINI_API_KEY),
        "has_cartesia": bool(CARTESIA_API_KEY),
        "has_livekit": bool(LIVEKIT_URL),
        "has_gpu":     bool(GPU_WS_URL),
        "livekit_url": LIVEKIT_URL or None,
    })


@app.get("/api/persona")
async def get_persona():
    """Serve the persona image or a default SVG placeholder."""
    p = Path(PERSONA_IMAGE_PATH)
    if not p.is_absolute():
        p = BASE_DIR.parent / PERSONA_IMAGE_PATH.lstrip("../")
    if p.exists():
        return FileResponse(str(p))
    # Default SVG avatar
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <defs>
    <radialGradient id="bg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" style="stop-color:#1a1a3e"/>
      <stop offset="100%" style="stop-color:#0a0a1a"/>
    </radialGradient>
    <radialGradient id="face" cx="50%" cy="40%" r="50%">
      <stop offset="0%" style="stop-color:#3a3a6a"/>
      <stop offset="100%" style="stop-color:#1e1e4a"/>
    </radialGradient>
  </defs>
  <rect width="512" height="512" fill="url(#bg)"/>
  <circle cx="256" cy="200" r="100" fill="url(#face)" stroke="#6366f1" stroke-width="2"/>
  <circle cx="256" cy="420" r="180" fill="url(#face)" stroke="#6366f1" stroke-width="1"/>
  <circle cx="220" cy="185" r="12" fill="#6366f1" opacity="0.8"/>
  <circle cx="292" cy="185" r="12" fill="#6366f1" opacity="0.8"/>
  <path d="M 226 230 Q 256 250 286 230" stroke="#8b5cf6" stroke-width="3" fill="none" stroke-linecap="round"/>
  <text x="256" y="490" text-anchor="middle" fill="#6366f1" font-size="14" font-family="Arial" opacity="0.6">Place persona.jpg in /persona/</text>
</svg>"""
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/livekit-token")
async def get_livekit_token():
    """Generate a viewer token for the browser to join the LiveKit room."""
    if not (LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
        return JSONResponse({"error": "LiveKit not configured"}, status_code=503)
    try:
        from streaming.livekit_client import LiveKitTokens
        lk = LiveKitTokens(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token = lk.generate_viewer_token()
        return JSONResponse({"token": token, "url": LIVEKIT_URL})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "online", "service": "AI Video Agent", "version": "1.0.0"}


# ── Conversation WebSocket ────────────────────────────────────────────────────

@app.websocket("/ws/conversation")
async def conversation_ws(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info("🔌 Browser client connected")

    # Send current state immediately
    await websocket.send_text(json.dumps({
        "type": "status", "services": service_status
    }))
    await websocket.send_text(json.dumps({
        "type": "config",
        "data": {
            "agent_name":  AGENT_NAME,
            "has_gemini":  bool(GEMINI_API_KEY),
            "has_cartesia": bool(CARTESIA_API_KEY),
            "has_gpu":     bool(GPU_WS_URL),
        }
    }))

    # Engine instances (per connection)
    gemini: Optional[object] = None
    cartesia: Optional[object] = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def on_ai_text(text: str):
        """Gemini produced a text response."""
        await websocket.send_text(json.dumps({"type": "ai_text", "text": text}))
        # Send to Cartesia TTS if available
        if cartesia and text.strip():
            try:
                audio_b64 = await cartesia.synthesize(text)
                await websocket.send_text(json.dumps({
                    "type": "ai_audio",
                    "audio": audio_b64,
                    "format": "mp3",
                }))
            except Exception as e:
                logger.error(f"TTS error: {e}")

    async def on_ai_audio(audio_bytes: bytes):
        """Gemini produced raw PCM audio response."""
        audio_b64 = base64.b64encode(audio_bytes).decode()
        await websocket.send_text(json.dumps({
            "type": "ai_audio",
            "audio": audio_b64,
            "format": "pcm",
            "sampleRate": 24000,
        }))
        if gpu_ws:
            try:
                await gpu_ws.send(audio_bytes)
            except Exception as e:
                logger.error(f"Error forwarding audio to GPU: {e}")

    async def on_transcript(text: str, is_final: bool):
        """Gemini transcribed what the user said."""
        await websocket.send_text(json.dumps({
            "type": "transcript",
            "text": text,
            "final": is_final,
        }))

    # ── Message Loop ──────────────────────────────────────────────────────────

    try:
        from conversation.gemini_live import GeminiLiveClient
        from tts.cartesia_tts import CartesiaTTS

        if GEMINI_API_KEY:
            gemini = GeminiLiveClient(api_key=GEMINI_API_KEY, agent_name=AGENT_NAME)
        if CARTESIA_API_KEY:
            cartesia = CartesiaTTS(api_key=CARTESIA_API_KEY, voice_id=CARTESIA_VOICE_ID)

        gpu_ws = None

        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            # ── Start conversation ─────────────────────────────────────────
            if msg_type == "start_conversation":
                if GPU_WS_URL:
                    try:
                        from streaming.livekit_client import LiveKitTokens
                        lk = LiveKitTokens(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
                        gpu_token = lk.generate_publisher_token()
                        logger.info(f"Connecting to GPU websocket at {GPU_WS_URL}...")
                        gpu_ws = await websockets.connect(GPU_WS_URL)
                        handshake = {
                            "livekit_url": LIVEKIT_URL,
                            "livekit_token": gpu_token
                        }
                        await gpu_ws.send(json.dumps(handshake))
                        logger.info("✓ Connected and handshake sent to GPU server")
                    except Exception as e:
                        logger.error(f"Failed to connect to GPU WebSocket: {e}")
                if gemini:
                    await set_status("gemini", "connecting")
                    try:
                        modalities = msg.get("modalities", ["AUDIO"])
                        await gemini.connect(
                            on_text=on_ai_text,
                            on_audio=on_ai_audio,
                            on_transcript=on_transcript,
                            modalities=modalities,
                        )
                        await set_status("gemini", "online")
                        if cartesia:
                            await set_status("cartesia", "online")
                        logger.info("✓ Conversation started")
                        await websocket.send_text(json.dumps({
                            "type": "conversation_started"
                        }))
                    except Exception as e:
                        logger.error(f"Gemini connect error: {e}")
                        await set_status("gemini", "error")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"Failed to connect to Gemini: {e}"
                        }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "info",
                        "message": "🎭 Demo Mode — Add GEMINI_API_KEY to .env for live AI conversation."
                    }))
                    await websocket.send_text(json.dumps({"type": "conversation_started"}))

            # ── Audio chunk from mic ───────────────────────────────────────
            elif msg_type == "audio_chunk":
                if gemini and getattr(gemini, "is_connected", False):
                    audio_data = base64.b64decode(msg["data"])
                    await gemini.send_audio(audio_data)

            # ── Text message ──────────────────────────────────────────────
            elif msg_type == "text_message":
                text = msg.get("text", "").strip()
                if text:
                    if gemini and getattr(gemini, "is_connected", False):
                        await gemini.send_text(text)
                    else:
                        # Demo mode response
                        demo = (
                            f"👋 Demo Mode: You said \"{text}\". "
                            "Add your GEMINI_API_KEY to .env for real AI responses!"
                        )
                        await websocket.send_text(json.dumps({
                            "type": "ai_text", "text": demo
                        }))

            # ── Stop conversation ──────────────────────────────────────────
            elif msg_type == "stop_conversation":
                if gemini and getattr(gemini, "is_connected", False):
                    await gemini.disconnect()
                    await set_status("gemini", "offline")
                    await set_status("cartesia", "offline")
                if gpu_ws:
                    try:
                        await gpu_ws.close()
                    except Exception:
                        pass
                    gpu_ws = None
                logger.info("Conversation stopped")
                await websocket.send_text(json.dumps({"type": "conversation_stopped"}))

            # ── Sync active module context ─────────────────────────────────
            elif msg_type == "sync_module":
                module_id = msg.get("module", "")
                logger.info(f"Operator active module synced: {module_id}")
                if gemini and getattr(gemini, "is_connected", False):
                    # Inject module shift system context
                    context_msg = f"[SYSTEM: Operator switched view to module '{module_id}'. Provide operational intelligence regarding this context if asked.]"
                    await gemini.send_text(context_msg)

            # ── Ping ──────────────────────────────────────────────────────
            elif msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup
        if gpu_ws:
            try:
                await gpu_ws.close()
            except Exception:
                pass
            gpu_ws = None
        if gemini and getattr(gemini, "is_connected", False):
            try:
                await gemini.disconnect()
            except Exception:
                pass
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.info("🔌 Browser client disconnected")


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    logger.info("━" * 55)
    logger.info(f"  AI Video Agent Dashboard  —  port {APP_PORT}")
    logger.info("━" * 55)
    logger.info(f"  Gemini API  : {'✓ configured' if GEMINI_API_KEY else '✗ missing (demo mode)'}")
    logger.info(f"  Cartesia    : {'✓ configured' if CARTESIA_API_KEY else '✗ missing'}")
    logger.info(f"  LiveKit     : {'✓ configured' if LIVEKIT_URL else '✗ missing'}")
    logger.info(f"  GPU Server  : {'✓ configured' if GPU_WS_URL else '✗ not set (Phase 2)'}")
    logger.info("━" * 55)
    logger.info(f"  Open: http://localhost:{APP_PORT}")
    logger.info("━" * 55)
    if GPU_API_URL:
        asyncio.create_task(check_gpu_health_loop())


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=True,
        log_level="info",
    )
