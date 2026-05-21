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
    "dashboard":      {"status": "online",  "label": "Dashboard",      "side": "local"},
    "gemini":         {"status": "offline", "label": "Gemini API",     "side": "local"},
    "cartesia":       {"status": "offline", "label": "Cartesia TTS",   "side": "local"},
    "gpu_server":     {"status": "offline", "label": "GPU Server",     "side": "gpu"},
    "musetalk":       {"status": "offline", "label": "MuseTalk",       "side": "gpu"},
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
                    else:
                        await set_status("gpu_server", "offline")
                        await set_status("musetalk", "offline")
            except Exception:
                await set_status("gpu_server", "offline")
                await set_status("musetalk", "offline")
        await asyncio.sleep(5.0)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {
        "request":       request,
        "agent_name":    AGENT_NAME,
        "has_gemini":    bool(GEMINI_API_KEY),
        "has_cartesia":  bool(CARTESIA_API_KEY),
        "has_gpu":       bool(GPU_API_URL),
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
        "has_gpu":     bool(GPU_API_URL),
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



@app.get("/health")
async def health():
    return {"status": "online", "service": "AI Video Agent", "version": "1.0.0"}


@app.get("/api/gpu-video-url")
async def gpu_video_url():
    """Returns the local MJPEG proxy URL (browser-accessible, no auth needed)."""
    if GPU_API_URL:
        return JSONResponse({"url": "/api/avatar-stream"})
    return JSONResponse({"url": None})


@app.get("/api/avatar-stream")
async def avatar_stream():
    """Proxies the GPU MJPEG stream through localhost so the browser has no auth issues."""
    if not GPU_API_URL:
        return JSONResponse({"error": "no GPU configured"}, status_code=503)

    async def proxy():
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("GET", f"{GPU_API_URL}/video") as resp:
                    async for chunk in resp.aiter_bytes(4096):
                        yield chunk
            except Exception as e:
                logger.debug(f"MJPEG proxy ended: {e}")

    return StreamingResponse(
        proxy(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


# ── Conversation WebSocket ────────────────────────────────────────────────────

@app.websocket("/ws/conversation")
async def conversation_ws(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info("🔌 Browser client connected")

    # Engine instances (per connection)
    cartesia = None
    gpu_http = None

    try:
        from tts.cartesia_tts import CartesiaTTS
        from google import genai
        
        if GEMINI_API_KEY:
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            gemini_client = None

        if CARTESIA_API_KEY:
            cartesia = CartesiaTTS(api_key=CARTESIA_API_KEY, voice_id=CARTESIA_VOICE_ID)

        if GPU_API_URL:
            gpu_http = httpx.AsyncClient(base_url=GPU_API_URL, timeout=30.0)

        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            # ── Start conversation ─────────────────────────────────────────
            if msg_type == "start_conversation":
                if gemini_client:
                    await set_status("gemini", "online")
                if cartesia:
                    await set_status("cartesia", "online")
                if gpu_http:
                    await set_status("gpu_server", "online")
                
                await websocket.send_text(json.dumps({"type": "conversation_started"}))
                logger.info("✓ Conversation started")

            # ── User Text Input (from Web Speech API or typing) ─────────────
            elif msg_type == "text_message" or msg_type == "user_text":
                user_text = msg.get("text", "").strip()
                if not user_text:
                    continue
                    
                if not gemini_client:
                    demo = f"👋 Demo Mode: You said \"{user_text}\". Add GEMINI_API_KEY for AI responses!"
                    await websocket.send_text(json.dumps({"type": "ai_text", "text": demo}))
                    continue
                
                logger.info(f"User: {user_text}")
                
                # 1. Call Gemini for text response
                try:
                    response = gemini_client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=user_text,
                        config=genai.types.GenerateContentConfig(
                            system_instruction="You are an intelligent AI assistant with a warm, expressive personality. Keep responses concise (1-3 sentences).",
                            temperature=0.7,
                        )
                    )
                    ai_text = response.text
                    logger.info(f"AI: {ai_text}")
                    await websocket.send_text(json.dumps({"type": "ai_text", "text": ai_text}))
                except Exception as e:
                    logger.error(f"Gemini error: {e}")
                    await set_status("gemini", "error")
                    await websocket.send_text(json.dumps({"type": "error", "message": f"Gemini API error: {e}"}))
                    continue

                # 2. Call Cartesia for TTS
                if cartesia:
                    try:
                        logger.info("Generating TTS...")
                        audio_b64 = await cartesia.synthesize(ai_text)
                        
                        # 3. Call GPU for rendering perfectly synced MP4
                        if gpu_http:
                            logger.info("Requesting GPU render...")
                            audio_bytes = base64.b64decode(audio_b64)
                            try:
                                gpu_resp = await gpu_http.post("/generate_video", content=audio_bytes)
                                if gpu_resp.status_code == 200:
                                    video_b64 = gpu_resp.json().get("video")
                                    if video_b64:
                                        await websocket.send_text(json.dumps({
                                            "type": "video_ready",
                                            "video": video_b64
                                        }))
                                        logger.info("✓ GPU Video generated and sent to browser")
                                else:
                                    logger.error(f"GPU render failed: {gpu_resp.text}")
                            except Exception as e:
                                logger.error(f"GPU connection error: {e}")
                        else:
                            # Fallback if GPU is down
                            await websocket.send_text(json.dumps({
                                "type": "ai_audio",
                                "audio": audio_b64,
                                "format": "mp3",
                            }))
                            
                    except Exception as e:
                        logger.error(f"TTS error: {e}")

            # ── Stop conversation ──────────────────────────────────────────
            elif msg_type == "stop_conversation":
                await set_status("gemini", "offline")
                await set_status("cartesia", "offline")
                logger.info("Conversation stopped by user")
                break
                
            elif msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        logger.info("Browser client disconnected normally")
    except Exception as e:
        logger.error(f"Conversation error: {e}", exc_info=True)
    finally:
        if gpu_http:
            await gpu_http.aclose()
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.info("🔌 Cleaned up connection resources")


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    logger.info("━" * 55)
    logger.info(f"  AI Video Agent Dashboard  —  port {APP_PORT}")
    logger.info("━" * 55)
    logger.info(f"  Gemini API  : {'✓ configured' if GEMINI_API_KEY else '✗ missing (demo mode)'}")
    logger.info(f"  Cartesia    : {'✓ configured' if CARTESIA_API_KEY else '✗ missing'}")
    logger.info(f"  GPU Server  : {'✓ configured' if GPU_WS_URL else '✗ not set'}")
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
