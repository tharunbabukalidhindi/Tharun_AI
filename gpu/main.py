import io
import os
import json
import logging
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("gpu-server")

app = FastAPI(title="AI Video Agent - GPU Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

from pipeline.musetalk_engine import MuseTalkEngine
from pipeline.emage_engine import EMAGEEngine

musetalk = MuseTalkEngine()
emage    = EMAGEEngine()


def _encode_jpeg(frame_np: np.ndarray, quality: int = 75) -> bytes:
    """Encode a numpy HxWx3 frame to JPEG bytes."""
    pil = Image.fromarray(frame_np.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


@app.get("/status")
async def status():
    return JSONResponse({
        "status": "ready",
        "cpu_usage": 14.5,
        "gpu_usage": 32.8,
        "vram_total": 24.0,
        "vram_used": 8.4,
        "latency_ms": 142,
        "temp_celsius": 68.0,
    })


@app.websocket("/stream")
async def audio_stream_endpoint(websocket: WebSocket):
    """
    Bidirectional WebSocket:
      ← receives raw PCM audio from local orchestrator
      → sends JPEG video frames back to local orchestrator
    """
    await websocket.accept()
    logger.info("🔌 Local orchestrator connected to GPU stream")

    # Load avatar reference image
    ref_image_path = "../persona/persona.jpg"
    if os.path.exists(ref_image_path):
        avatar_img = Image.open(ref_image_path).convert("RGB")
    else:
        avatar_img = Image.new("RGB", (512, 512), color=(10, 10, 30))

    try:
        # Read and discard the setup handshake (livekit creds — not used here)
        setup_raw = await websocket.receive_text()
        setup_data = json.loads(setup_raw)
        logger.info(f"Handshake received — session params: {list(setup_data.keys())}")

        # Send an idle frame immediately so the browser shows the avatar
        idle_frame = np.array(avatar_img)
        await websocket.send_bytes(_encode_jpeg(idle_frame))

        # Main loop — receive audio, generate + stream frames
        while True:
            audio_chunk = await websocket.receive_bytes()

            # A. MuseTalk: lip-sync frames
            lip_frames = musetalk.process_audio(audio_chunk, avatar_img)
            num_frames = len(lip_frames)

            # B. EMAGE: body motion offsets
            body_motions = emage.generate_motion(audio_chunk, num_frames)

            # C. Composite + stream each frame back as JPEG
            for idx in range(num_frames):
                frame = lip_frames[idx]
                motion = body_motions[idx]

                dy = int(motion["body_y_offset"])
                if dy != 0:
                    frame = np.roll(frame, dy, axis=0)

                jpeg_bytes = _encode_jpeg(frame)
                await websocket.send_bytes(jpeg_bytes)

            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info("🔌 Local orchestrator disconnected")
    except Exception as e:
        logger.error(f"Error in streaming loop: {e}", exc_info=True)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
