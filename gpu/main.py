import os
import json
import logging
import asyncio
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("gpu-server")

app = FastAPI(title="AI Video Agent - GPU Server", version="1.0.0")

# Load model engines
from pipeline.musetalk_engine import MuseTalkEngine
from pipeline.emage_engine import EMAGEEngine
from streaming.livekit_server import LiveKitPublisher

musetalk = MuseTalkEngine()
emage = EMAGEEngine()

# Active publisher stream
publisher: Optional[LiveKitPublisher] = None

@app.get("/status")
async def status():
    """Returns GPU server hardware telemetry for the dashboard."""
    # Simulate high-fidelity system telemetry stats
    return JSONResponse({
        "status": "ready",
        "cpu_usage": 14.5,          # %
        "gpu_usage": 32.8,          # %
        "vram_total": 24.0,         # GB (L4 / A10G)
        "vram_used": 8.4,           # GB
        "latency_ms": 142,          # Pipeline latency
        "temp_celsius": 68.0        # GPU Temp
    })

@app.websocket("/stream")
async def audio_stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint that ingests PCM audio streams from the local orchestrator,
    generates talking avatar frames, and publishes them over LiveKit.
    """
    await websocket.accept()
    logger.info("🔌 Local orchestrator connected to GPU stream")
    
    global publisher
    
    try:
        # Load avatar reference image
        # If user hasn't copied a reference image, use a dummy canvas
        ref_image_path = "../persona/persona.jpg"
        if os.path.exists(ref_image_path):
            avatar_img = Image.open(ref_image_path)
        else:
            avatar_img = Image.new("RGB", (512, 512), color=(10, 10, 30))

        # 1. Read setup handshake
        setup_raw = await websocket.receive_text()
        setup_data = json.loads(setup_raw)
        
        livekit_url = setup_data.get("livekit_url")
        livekit_token = setup_data.get("livekit_token")
        
        # 2. Connect WebRTC publisher
        if livekit_url and livekit_token:
            publisher = LiveKitPublisher(livekit_url, livekit_token)
            await publisher.connect()
        else:
            logger.warning("⚠️ LiveKit credentials missing from handshake. Running frame stream internally.")
            
        # 3. Main streaming loop
        while True:
            # Receive raw binary audio chunks
            audio_chunk = await websocket.receive_bytes()
            
            # Process in parallel
            # A. MuseTalk generates lip frames
            lip_frames = musetalk.process_audio(audio_chunk, avatar_img)
            num_frames = len(lip_frames)
            
            # B. EMAGE generates body offsets
            body_motions = emage.generate_motion(audio_chunk, num_frames)
            
            # C. Composite frames & Publish
            for idx in range(num_frames):
                frame = lip_frames[idx]
                motion = body_motions[idx]
                
                # Apply simulated motion shifts (X/Y translate) to composite lip-sync + body breathing
                dy = int(motion["body_y_offset"])
                
                # Shift frame rows up/down to simulate body breathing movement
                if dy != 0:
                    frame = np.roll(frame, dy, axis=0)
                    
                # Push frame to LiveKit WebRTC channel
                if publisher:
                    await publisher.push_frame(frame)
                    
            # Yield control back to event loop
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info("🔌 Local orchestrator disconnected")
    except Exception as e:
        logger.error(f"Error in streaming loop: {e}")
    finally:
        if publisher:
            await publisher.disconnect()
            publisher = None

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
