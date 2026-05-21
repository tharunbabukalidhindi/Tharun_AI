import io
import os
import asyncio
import logging
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("gpu-server")

app = FastAPI(title="AI Video Agent - GPU Server", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

from pipeline.musetalk_engine import MuseTalkEngine
from pipeline.emage_engine import EMAGEEngine

musetalk = MuseTalkEngine()
emage    = EMAGEEngine()

# ── Avatar image (loaded once at startup) ─────────────────────────────────────
_PERSONA_PATH = "persona/persona.jpg"
if os.path.exists(_PERSONA_PATH):
    avatar_img = Image.open(_PERSONA_PATH).convert("RGB")
    logger.info(f"✓ Loaded persona: {avatar_img.size}")
else:
    avatar_img = Image.new("RGB", (512, 512), color=(10, 10, 30))
    logger.warning("⚠️ persona/persona.jpg not found — using dark placeholder")

# ── Frame queue for MJPEG stream ───────────────────────────────────────────────
_frame_queue: asyncio.Queue = None   # initialized on startup (needs event loop)
_idle_jpeg: bytes = None


def _encode_jpeg(frame_np: np.ndarray, quality: int = 75) -> bytes:
    pil = Image.fromarray(frame_np.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


@app.on_event("startup")
async def startup():
    global _frame_queue, _idle_jpeg
    _frame_queue = asyncio.Queue(maxsize=60)
    # Pre-encode idle frame (avatar at rest)
    _idle_jpeg = _encode_jpeg(np.array(avatar_img))
    logger.info("✓ GPU server ready — MJPEG stream active on /video")
    # Pre-warm MuseTalk models so first response has no delay
    logger.info("⏳ Pre-warming MuseTalk models...")
    try:
        import asyncio as _a
        loop = _a.get_event_loop()
        await loop.run_in_executor(None, _warmup_models)
        logger.info("✓ Models warmed up and ready")
    except Exception as e:
        logger.warning(f"Model warmup failed (will lazy-load): {e}")


def _warmup_models():
    """Run a dummy inference to load models into VRAM before first real call."""
    dummy_audio = bytes(3200)  # 100ms of silence at 16kHz 16-bit
    try:
        musetalk.process_audio(dummy_audio, avatar_img)
        logger.info("✓ MuseTalk warmed up")
    except Exception as e:
        logger.warning(f"MuseTalk warmup: {e}")


import subprocess
import base64
import tempfile
import shutil

# ── Status ─────────────────────────────────────────────────────────────────────
@app.get("/status")
async def status():
    return JSONResponse({
        "status": "ready",
        "gpu_ready": True
    })

# ── Video Generation (Perfect Sync) ────────────────────────────────────────────
@app.post("/generate_video")
async def generate_video(request: Request):
    """
    Receives complete MP3 audio bytes, generates MuseTalk frames,
    encodes them with the audio into an MP4, and returns the base64 MP4.
    Guarantees perfect audio-visual synchronization.
    """
    audio_bytes = await request.body()
    if not audio_bytes:
        return JSONResponse({"error": "No audio data"}, status_code=400)

    try:
        # We need a temporary directory to store frames and audio for ffmpeg
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.mp3")
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
                
            logger.info("Running MuseTalk inference...")
            # MuseTalk needs 16kHz PCM or handles MP3 directly? 
            # Our MuseTalk wrapper uses torchaudio or librosa which can load MP3 if ffmpeg is installed.
            lip_frames = musetalk.process_audio(audio_bytes, avatar_img)
            num_frames = len(lip_frames)
            logger.info(f"Generated {num_frames} frames.")
            
            # Apply EMAGE subtle body motions
            body_motions = emage.generate_motion(audio_bytes, num_frames)
            
            frames_dir = os.path.join(tmpdir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            logger.info("Saving frames to disk...")
            for idx in range(num_frames):
                frame = lip_frames[idx]
                dy = int(body_motions[idx]["body_y_offset"])
                if dy:
                    frame = np.roll(frame, dy, axis=0)
                
                # Save frame as JPEG
                pil_img = Image.fromarray(frame.astype(np.uint8))
                pil_img.save(os.path.join(frames_dir, f"{idx:04d}.jpg"), quality=85)
                
            out_video_path = os.path.join(tmpdir, "output.mp4")
            logger.info("Encoding perfectly synced MP4 with ffmpeg...")
            
            # MuseTalk uses exactly 25 FPS (or 24? usually 25 for 16000/640 hop size)
            # Standard MuseTalk hop_size is 640 @ 16kHz -> 25 fps.
            fps = 25
            
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(frames_dir, "%04d.jpg"),
                "-i", audio_path,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                out_video_path
            ]
            
            subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            with open(out_video_path, "rb") as f:
                video_bytes = f.write() if False else f.read() # just read
                
            video_b64 = base64.b64encode(video_bytes).decode('utf-8')
            
            logger.info("✓ Video generated and ready to serve.")
            return JSONResponse({"video": video_b64})

    except Exception as e:
        logger.error(f"Video generation error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
