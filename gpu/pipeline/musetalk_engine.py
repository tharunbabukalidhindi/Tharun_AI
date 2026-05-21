import os
import io
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger("musetalk-engine")

# ── Lazy imports (only loaded when weights exist) ────────────────────────────
_models_loaded = False
_whisper_processor = None
_whisper_model = None
_unet = None
_vae = None
_face_mesh = None


def _lazy_load_models(checkpoints_dir: str):
    """Load all models into GPU memory on first inference call."""
    global _models_loaded, _whisper_processor, _whisper_model, _unet, _vae, _face_mesh
    if _models_loaded:
        return

    import torch
    from transformers import WhisperProcessor, WhisperModel
    from diffusers import AutoencoderKL, UNet2DConditionModel
    import mediapipe as mp

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading MuseTalk models on {device}...")

    # 1. Whisper tiny encoder
    whisper_dir = os.path.join(checkpoints_dir, "whisper")
    logger.info("Loading Whisper encoder...")
    _whisper_processor = WhisperProcessor.from_pretrained(whisper_dir)
    _whisper_model = WhisperModel.from_pretrained(whisper_dir).to(device)
    _whisper_model.eval()

    # 2. VAE (Stable Diffusion VAE ft-mse)
    vae_dir = os.path.join(checkpoints_dir, "musetalk/sd-vae-ft-mse")
    logger.info("Loading VAE...")
    _vae = AutoencoderKL.from_pretrained(vae_dir).to(device)
    _vae.eval()

    # 3. MuseTalk UNet
    unet_dir = os.path.join(checkpoints_dir, "musetalk/musetalk")
    logger.info("Loading MuseTalk UNet...")
    _unet = UNet2DConditionModel.from_pretrained(unet_dir).to(device)
    _unet.eval()

    # 4. MediaPipe FaceMesh for landmark detection
    logger.info("Loading MediaPipe FaceMesh...")
    _face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    _models_loaded = True
    logger.info("✓ All MuseTalk models loaded successfully")


# Mouth landmark indices from MediaPipe FaceMesh 468-point model
# These are the outer lip contour points
MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
               291, 375, 321, 405, 314, 17, 84, 181, 91, 146]


def _get_mouth_bbox(landmarks, img_w, img_h, pad: int = 20):
    """Extract bounding box of mouth region with padding."""
    pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
           for i in MOUTH_OUTER]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(img_w, max(xs) + pad)
    y2 = min(img_h, max(ys) + pad)
    return x1, y1, x2, y2


def _extract_audio_features(pcm_bytes: bytes, sample_rate: int = 16000):
    """Convert raw PCM bytes → Whisper encoder hidden states."""
    import torch
    import numpy as np

    # PCM bytes → float32 array in [-1, 1]
    audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Pad or trim to 30s max (Whisper requires this)
    max_samples = 30 * sample_rate
    if len(audio_np) > max_samples:
        audio_np = audio_np[:max_samples]
    elif len(audio_np) < 400:
        # Too short — pad with silence
        audio_np = np.pad(audio_np, (0, 400 - len(audio_np)))

    device = next(_whisper_model.parameters()).device

    # Whisper feature extraction
    inputs = _whisper_processor(
        audio_np,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        # Use encoder only (we don't need the decoder for feature extraction)
        encoder_outputs = _whisper_model.encoder(input_features)
        # Shape: [1, seq_len, 384]
        audio_features = encoder_outputs.last_hidden_state

    return audio_features  # [1, T, 384]


def _run_musetalk_inference(mouth_crop_np: np.ndarray, audio_features) -> np.ndarray:
    """
    Run the MuseTalk UNet on a mouth crop + audio features.
    Returns a generated mouth patch as numpy array (H, W, 3).
    """
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    device = next(_unet.parameters()).device
    crop_h, crop_w = mouth_crop_np.shape[:2]

    # Resize to 256x256 for model (standard MuseTalk input size)
    MODEL_SIZE = 256
    to_tensor = transforms.Compose([
        transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    mouth_pil = Image.fromarray(mouth_crop_np)
    mouth_tensor = to_tensor(mouth_pil).unsqueeze(0).to(device)  # [1, 3, 256, 256]

    with torch.no_grad():
        # Encode mouth crop into latent space via VAE
        latent = _vae.encode(mouth_tensor).latent_dist.sample()
        latent = latent * 0.18215  # SD scaling factor

        # Create a masked version (zero out lower half = mask the mouth)
        masked_latent = latent.clone()
        masked_latent[:, :, latent.shape[2] // 2:, :] = 0.0

        # Concatenate noisy latent + masked latent = 8 channels input
        noise = torch.randn_like(latent)
        unet_input = torch.cat([noise, masked_latent], dim=1)  # [1, 8, H/8, W/8]

        # Timestep (use t=1 for single-step inference — fast mode)
        timestep = torch.tensor([1], device=device).long()

        # Audio cross-attention: pool temporal dim to match UNet seq_len expectation
        # UNet cross_attention_dim = 384, audio_features shape = [1, T, 384]
        # Trim/pad to 77 tokens (standard for cross-attention)
        T = audio_features.shape[1]
        if T > 77:
            audio_cond = audio_features[:, :77, :]
        else:
            padding = torch.zeros(1, 77 - T, audio_features.shape[2], device=device)
            audio_cond = torch.cat([audio_features, padding], dim=1)

        # UNet forward pass
        noise_pred = _unet(
            unet_input,
            timestep,
            encoder_hidden_states=audio_cond,
        ).sample  # [1, 4, H/8, W/8]

        # Decode back to pixel space via VAE
        noise_pred = noise_pred / 0.18215
        decoded = _vae.decode(noise_pred).sample  # [1, 3, 256, 256]
        decoded = (decoded.clamp(-1, 1) + 1) / 2  # → [0, 1]

    # Convert to numpy [H, W, 3] uint8
    decoded_np = decoded[0].permute(1, 2, 0).cpu().numpy()
    decoded_np = (decoded_np * 255).astype(np.uint8)

    # Resize back to original crop size
    result = np.array(Image.fromarray(decoded_np).resize((crop_w, crop_h)))
    return result


def _blend_mouth(frame: np.ndarray, mouth_patch: np.ndarray,
                 bbox: tuple) -> np.ndarray:
    """
    Blend the generated mouth patch back into the frame.
    Uses alpha blending with a smooth feathered mask.
    """
    import cv2
    x1, y1, x2, y2 = bbox
    result = frame.copy()
    patch_h, patch_w = y2 - y1, x2 - x1

    # Feathered mask — smooth edges to avoid hard boundaries
    mask = np.zeros((patch_h, patch_w), dtype=np.float32)
    cv2.ellipse(mask,
                (patch_w // 2, patch_h // 2),
                (patch_w // 2 - 2, patch_h // 2 - 2),
                0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask_3ch = np.stack([mask] * 3, axis=-1)

    # Blend
    roi = result[y1:y2, x1:x2].astype(np.float32)
    patch_f = mouth_patch.astype(np.float32)
    blended = roi * (1 - mask_3ch) + patch_f * mask_3ch
    result[y1:y2, x1:x2] = blended.astype(np.uint8)
    return result


class MuseTalkEngine:
    """
    Wraps the MuseTalk inference pipeline.
    Performs real-time lip-sync frame generation from PCM audio.
    Falls back to simulation mode if weights are not present.
    """
    def __init__(self, checkpoints_dir: str = "checkpoints"):
        self.checkpoints_dir = checkpoints_dir
        self.weights_exist = self._verify_weights()
        self._ref_landmarks = None  # cached from first frame

        if self.weights_exist:
            logger.info("✓ MuseTalk weights found. Initializing PyTorch models...")
            # Defer actual model loading to first inference call (saves startup time)
        else:
            logger.warning("⚠️ MuseTalk weights not found in checkpoints/ folder. Initializing in Simulation Mode.")

    def _verify_weights(self) -> bool:
        # HuggingFace snapshot_download puts files in a subdirectory matching the repo structure.
        # TMElyralab/MuseTalk downloads to checkpoints/musetalk/musetalk/
        # openai/whisper-tiny downloads to checkpoints/whisper/ with pytorch_model.bin
        candidate_sets = [
            # Original expected paths (flat layout)
            [
                "musetalk/musetalk.json",
                "musetalk/pytorch_model.bin",
                "whisper/tiny.pt",
                "musetalk/sd-vae-ft-mse/config.json",
            ],
            # HuggingFace snapshot layout (nested) — what we download
            [
                "musetalk/musetalk/musetalk.json",
                "musetalk/musetalk/pytorch_model.bin",
                "whisper/pytorch_model.bin",
                "musetalk/sd-vae-ft-mse/config.json",
            ],
        ]
        for candidate in candidate_sets:
            if all(os.path.exists(os.path.join(self.checkpoints_dir, p)) for p in candidate):
                logger.info(f"✓ Weights found at: {candidate[0]}")
                return True
        # Check without VAE (VAE might still be downloading)
        no_vae_sets = [
            ["musetalk/musetalk/musetalk.json", "musetalk/musetalk/pytorch_model.bin", "whisper/pytorch_model.bin"],
        ]
        for candidate in no_vae_sets:
            if all(os.path.exists(os.path.join(self.checkpoints_dir, p)) for p in candidate):
                logger.warning("⚠️ VAE not yet downloaded — will use simulation until VAE is ready.")
                return False
        return False

    def process_audio(self, audio_bytes: bytes, reference_image: Image.Image) -> list[np.ndarray]:
        """
        Accepts PCM audio bytes and the reference avatar image,
        generates animated talking lip-sync video frames.
        """
        img_np = np.array(reference_image)
        h, w = img_np.shape[:2]

        if self.weights_exist:
            try:
                # Lazy-load models on first call
                _lazy_load_models(self.checkpoints_dir)

                # Detect face landmarks (cache for performance)
                if self._ref_landmarks is None:
                    results = _face_mesh.process(
                        img_np[:, :, :3]  # ensure RGB
                    )
                    if results.multi_face_landmarks:
                        self._ref_landmarks = results.multi_face_landmarks[0].landmark
                    else:
                        logger.warning("No face detected in reference image — falling back to simulation")
                        return self._simulate(img_np, audio_bytes)

                # Get mouth bounding box
                bbox = _get_mouth_bbox(self._ref_landmarks, w, h, pad=24)
                x1, y1, x2, y2 = bbox
                mouth_crop = img_np[y1:y2, x1:x2]

                if mouth_crop.size == 0:
                    return self._simulate(img_np, audio_bytes)

                # Extract Whisper audio features
                audio_features = _extract_audio_features(audio_bytes)

                # Generate frames — one frame per ~100ms audio chunk
                num_frames = max(5, len(audio_bytes) // 3200)
                frames = []
                for i in range(num_frames):
                    # Shift audio window for each frame for temporal variation
                    frame_offset = min(i * 3, audio_features.shape[1] - 10)
                    frame_features = audio_features[:, frame_offset:frame_offset + 77, :]
                    if frame_features.shape[1] < 77:
                        import torch
                        pad = torch.zeros(
                            1, 77 - frame_features.shape[1], frame_features.shape[2],
                            device=frame_features.device
                        )
                        frame_features = torch.cat([frame_features, pad], dim=1)

                    generated_mouth = _run_musetalk_inference(mouth_crop, frame_features)
                    blended_frame = _blend_mouth(img_np, generated_mouth, bbox)
                    frames.append(blended_frame)

                return frames

            except Exception as e:
                logger.error(f"MuseTalk inference error: {e} — falling back to simulation")
                return self._simulate(img_np, audio_bytes)

        return self._simulate(img_np, audio_bytes)

    def _simulate(self, img_np: np.ndarray, audio_bytes: bytes) -> list:
        """Fallback simulation: simple oval mouth animation."""
        h, w = img_np.shape[:2]
        frames = []
        num_frames = max(5, len(audio_bytes) // 3200)

        for i in range(num_frames):
            frame = img_np.copy()
            mouth_y = int(h * 0.7)
            mouth_x = int(w * 0.5)
            radius_y = int(12 * (1 + np.sin(i * 0.8)))
            radius_x = 24

            for dy in range(-radius_y, radius_y):
                for dx in range(-radius_x, radius_x):
                    if (dx * dx) / (radius_x * radius_x) + (dy * dy) / max(1, (radius_y * radius_y)) <= 1.0:
                        cy = mouth_y + dy
                        cx = mouth_x + dx
                        if 0 <= cy < h and 0 <= cx < w:
                            frame[cy, cx] = [20, 10, 10]
            frames.append(frame)

        return frames
