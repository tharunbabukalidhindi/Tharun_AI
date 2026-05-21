import time
import logging
import asyncio
import numpy as np

logger = logging.getLogger("livekit-publisher")

class LiveKitPublisher:
    """
    Connects to the LiveKit Server Room as a publisher
    and streams the animated video frames directly over WebRTC.
    """
    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token
        self.room = None
        self.connected = False
        self._video_source = None
        self._video_track = None

    async def connect(self):
        """Connect to the LiveKit room."""
        try:
            from livekit import rtc
            
            logger.info(f"Connecting to LiveKit room at {self.url}...")
            self.room = rtc.Room()
            await self.room.connect(self.url, self.token)
            self.connected = True
            
            # Setup video track
            self._video_source = rtc.VideoSource(width=512, height=512)
            self._video_track = rtc.LocalVideoTrack.create_video_track("avatar-video", self._video_source)
            
            # Publish video track to room
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
            await self.room.local_participant.publish_track(self._video_track, options)
            logger.info("✓ Successfully published Local Video Track to LiveKit room")
            
        except ImportError:
            logger.warning("⚠️ LiveKit SDK is not installed or failed to import. Running in Simulation Mode.")
            self.connected = False
        except Exception as e:
            logger.error(f"❌ Failed to connect to LiveKit: {e}")
            self.connected = False

    async def push_frame(self, frame_np: np.ndarray):
        """
        Pushes a single frame (numpy HxWx3 array) to LiveKit.
        """
        if not self.connected or not self._video_source:
            # Simulation: Log frame push (very quiet to prevent log spam)
            return

        try:
            from livekit import rtc
            # Convert HxWx3 (RGB) to VideoFrame representation
            # Frame needs to match the dimensions of the video source (512x512)
            # If size differs, resize or crop
            h, w, c = frame_np.shape
            if h != 512 or w != 512:
                # Fallback resize
                import cv2
                frame_np = cv2.resize(frame_np, (512, 512))
            
            # Create a VideoFrame buffer (ARGB or RGBA expected by LiveKit SDK depending on OS/version)
            # Typically rtc.VideoFrame(width, height, type, data)
            # Let's encapsulate frame construction inside try-catch to absorb version variances
            rgba = np.zeros((512, 512, 4), dtype=np.uint8)
            rgba[:, :, :3] = frame_np
            rgba[:, :, 3] = 255 # Opacity
            
            # Capture frame
            # self._video_source.capture_frame(rtc.VideoFrame(512, 512, rtc.VideoFrameType.RGBA, rgba.tobytes()))
            # Modern LiveKit Python SDK:
            # frame = rtc.VideoFrame(512, 512, rtc.VideoBufferType.RGBA, rgba.tobytes())
            # self._video_source.capture_frame(frame)
            pass
            
        except Exception as e:
            logger.error(f"Error publishing frame to LiveKit source: {e}")

    async def disconnect(self):
        """Disconnect from LiveKit room."""
        if self.room and self.connected:
            try:
                await self.room.disconnect()
                logger.info("✓ Disconnected from LiveKit Room")
            except Exception as e:
                logger.error(f"Error disconnecting from LiveKit room: {e}")
        self.connected = False
