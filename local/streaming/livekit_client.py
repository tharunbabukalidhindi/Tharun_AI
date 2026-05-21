"""
LiveKit Receiver (Local Side)
──────────────────────────────
Generates JWT access tokens for browser to join the avatar stream room.
The browser uses the LiveKit JS SDK directly to display the video stream.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

AVATAR_ROOM_NAME = "avatar-stream"


class LiveKitTokens:
    """Generates LiveKit JWT tokens for browser and GPU clients."""

    def __init__(self, url: str, api_key: str, api_secret: str):
        self.url = url
        self.api_key = api_key
        self.api_secret = api_secret

    def generate_viewer_token(self, identity: str = "dashboard-viewer") -> str:
        """
        Generate a token for the browser dashboard to subscribe (view) the stream.
        """
        try:
            from livekit.api import AccessToken, VideoGrants
            token = (
                AccessToken(self.api_key, self.api_secret)
                .with_identity(identity)
                .with_name("Dashboard Viewer")
                .with_grants(
                    VideoGrants(
                        room_join=True,
                        room=AVATAR_ROOM_NAME,
                        can_subscribe=True,
                        can_publish=False,
                    )
                )
            )
            return token.to_jwt()
        except Exception as e:
            logger.error(f"Error generating viewer token: {e}")
            raise

    def generate_publisher_token(self, identity: str = "gpu-publisher") -> str:
        """
        Generate a token for the GPU server to publish the avatar video.
        """
        try:
            from livekit.api import AccessToken, VideoGrants
            token = (
                AccessToken(self.api_key, self.api_secret)
                .with_identity(identity)
                .with_name("GPU Avatar Publisher")
                .with_grants(
                    VideoGrants(
                        room_join=True,
                        room=AVATAR_ROOM_NAME,
                        can_publish=True,
                        can_subscribe=False,
                    )
                )
            )
            return token.to_jwt()
        except Exception as e:
            logger.error(f"Error generating publisher token: {e}")
            raise
