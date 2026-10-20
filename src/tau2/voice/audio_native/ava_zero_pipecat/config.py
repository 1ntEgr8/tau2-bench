"""Connection settings for the Ava Zero Pipecat provider.

The Ava Zero server owns its own model, speech, and turn-taking configuration.
tau2 only needs to know where the server is, how to reach it, which agent to
start, and how long to wait for each step of the session handshake.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
    DEFAULT_AVA_ZERO_BASE_URL,
    DEFAULT_AVA_ZERO_BOOT_ATTEMPTS,
    DEFAULT_AVA_ZERO_BOOT_TIMEOUT_SECONDS,
    DEFAULT_AVA_ZERO_COST_REPORT_TIMEOUT_SECONDS,
    DEFAULT_AVA_ZERO_MODEL,
    DEFAULT_AVA_ZERO_TOOL_REGISTRATION_TIMEOUT_SECONDS,
)


class AvaZeroPipecatConfig(BaseModel):
    """Connection configuration for an Ava Zero server or hosted agent."""

    model_config = ConfigDict(extra="forbid")

    # "websocket" connects to a running server's /ws-client endpoint;
    # "webrtc" calls a hosted agent through Pipecat's agent-server protocol.
    transport: Literal["websocket", "webrtc"] = "websocket"
    # For webrtc, the hosted agent's URL,
    # https://<host>/inference/serverless/pipecat/v1/public/<agent_id>.
    base_url: str = DEFAULT_AVA_ZERO_BASE_URL
    agent: str = Field(
        default=DEFAULT_AVA_ZERO_MODEL,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    websocket_token_env: Optional[str] = None

    connect_timeout_seconds: float = Field(
        default=DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT,
        gt=0,
    )
    boot_timeout_seconds: float = Field(
        default=DEFAULT_AVA_ZERO_BOOT_TIMEOUT_SECONDS,
        gt=0,
    )
    tool_registration_timeout_seconds: float = Field(
        default=DEFAULT_AVA_ZERO_TOOL_REGISTRATION_TIMEOUT_SECONDS,
        gt=0,
    )
    cost_report_timeout_seconds: float = Field(
        default=DEFAULT_AVA_ZERO_COST_REPORT_TIMEOUT_SECONDS,
        gt=0,
    )
    boot_attempts: int = Field(default=DEFAULT_AVA_ZERO_BOOT_ATTEMPTS, ge=1)

    @model_validator(mode="after")
    def validate_transport(self) -> "AvaZeroPipecatConfig":
        """Reject settings that do not apply to the selected transport."""

        if self.transport == "webrtc":
            if not self.base_url.startswith(("https://", "http://")):
                raise ValueError(
                    "the webrtc transport needs the agent's http(s) URL, "
                    "https://<host>/inference/serverless/pipecat/v1/public/<agent_id>"
                )
            if self.websocket_token_env is not None:
                raise ValueError(
                    "websocket_token_env applies only to the websocket transport"
                )
        return self
