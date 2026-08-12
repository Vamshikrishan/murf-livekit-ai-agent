"""
outbound_call.py — Trigger an outbound SIP call for Vaani Voice Agent.

Usage:
    From the backend/ directory:

        uv run python src/outbound_call.py

Prerequisites:
    1. Your LiveKit agent must be running:
           uv run python src/agent.py dev

    2. LIVEKIT_SIP_TRUNK_ID must be configured in .env.local

    3. LINPHONE_SIP_URI must be configured in .env.local
       Example:
           LINPHONE_SIP_URI=sip:yourusername@sip.linphone.org

    4. Linphone must be open and registered on your phone if
       you are using Linphone for the outbound call.

The script:
    1. Creates a unique LiveKit room.
    2. Creates a SIP participant in that room.
    3. Dispatches "my-agent" into the same room.
    4. The phone rings.
    5. Once answered, Vaani Voice Agent handles the conversation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

from dotenv import load_dotenv
from livekit import api


# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------

load_dotenv(dotenv_path=".env.local")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

logger = logging.getLogger("vaani.outbound_call")


# ---------------------------------------------------------------------------
# Environment variable helper
# ---------------------------------------------------------------------------

def _require(key: str) -> str:
    """
    Get a required environment variable.

    Raises:
        EnvironmentError: If the variable is missing or empty.
    """

    value = os.environ.get(key, "").strip()

    if not value:
        raise EnvironmentError(
            f"\nMissing required environment variable: {key}\n"
            f"Add {key} to backend/.env.local and try again.\n"
        )

    return value


# ---------------------------------------------------------------------------
# Outbound call
# ---------------------------------------------------------------------------

async def make_outbound_call(
    sip_call_to: str | None = None,
    room_name: str | None = None,
    participant_name: str = "Vaani Voice Agent",
) -> None:
    """
    Start an outbound SIP call through LiveKit.

    Args:
        sip_call_to:
            SIP URI or phone number to call.

            If not supplied, the script uses:
                LINPHONE_SIP_URI

        room_name:
            Optional LiveKit room name.

            If not supplied, a unique room is generated.

        participant_name:
            Display name for the AI participant.
    """

    # -----------------------------------------------------------------------
    # LiveKit credentials
    # -----------------------------------------------------------------------

    livekit_url = _require("LIVEKIT_URL")
    api_key = _require("LIVEKIT_API_KEY")
    api_secret = _require("LIVEKIT_API_SECRET")

    # -----------------------------------------------------------------------
    # SIP trunk
    # -----------------------------------------------------------------------

    trunk_id = _require("LIVEKIT_SIP_TRUNK_ID")

    # -----------------------------------------------------------------------
    # Destination
    # -----------------------------------------------------------------------

    raw_uri = sip_call_to or _require("LINPHONE_SIP_URI")

    # -----------------------------------------------------------------------
    # LiveKit SIP expects the destination in the appropriate SIP format.
    #
    # For Linphone:
    #
    #     sip:username@sip.linphone.org
    #
    # We extract the user part because this matches the working
    # implementation used in the reference project.
    # -----------------------------------------------------------------------

    sip_user = raw_uri.removeprefix("sip:").split("@")[0]

    # -----------------------------------------------------------------------
    # Create a unique room.
    #
    # IMPORTANT:
    # This exact room name will be used for BOTH:
    #
    #   1. SIP participant
    #   2. AI agent dispatch
    #
    # This allows the phone participant and Vaani Voice Agent
    # to communicate inside the same LiveKit room.
    # -----------------------------------------------------------------------

    room = room_name or f"vaani-outbound-{uuid.uuid4().hex[:8]}"

    logger.info("==============================================")
    logger.info("Vaani Voice Agent - Outbound Call")
    logger.info("==============================================")

    logger.info("LiveKit URL:      %s", livekit_url)
    logger.info("SIP trunk ID:     %s", trunk_id)
    logger.info("SIP destination:  %s", sip_user)
    logger.info("Room:             %s", room)
    logger.info("Agent:            my-agent")

    # -----------------------------------------------------------------------
    # Create LiveKit API client
    # -----------------------------------------------------------------------

    lk = api.LiveKitAPI(
        url=livekit_url,
        api_key=api_key,
        api_secret=api_secret,
    )

    try:
        # ================================================================
        # STEP 1 — Create SIP participant
        # ================================================================

        logger.info("Creating SIP participant...")

        sip_participant = await lk.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=trunk_id,

                # Destination being dialed.
                sip_call_to=sip_user,

                # LiveKit room where the phone participant will join.
                room_name=room,

                # Identity of the phone participant.
                participant_identity="phone-user",

                # Display name.
                participant_name=participant_name,

                # Enable LiveKit/Krisp processing where supported.
                krisp_enabled=True,
            )
        )

        logger.info(
            "SIP participant created successfully: %s",
            sip_participant.participant_id,
        )

        # ================================================================
        # STEP 2 — Dispatch Vaani Voice Agent
        # ================================================================

        logger.info("Dispatching Vaani Voice Agent...")

        agent_name = os.environ.get(
            "AGENT_NAME",
            "my-agent",
        ).strip()

        dispatch = await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=agent_name,
                room=room,
            )
        )

        logger.info(
            "Agent dispatched successfully: %s",
            dispatch.agent_name,
        )

        # ================================================================
        # STEP 3 — Success
        # ================================================================

        logger.info("==============================================")
        logger.info("Outbound call initiated successfully!")
        logger.info("==============================================")

        logger.info("Room: %s", room)
        logger.info("Agent: %s", agent_name)

        logger.info(
            "The destination should now receive the call."
        )

        logger.info(
            "Answer the call to speak with Vaani Voice Agent."
        )

    except Exception as exc:
        # ---------------------------------------------------------------
        # Never expose API keys or secrets in the error message.
        # ---------------------------------------------------------------

        logger.exception(
            "Failed to initiate outbound call: %s",
            exc,
        )

        raise

    finally:
        # ---------------------------------------------------------------
        # Always close the LiveKit API connection.
        # ---------------------------------------------------------------

        await lk.aclose()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(
            make_outbound_call()
        )

    except KeyboardInterrupt:
        logger.info("Outbound call cancelled.")

    except Exception as exc:
        logger.error(
            "Outbound call failed: %s",
            exc,
        )
        raise