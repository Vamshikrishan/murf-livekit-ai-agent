import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    tokenize,
    room_io,
)
from livekit.plugins import (
    murf,
    silero,
    google,
    deepgram,
    noise_cancellation,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# ----------------------------------------------------
# Logging
# ----------------------------------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("agent")

load_dotenv(".env.local")

# ----------------------------------------------------
# Prompt
# ----------------------------------------------------
SYSTEM_PROMPT = """
You are a friendly AI assistant.

Whenever the user greets you, greet them back.

Keep your answers short.

You always reply in English.
"""

# ----------------------------------------------------
# Assistant
# ----------------------------------------------------
class Assistant(Agent):
    def __init__(self):
        super().__init__(instructions=SYSTEM_PROMPT)


# ----------------------------------------------------
# Server
# ----------------------------------------------------
server = AgentServer()


def prewarm(proc: JobProcess):
    print("\nLoading Silero VAD...")
    proc.userdata["vad"] = silero.VAD.load()
    print("VAD Loaded Successfully\n")


server.setup_fnc = prewarm


# ----------------------------------------------------
# Agent
# ----------------------------------------------------
@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):

    print("\n" + "=" * 70)
    print("VOICE AGENT STARTED")
    print("=" * 70)

    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(

        # ---------------- STT ----------------
        stt=deepgram.STT(
            model="nova-3",
        ),

        # ---------------- Gemini ----------------
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),

        # ---------------- Murf ----------------
        tts=murf.TTS(
            voice="Anisha",
            locale="en-IN",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(
                min_sentence_len=2,
            ),
            text_pacing=True,
        ),

        # ---------------- Turn Detection ----------------
        turn_detection=MultilingualModel(),

        # ---------------- Voice Activity Detection ----------------
        vad=ctx.proc.userdata["vad"],

        preemptive_generation=True,
    )

    # ============================================================
    # DEBUG EVENTS
    # ============================================================

    @session.on("speech_started")
    def _():
        print("\nUSER STARTED SPEAKING")

    @session.on("speech_stopped")
    def _():
        print("\nUSER STOPPED SPEAKING")

    @session.on("user_input_transcribed")
    def _(event):
        print("\n==============================")
        print("TRANSCRIPTION")
        print("==============================")
        print(event.transcript)

    @session.on("agent_response")
    def _(event):
        print("\n==============================")
        print("GEMINI RESPONSE")
        print("==============================")
        print(event.text)

    @session.on("error")
    def _(event):
        print("\n==============================")
        print("SESSION ERROR")
        print("==============================")
        print(event)

    # ============================================================

    print("\nStarting Agent Session...")

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                )
            ),
        ),
    )

    print("Agent Session Started Successfully")

    print("\nConnecting to LiveKit Room...")
    await ctx.connect()
    print("Connected Successfully")

    print("\n======================================")
    print("READY")
    print("Now say: Hello")
    print("======================================\n")


if __name__ == "__main__":
    cli.run_app(server)