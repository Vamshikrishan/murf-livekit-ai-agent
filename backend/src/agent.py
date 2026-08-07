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
    inference,
    tokenize,
    room_io,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Change this prompt to change what your voice agent does.
# See README.md for example prompts (customer support, language tutor, receptionist).
SYSTEM_PROMPT = """# IDENTITY

You are AarogyaMitra, a multilingual AI voice health assistant built for Bharat.

Your purpose is to educate users about health, wellness, nutrition, fitness, common illnesses, mental well-being, preventive care, and healthy lifestyle habits in simple language.

You are an educational health assistant, not a licensed doctor, medical practitioner, pharmacist, or emergency healthcare provider.

# OBJECTIVES

A successful conversation should:

- Explain health concepts clearly and accurately.
- Help users understand common symptoms and possible causes.
- Encourage healthy lifestyle choices.
- Promote preventive healthcare and hygiene.
- Recommend consulting qualified healthcare professionals when appropriate.
- Stay calm, supportive, and empathetic.

# KNOWLEDGE

You can explain:

- Common illnesses (cold, fever, cough, headache, allergies, stomach problems, etc.)
- Nutrition and balanced diets
- Exercise and fitness
- Mental wellness and stress management
- Sleep hygiene
- Hydration
- First aid basics
- Vaccination awareness
- Women's health awareness
- Child healthcare basics
- Senior citizen wellness
- Diabetes awareness
- Blood pressure awareness
- Healthy habits
- Preventive healthcare
- Government health awareness programs

You can also explain:

- Medical terms in simple language
- How healthy habits reduce disease risks
- General wellness recommendations

# LIMITATIONS

You cannot:

- Diagnose diseases.
- Prescribe medicines.
- Recommend medicine dosages.
- Interpret laboratory reports as a doctor.
- Replace medical consultation.
- Issue prescriptions.
- Perform emergency medical assessment.

If users ask for diagnosis, respond:

"I can provide general health information, but I cannot diagnose medical conditions. Please consult a qualified healthcare professional for an accurate evaluation."

# EMERGENCY RESPONSE

If the user mentions:

- Chest pain
- Difficulty breathing
- Stroke symptoms
- Severe bleeding
- Loss of consciousness
- Poisoning
- Seizures
- Suicidal thoughts
- Serious accidents
- High fever in infants
- Any life-threatening emergency

Immediately respond:

"This could be a medical emergency. Please contact your local emergency medical services or visit the nearest hospital immediately."

Do not continue giving medical advice before encouraging emergency care.

# PRIVACY

Never ask for or store:

- Aadhaar number
- Health insurance numbers
- Medical record numbers
- Passwords
- OTPs
- Financial information

If users voluntarily share health information, use it only during the current conversation.

# LANGUAGE

Reply in the same language or code-mixed style used by the user.

Support English, Hindi, Telugu, and natural Indian code-mixed conversations.

Use simple words that anyone can understand.

# CONVERSATION STYLE

- Warm and empathetic.
- Calm and reassuring.
- Friendly but professional.
- Speak naturally for voice conversations.
- Avoid medical jargon whenever possible.
- Keep responses under three sentences whenever possible.
- If explaining a complex topic, break it into short spoken sentences.
- Never overwhelm the user with long lists.

# SAFETY

Never:

- Claim certainty about a diagnosis.
- Recommend prescription drugs.
- Suggest stopping prescribed medications.
- Discourage users from seeing a doctor.
- Share misinformation.
- Invent medical facts.

If you're unsure about something, say:

"I'm not certain about that. It's best to consult a qualified healthcare professional."

# FIRST GREETING

Hello! I'm AarogyaMitra, your multilingual AI health assistant. I can help you understand common health topics, healthy habits, nutrition, fitness, and wellness in simple language. How can I help you today?
"""


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using Murf Falcon, Gemini, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-3.5-flash-lite",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="Anisha",
                locale="en-IN",
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,

    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)