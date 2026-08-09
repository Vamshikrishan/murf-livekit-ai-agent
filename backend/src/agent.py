import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    tokenize,
    room_io,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import memory

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
    def __init__(self, instructions: str = SYSTEM_PROMPT) -> None:
        super().__init__(instructions=instructions)

    @function_tool(
        name="lookup_user",
        description="Look up persistent caller memory by stable user ID. Returns stored name, language preference, facts, and last interaction if found.",
    )
    async def lookup_user(self, context: RunContext, user_id: str):
        """Look up a caller's saved memory.

        Args:
            user_id: The stable caller identifier used to retrieve memory.
        """

        logger.info(f"Looking up memory for user_id={user_id}")
        result = memory.lookup_user(user_id)
        if result is None:
            return {
                "status": "not_found",
                "message": "No stored memory was found for this caller.",
            }

        return {
            "status": "found",
            "user": result,
        }

    @function_tool(
        name="save_user_memory",
        description="Save or update caller memory after explicit consent. Use it only when the caller authorizes storing the information.",
    )
    async def save_user_memory(
        self,
        context: RunContext,
        user_id: str,
        name: str | None = None,
        language_preference: str | None = None,
        facts: dict | None = None,
        last_interaction: str | None = None,
    ):
        """Save or update persistent caller memory.

        Args:
            user_id: The stable caller identifier.
            name: Caller name if available.
            language_preference: Preferred language for future conversations.
            facts: Useful, non-sensitive facts to remember.
            last_interaction: ISO timestamp of the latest interaction.
        """

        logger.info(f"Saving memory for user_id={user_id}")
        saved = memory.save_user_memory(
            user_id=user_id,
            name=name,
            language_preference=language_preference,
            facts=facts,
            last_interaction=last_interaction,
        )
        return {
            "status": "saved",
            "user": saved,
        }


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
        stt=deepgram.STT(model="nova-3", language="multi"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-3.5-flash-lite",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="Anisha",
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

    # Initialize the persistent memory database if needed.
    memory.init_db()

    user_id = ctx.local_participant_identity
    if not user_id:
        user_id = ctx.room.name

    dynamic_prompt = SYSTEM_PROMPT + f"""

# PERSISTENT MEMORY
A stable caller identifier is available as user_id = \"{user_id}\".
If the caller is known, call lookup_user(user_id) before responding.
If lookup_user returns stored memory, greet the caller by name and refer naturally to one relevant saved fact.
If you learn a useful personal detail, ask the caller for explicit permission before saving.
If the caller says yes, call save_user_memory(user_id=..., name=..., language_preference=..., facts=..., last_interaction=...).
If the caller says no or refuses, do not save any new information.
Use clear phrases like "Would you like me to remember that for future conversations?" before saving.
"""

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(instructions=dynamic_prompt),
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