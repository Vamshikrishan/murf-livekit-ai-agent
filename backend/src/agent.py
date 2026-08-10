import json
import logging
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

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
from livekit.plugins import murf, silero, google, deepgram

try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
except ImportError:
    MultilingualModel = None

try:
    import livekit.plugins.noise_cancellation as noise_cancellation
except ImportError:
    noise_cancellation = None

import memory

logger = logging.getLogger("agent")

load_dotenv(".env.local")

OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_USER_AGENT = "AarogyaMitraWeatherTool/1.0"

WEATHER_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog or mist",
    48: "Fog with depositing rime",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Freezing drizzle",
    57: "Freezing drizzle",
    61: "Light rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Freezing rain",
    71: "Light snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}


def _http_get_json(url: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    request = Request(url, headers={"User-Agent": OPEN_METEO_USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload)
    except (HTTPError, URLError, socket.timeout, ValueError):
        return None


def _geocode_location(location: str) -> Optional[Dict[str, Any]]:
    if not location or not location.strip():
        return None

    query = quote_plus(location.strip())
    url = f"{OPEN_METEO_GEOCODING_URL}?name={query}&count=1&language=en&format=json"
    data = _http_get_json(url)
    if not data or not isinstance(data, dict):
        return None

    results = data.get("results")
    if not results or not isinstance(results, list):
        return None

    return results[0]


def _weather_description(code: int) -> str:
    return WEATHER_CODE_MAP.get(code, "Weather condition")


def _fetch_weather(latitude: float, longitude: float, timezone: str) -> Optional[Dict[str, Any]]:
    url = (
        f"{OPEN_METEO_WEATHER_URL}?latitude={latitude}&longitude={longitude}"
        f"&current_weather=true&hourly=apparent_temperature,precipitation&timezone={quote_plus(timezone)}"
    )
    data = _http_get_json(url)
    if not data or not isinstance(data, dict):
        return None

    current = data.get("current_weather")
    hourly = data.get("hourly") or {}
    if not current or not isinstance(current, dict):
        return None

    observation_time = current.get("time")
    temperature = current.get("temperature")
    weather_code = current.get("weathercode")
    wind_speed = current.get("windspeed")

    apparent_temp = None
    precipitation = None
    if hourly:
        time_index = None
        times = hourly.get("time") or []
        if observation_time in times:
            time_index = times.index(observation_time)
        if time_index is not None:
            apparent_values = hourly.get("apparent_temperature") or []
            precipitation_values = hourly.get("precipitation") or []
            if len(apparent_values) > time_index:
                apparent_temp = apparent_values[time_index]
            if len(precipitation_values) > time_index:
                precipitation = precipitation_values[time_index]

    return {
        "observation_time": observation_time,
        "temperature_c": temperature,
        "apparent_temperature_c": apparent_temp,
        "condition": _weather_description(weather_code) if isinstance(weather_code, int) else "Unknown",
        "precipitation_mm": precipitation,
        "wind_speed_kmh": wind_speed,
    }

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
        name="weather_lookup",
        description=(
            "Get current weather information for a city or location. "
            "Use this tool when the user asks about current weather, temperature, rain, wind, or weather conditions. "
            "The tool requires a location string and returns factual, external weather data only. "
            "Do not use this tool for unrelated questions or to invent weather information."
        ),
    )
    async def weather_lookup(self, context: RunContext, location: str):
        """Fetch current weather for the requested location using Open-Meteo."""
        logger.info("Weather lookup requested for location=%s", location)

        if not location or not location.strip():
            return {
                "status": "error",
                "message": (
                    "Location is required to fetch current weather information. "
                    "Please provide a city or place name."
                ),
            }

        location_data = _geocode_location(location)
        if not location_data:
            return {
                "status": "error",
                "message": (
                    "I could not find the requested location. "
                    "Please try a different city or spelling."
                ),
            }

        latitude = location_data.get("latitude")
        longitude = location_data.get("longitude")
        name = location_data.get("name")
        country = location_data.get("country")
        timezone = location_data.get("timezone") or "auto"

        if latitude is None or longitude is None:
            return {
                "status": "error",
                "message": (
                    "The location was found, but I could not resolve its coordinates. "
                    "Please try a different city or location."
                ),
            }

        weather = _fetch_weather(latitude, longitude, timezone)
        if not weather:
            return {
                "status": "error",
                "message": (
                    "Sorry, I could not retrieve current weather information from the external weather service right now. "
                    "Please try again later."
                ),
            }

        return {
            "status": "ok",
            "source": "open-meteo",
            "location": {
                "name": name,
                "country": country,
                "latitude": latitude,
                "longitude": longitude,
            },
            "weather": weather,
        }

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
        turn_detection=MultilingualModel() if MultilingualModel is not None else None,
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

# TOOL USAGE
Use the weather_lookup tool only when the user asks for current weather information, temperature, rain, wind, or weather conditions for a specific city or location.
The weather_lookup tool uses an external weather service and returns factual information only. If the location cannot be resolved, or the service is unavailable, do not invent weather details.
If the user asks about weather without a location and the location is known from conversation context or memory, use that remembered location.

# LANGUAGE & SCRIPT
Always write every language in its own native script. Hindi responses must use Devanagari script, never romanized Hindi.
"""

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(instructions=dynamic_prompt),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=(
                    None
                    if noise_cancellation is None
                    else lambda params: (
                        noise_cancellation.BVCTelephony()
                        if params.participant.kind
                        == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                        else noise_cancellation.BVC()
                    )
                )
            )
        ),
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)