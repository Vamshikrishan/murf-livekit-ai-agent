import json
import logging
import os
import random
import socket
import string
from datetime import datetime, timezone
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

# The optional local turn-detector model can fail to initialize on some Windows
# environments because it loads native inference dependencies. We disable it here
# to keep the voice session stable and avoid the room being torn down on startup.
try:
    import livekit.plugins.noise_cancellation as noise_cancellation
except ImportError:
    noise_cancellation = None

import analytics
import memory

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# ── Open-Meteo weather constants ─────────────────────────────────────────────
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


def _detect_call_channel(ctx: JobContext) -> str:
    """Determine a safe channel label for the active call without storing PII."""
    try:
        participant_kind = getattr(getattr(ctx, "local_participant", None), "kind", None)
        if participant_kind is not None:
            try:
                from livekit import rtc

                if participant_kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                    return "sip"
            except Exception:
                pass

        room_name = getattr(ctx.room, "name", "")
        if room_name and "sip" in room_name.lower():
            return "sip"

        if room_name and "browser" in room_name.lower():
            return "browser"

        return "browser"
    except Exception:
        return "unknown"


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


# ── Escalation helpers ────────────────────────────────────────────────────────

def _generate_reference_id() -> str:
    """Generate a unique escalation reference ID in the format ESC-YYYYMMDD-XXXXXX."""
    date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"ESC-{date_part}-{suffix}"


def _send_discord_webhook(webhook_url: str, payload: dict) -> bool:
    """Send a Discord webhook message. Returns True on success, False on failure.
    Never logs the webhook URL itself."""
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "AarogyaMitra-Escalation/1.0"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except (HTTPError, URLError, socket.timeout, ValueError) as exc:
        logger.error("Discord webhook delivery failed: %s", type(exc).__name__)
        return False


def _build_discord_message(
    reference_id: str,
    urgency: str,
    problem_description: str,
    agent_action: str,
    caller_language: str,
    preferred_followup: str,
    created_at: str,
) -> dict:
    """Build the Discord embed/content payload. No secrets included."""
    urgency_emoji = {
        "EMERGENCY": "🚨",
        "HIGH": "🔴",
        "MEDIUM": "🟡",
        "LOW": "🟢",
    }.get(urgency, "ℹ️")

    content = (
        f"{urgency_emoji} **NEW HUMAN ASSISTANCE REQUEST**\n\n"
        f"**Reference ID:** `{reference_id}`\n"
        f"**Urgency:** {urgency}\n\n"
        f"**Problem:**\n{problem_description}\n\n"
        f"**Agent action:**\n{agent_action}\n\n"
        f"**Language:** {caller_language}\n"
        f"**Preferred follow-up:** {preferred_followup}\n\n"
        f"**Status:** OPEN\n"
        f"**Created:** {created_at}"
    )
    return {"content": content}


# ── Specialist System Prompt ─────────────────────────────────────────────────

SPECIALIST_SYSTEM_PROMPT = """# IDENTITY

You are a Government Scheme Specialist, an AI voice assistant built exclusively to help citizens of India understand, discover, and apply for government welfare schemes, entitlement programmes, and financial-support benefits.

You were connected from AarogyaMitra, the main health assistant. The user's current request is provided to you in the handoff context — read it carefully and continue from where the main agent stopped. Do NOT ask the user to repeat what they already said.

# OBJECTIVES

A successful conversation should:

- Identify which government schemes the user may be eligible for based on their age, income, occupation, gender, family situation, or other relevant details they share.
- Explain scheme benefits, eligibility criteria, required documents, and application steps clearly.
- Help the user understand how to check their application status.
- Refer the user to the appropriate official government portal for final applications.
- Stay accurate, honest, and helpful.

# SCHEMES YOU CAN HELP WITH

You have working knowledge of Indian central and state government schemes, including but not limited to:

- PM-KISAN (Pradhan Mantri Kisan Samman Nidhi) — farmer income support
- PM Jan Dhan Yojana — financial inclusion / bank accounts
- PM Ujjwala Yojana — LPG for BPL families
- Ayushman Bharat PM-JAY — health insurance for low-income families
- PM Awas Yojana — affordable housing (urban and rural)
- PM Mudra Yojana — small business loans
- Sukanya Samriddhi Yojana — savings scheme for girl children
- National Pension Scheme (NPS)
- Atal Pension Yojana
- MGNREGS — rural employment guarantee
- PM SVANidhi — street vendor loans
- PM Garib Kalyan Yojana
- Pradhan Mantri Fasal Bima Yojana — crop insurance
- Stand Up India Scheme — loans for SC/ST/women entrepreneurs
- PM Scholarship Scheme
- National Scholarship Portal schemes
- Senior citizen benefit schemes (Indira Gandhi National Old Age Pension, etc.)
- Self-help group (SHG) and women empowerment schemes
- Disability support schemes (ADIP, DISHA)
- State-level health, housing, education, and livelihood schemes

If the specific scheme the user asks about is outside your current knowledge, say so honestly and refer them to the official source.

# LIMITATIONS

You cannot:

- File or submit applications on behalf of the user.
- Access government databases to check real-time eligibility.
- Guarantee approval of any scheme.
- Provide legal or financial advice.

If asked for something outside your scope, say:
"I can share general information about this scheme, but for accurate and official guidance, please visit the relevant government portal or your nearest Common Service Centre (CSC)."

# HOW TO HANDLE THE HANDOFF

When you receive the handoff, you already have context about the user's original request. Use it.

Do NOT open with "How can I help you?" — you already know what they need.

Open naturally, for example:
"Hello, I'm the government scheme specialist. Based on what you shared, let me help you find the right scheme for your situation. Could you tell me a bit more about [relevant detail]?"

Personalise your opening based on the handoff context provided.

# RETURNING CONTROL TO MAIN AGENT

If the user asks about health topics, medical questions, or anything outside government schemes and benefits:
- Politely acknowledge it.
- Explain that you are a specialist and the main assistant AarogyaMitra can better help with that.
- Call the return_to_main_agent tool to hand back the conversation.

Example:
"That's a health question — my colleague AarogyaMitra would be better placed to help with that. Let me connect you back."

# LANGUAGE & SCRIPT

Always write every language in its own native script.

Reply in the same language used by the user whenever possible.

English → English script.
Hindi → Devanagari script only.
Telugu → Telugu script only.
All other non-English languages must use their native script.

Never romanize Hindi, Telugu, or any other non-English language. This is a hard rule.

Hindi example:
Correct: नमस्ते, मैं आपकी मदद कर सकता हूँ।
Incorrect: Namaste, main aapki madad kar sakta hoon.

Telugu example:
Correct: నమస్కారం, నేను మీకు సహాయం చేయగలను.
Incorrect: Namaskaram, nenu meeku sahayam cheyagalanu.

# CONVERSATION STYLE

- Warm, clear, and patient.
- Use simple language — many users may not be familiar with official scheme jargon.
- Keep responses short and suitable for voice conversation.
- Ask one question at a time to gather eligibility details.
- Never overwhelm the user with a long list of schemes at once — narrow it down based on their situation.

# PRIVACY

Do not ask for Aadhaar numbers, bank account numbers, mobile OTPs, passwords, or any sensitive financial or identification data.

General eligibility information (age, occupation, family size, income range) is fine to ask about.
"""

# ── System Prompt ────────────────────────────────────────────────────────────

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

After recommending emergency care, offer human assistance:
"I can also create a human assistance request so someone can follow up with you. Would you like me to do that? I would share a brief summary of what you told me, the urgency, your preferred language, and your preferred follow-up method."

# PRIVACY

Never ask for or store:

- Aadhaar number
- Health insurance numbers
- Medical record numbers
- Passwords
- OTPs
- Financial information

If users voluntarily share health information, use it only during the current conversation.

# LANGUAGE & SCRIPT

Reply in the same language used by the user whenever possible.

Support:
- English
- Hindi
- Telugu
- Natural Indian code-mixed conversations

Always write every language in its own native script.

Hindi → Devanagari script only.
Correct example: नमस्ते, मैं आपकी मदद कर सकता हूँ।
Incorrect: Namaste, main aapki madad kar sakta hoon.

Telugu → Telugu script only.
Correct example: నమస్కారం, నేను మీకు సహాయం చేయగలను.
Incorrect: Namaskaram, nenu meeku sahayam cheyagalanu.

English → English script.

Never romanize Hindi or Telugu in any response. This is a hard rule.

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

# HUMAN ESCALATION

You know when human help is appropriate.

Escalate to a human support request when:

1. The caller reports a potentially life-threatening emergency (chest pain, breathing difficulty, stroke, severe bleeding, unconsciousness, poisoning, seizures, suicidal thoughts, serious accidents, etc.)
2. The caller asks you to diagnose them ("Do I have diabetes?", "Is this dengue?", "What disease do I have?", "Can you diagnose me?")
3. The caller clearly needs qualified human medical assistance beyond what you can provide

For emergencies: always recommend emergency medical services FIRST, before offering escalation.

For diagnosis requests: explain you cannot diagnose, then offer human assistance.

CONSENT IS MANDATORY before creating any escalation request.

Before calling create_escalation, you MUST explain:
"I can create a request for human assistance. I would share a short summary of what you told me, the urgency, your preferred language, and your preferred follow-up method. Is that okay?"

Only call create_escalation if the user gives clear permission.

Affirmative responses: "yes", "okay", "sure", "go ahead", "please do", "that's fine", "हाँ", "ठीक है", "అవును"
Refusal responses: "no", "don't", "cancel", "I don't want that", "नहीं", "వద్దు"

If the user refuses:
- Do NOT call create_escalation
- Do NOT send any information
- Acknowledge their decision kindly: "That's completely fine. I won't share any information."
- Continue supporting them safely

After a successful escalation:
- Tell the caller their reference ID
- Explain the request is open
- Do NOT promise an immediate human response

Do not diagnose.
Do not pretend to be a doctor.
Do not promise an immediate human response.

# GOVERNMENT SCHEME SPECIALIST HANDOFF

You have access to a dedicated Government Scheme Specialist agent.

Transfer the caller to the specialist using the handoff_to_government_scheme_specialist tool ONLY when the caller's question specifically requires knowledge about government welfare schemes, entitlement programmes, eligibility for benefits, required documents, application process for government schemes, or related financial-assistance schemes.

Examples that SHOULD trigger a handoff:
- "Which government scheme am I eligible for?"
- "How do I apply for PM-KISAN / Ayushman Bharat / PM Awas Yojana / PM Jan Dhan?"
- "I am a farmer, which government benefits can I get?"
- "I am below the poverty line, what government help is available?"
- "What documents do I need for PM Ujjwala Yojana?"
- "सरकारी योजनाओं के बारे में बताइए" (Tell me about government schemes — Hindi)
- "నేను ఏ ప్రభుత్వ పథకానికి అర్హుడను?" (Which government scheme am I eligible for — Telugu)

Examples that should NOT trigger a handoff (handle them yourself):
- General health questions
- Medicine or nutrition questions
- Weather questions
- Mental health questions
- Emergency health situations
- General conversation

Before calling the tool, say a brief acknowledgment so the user knows they are being connected:
For English: "Sure, I'll connect you to our government scheme specialist."
For Hindi: "ज़रूर, मैं आपको हमारे सरकारी योजना विशेषज्ञ से जोड़ता हूँ।"
For Telugu: "తప్పకుండా, నేను మీకు మా ప్రభుత్వ పథకాల నిపుణుడిని కనెక్ట్ చేస్తాను."

Do NOT hand off based merely on financial words appearing in conversation. Only hand off when specialist knowledge about government schemes is genuinely needed.

If the specialist is unavailable, continue helping the user as best as you can with general awareness about the topic.

# FIRST GREETING

Hello! I'm AarogyaMitra, your multilingual AI health assistant. I can help you understand common health topics, healthy habits, nutrition, fitness, and wellness in simple language. How can I help you today?
"""


class GovernmentSchemeSpecialist(Agent):
    """Specialist agent for government schemes and entitlement programmes.

    This agent is instantiated when the main AarogyaMitra agent hands off a caller
    who needs help with government welfare schemes, eligibility checks, or application
    guidance. It receives the original user request as part of its instructions so
    it can continue naturally without making the user repeat themselves.
    """

    def __init__(self, handoff_context: str = "") -> None:
        # Append handoff context so the specialist knows the user's original request
        instructions = SPECIALIST_SYSTEM_PROMPT
        if handoff_context:
            instructions += f"""

# HANDOFF CONTEXT (from AarogyaMitra)

The user was speaking with AarogyaMitra and the following request prompted this handoff.
Use this to personalise your opening and skip re-asking for information the user already provided.

Original user request: {handoff_context}
"""
        super().__init__(instructions=instructions)
        logger.info(
            "GovernmentSchemeSpecialist initialised — handoff_context_length=%d",
            len(handoff_context),
        )

    @function_tool(
        name="return_to_main_agent",
        description=(
            "Return the caller to the main AarogyaMitra health assistant. "
            "Use this when the user asks about health, medical, or any topic "
            "that is outside the scope of government schemes and financial benefits. "
            "Before calling this tool, tell the user you are connecting them back to AarogyaMitra."
        ),
    )
    async def return_to_main_agent(self, context: RunContext):
        """Transfer the caller back to the main AarogyaMitra agent."""
        logger.info(
            "GovernmentSchemeSpecialist: return_to_main_agent called — handing back to Assistant"
        )
        try:
            await context.session.transfer_to(Assistant())
            logger.info("GovernmentSchemeSpecialist: transfer back to Assistant successful")
            return {"status": "transferred_to_main_agent"}
        except Exception as exc:
            logger.error(
                "GovernmentSchemeSpecialist: transfer back to main agent failed — %s: %s",
                type(exc).__name__,
                exc,
            )
            return {
                "status": "transfer_failed",
                "message": "Could not connect back to the main assistant. Please continue with me.",
            }


class Assistant(Agent):
    def __init__(self, instructions: str = SYSTEM_PROMPT) -> None:
        super().__init__(instructions=instructions)
        self.call_id: str | None = None
        self.call_outcome: str | None = None
        self.call_failure_reason: str | None = None
        # Track escalations created this session to prevent duplicates
        self._escalated_ref_ids: set[str] = set()

    def set_call_outcome(self, outcome: str, failure_reason: str | None = None) -> None:
        normalized = (outcome or "failed").strip().lower()
        if normalized not in {"success", "failed"}:
            normalized = "failed"
        if normalized == "success":
            self.call_outcome = "success"
            self.call_failure_reason = None
        else:
            if self.call_outcome is None:
                self.call_outcome = "failed"
            self.call_failure_reason = failure_reason

    async def on_user_turn_completed(
        self, turn_ctx: Any, new_message: Any
    ) -> None:
        content = getattr(new_message, "content", "")
        if self.call_outcome is None and isinstance(content, str) and content.strip():
            if len(content.strip()) > 32:
                self.set_call_outcome("success")

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
        tz = location_data.get("timezone") or "auto"

        if latitude is None or longitude is None:
            return {
                "status": "error",
                "message": (
                    "The location was found, but I could not resolve its coordinates. "
                    "Please try a different city or location."
                ),
            }

        weather = _fetch_weather(latitude, longitude, tz)
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

        logger.info("Looking up memory for user_id=%s", user_id)
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

        logger.info("Saving memory for user_id=%s", user_id)
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

    @function_tool(
        name="create_escalation",
        description=(
            "Create a human assistance request after the caller has given explicit consent. "
            "Call this ONLY when the user has clearly said yes to sharing their information. "
            "Never call this tool without prior user consent. "
            "Use urgency='EMERGENCY' for life-threatening symptoms, 'HIGH' for serious non-emergency issues, "
            "'MEDIUM' for diagnosis requests, 'LOW' for general human support requests. "
            "Do not include passwords, OTPs, Aadhaar numbers, financial information, insurance numbers, "
            "medical record numbers, or full conversation transcripts."
        ),
    )
    async def create_escalation(
        self,
        context: RunContext,
        urgency: str,
        problem_description: str,
        agent_action: str,
        caller_language: str,
        preferred_followup: str,
    ):
        """Create a human escalation request and send it to the configured destination.

        Args:
            urgency: One of EMERGENCY, HIGH, MEDIUM, or LOW.
            problem_description: Short description of what the caller reported. No sensitive data.
            agent_action: What AarogyaMitra already did (e.g. recommended emergency services).
            caller_language: Language the caller used (e.g. English, Hindi, Telugu).
            preferred_followup: How the caller wants to be reached (e.g. Phone, WhatsApp, Email).
        """

        # Normalise and validate urgency
        urgency = (urgency or "MEDIUM").strip().upper()
        if urgency not in ("EMERGENCY", "HIGH", "MEDIUM", "LOW"):
            urgency = "MEDIUM"

        # Generate a unique reference ID
        reference_id = _generate_reference_id()

        # Duplicate-prevention: if an identical problem description was already
        # escalated this session, return the existing reference without re-sending.
        problem_key = problem_description.strip().lower()[:120]
        if problem_key in self._escalated_ref_ids:
            logger.warning(
                "Duplicate escalation detected for problem key, suppressing re-send."
            )
            self.set_call_outcome("success")
            return {
                "status": "duplicate",
                "message": "A human assistance request for this issue was already created this session.",
            }
        self._escalated_ref_ids.add(problem_key)

        created_at = datetime.now(timezone.utc).isoformat()

        # Build the structured escalation record (no secrets)
        escalation_record = {
            "reference_id": reference_id,
            "urgency": urgency,
            "problem_description": problem_description,
            "agent_action": agent_action,
            "caller_language": caller_language,
            "preferred_followup": preferred_followup,
            "created_at": created_at,
            "status": "OPEN",
        }

        logger.info(
            "Creating escalation reference_id=%s urgency=%s language=%s",
            reference_id,
            urgency,
            caller_language,
        )

        # Send to Discord webhook
        webhook_url = os.environ.get("DISCORD_ESCALATION_WEBHOOK_URL", "").strip()
        if not webhook_url:
            logger.warning(
                "DISCORD_ESCALATION_WEBHOOK_URL is not configured — escalation record created locally only."
            )
            return {
                "status": "created_locally",
                "reference_id": reference_id,
                "message": (
                    "Your request has been recorded. Human escalation delivery is currently unavailable, "
                    "but your reference ID is ready."
                ),
                "escalation": escalation_record,
            }

        discord_payload = _build_discord_message(
            reference_id=reference_id,
            urgency=urgency,
            problem_description=problem_description,
            agent_action=agent_action,
            caller_language=caller_language,
            preferred_followup=preferred_followup,
            created_at=created_at,
        )

        delivered = _send_discord_webhook(webhook_url, discord_payload)

        if delivered:
            logger.info("Escalation delivered to Discord successfully reference_id=%s", reference_id)
            self.set_call_outcome("success")
            return {
                "status": "created",
                "reference_id": reference_id,
                "message": (
                    f"Your request has been created successfully. "
                    f"Your reference ID is {reference_id}. "
                    "A human support request is now open. "
                    "Please note that response times may vary."
                ),
                "escalation": escalation_record,
            }
        else:
            logger.error("Escalation Discord delivery failed reference_id=%s", reference_id)
            self.set_call_outcome("failed", failure_reason="escalation_delivery_failed")
            return {
                "status": "delivery_failed",
                "reference_id": reference_id,
                "message": (
                    "Your request has been recorded but could not be delivered right now. "
                    f"Your reference ID is {reference_id}. Please keep this for follow-up."
                ),
                "escalation": escalation_record,
            }


    @function_tool(
        name="handoff_to_government_scheme_specialist",
        description=(
            "Transfer the caller to the Government Scheme Specialist agent. "
            "Use this ONLY when the caller specifically needs help with Indian government "
            "welfare schemes, entitlement programmes, scheme eligibility, required documents, "
            "application process for government schemes, or related financial-assistance benefits. "
            "Do NOT use this for general health questions, weather, or unrelated topics. "
            "Pass the caller's current request as user_request so the specialist has context "
            "and does not need the user to repeat themselves."
        ),
    )
    async def handoff_to_government_scheme_specialist(
        self,
        context: RunContext,
        user_request: str,
    ):
        """Hand the caller off to the GovernmentSchemeSpecialist.

        Args:
            user_request: A concise summary of what the user asked for, used to
                          initialise the specialist with the correct context.
        """
        logger.info(
            "Assistant: handoff_to_government_scheme_specialist called — "
            "user_request_summary=%r",
            user_request[:120] if user_request else "",
        )

        try:
            specialist = GovernmentSchemeSpecialist(handoff_context=user_request)
            logger.info(
                "Assistant: GovernmentSchemeSpecialist created — initiating session transfer"
            )
            await context.session.transfer_to(specialist)
            logger.info("Assistant: session transfer to GovernmentSchemeSpecialist successful")
            return {"status": "transferred_to_specialist"}

        except Exception as exc:
            logger.error(
                "Assistant: handoff to GovernmentSchemeSpecialist failed — %s: %s",
                type(exc).__name__,
                exc,
            )
            # Graceful degradation — stay in main agent and inform the user
            return {
                "status": "handoff_failed",
                "message": (
                    "I'm sorry, the government scheme specialist is currently unavailable. "
                    "I'll do my best to help you with general information about this topic."
                ),
            }


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    analytics.init_db()
    call_id = analytics.start_call_record(channel=_detect_call_channel(ctx))
    agent = None

    def finalize_call(reason: str | None = None) -> None:
        nonlocal agent
        if agent is not None and agent.call_id is None:
            agent.call_id = call_id
        if call_id:
            outcome = "failed"
            if getattr(agent, "call_outcome", None) == "success":
                outcome = "success"
            analytics.close_call_record(
                call_id,
                outcome=outcome,
                failure_reason=reason if outcome == "failed" else None,
            )
            logger.info(
                "Call analytics closed call_id=%s outcome=%s failure_reason=%s",
                call_id,
                outcome,
                reason,
            )

    # Logging setup — no credentials logged here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using Murf Falcon, Gemini, Deepgram, and Silero VAD.
    # turn_detection is intentionally None: the local turn-detector DLL can fail on
    # some Windows environments. Silero VAD alone is sufficient for stable operation.
    session = AgentSession(
        # STT: Deepgram nova-3 with multilingual support (English, Hindi, Telugu, etc.)
        stt=deepgram.STT(model="nova-3", language="multi"),
        # LLM: Gemini for conversational health Q&A
        llm=google.LLM(
            model="gemini-3.5-flash-lite",
        ),
        # TTS: Murf Falcon — the fastest TTS API — voice Anisha (Indian English)
        tts=murf.TTS(
            voice="Anisha",
            locale="en-IN",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        # VAD-only turn detection (local model intentionally disabled for Windows stability)
        turn_detection=None,
        vad=ctx.proc.userdata["vad"],
        # Allow LLM to start generating while waiting for end-of-turn
        preemptive_generation=True,
    )

    # Initialize the persistent memory database if needed.
    memory.init_db()

    user_id = ctx.local_participant_identity
    if not user_id:
        user_id = ctx.room.name

    ctx.room.on("disconnected", lambda reason: finalize_call(f"room_disconnected:{reason}"))
    ctx.room.on(
        "participant_disconnected",
        lambda participant: finalize_call(f"participant_disconnected:{getattr(participant, 'identity', 'unknown')}"),
    )

    dynamic_prompt = SYSTEM_PROMPT + f"""

# PERSISTENT MEMORY

A stable caller identifier is available as user_id = "{user_id}".
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

Always respond in the language used by the user whenever possible.

English → English script.
Hindi → Devanagari script only.
Telugu → Telugu script only.
All other non-English languages should also use their native script.

Never romanize Hindi, Telugu, or any other non-English language. This is a hard rule.

Hindi example:
Correct: नमस्ते, मैं आपकी मदद कर सकता हूँ।
Incorrect: Namaste, main aapki madad kar sakta hoon.

Telugu example:
Correct: నమస్కారం, నేను మీకు సహాయం చేయగలను.
Incorrect: Namaskaram, nenu meeku sahayam cheyagalanu.

Keep responses short, natural, and suitable for spoken conversation.
"""

    agent = Assistant(instructions=dynamic_prompt)
    agent.call_id = call_id

    try:
        # Connect to the room and start the agent session
        await ctx.connect()

        await session.start(
            agent=agent,
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
    except Exception:
        finalize_call("session_exception")
        raise
    finally:
        finalize_call("session_end")


if __name__ == "__main__":
    cli.run_app(server)
