"""
alerts.py — Twilio WhatsApp + Voice Call integration for Aegis Surveillance.
"""

import os
import time
from dotenv import load_dotenv

load_dotenv(override=True)

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
ADMIN_TO = os.getenv("ADMIN_WHATSAPP_TO", "")

# Voice call settings
TWILIO_PHONE_FROM = os.getenv("TWILIO_PHONE_FROM", "")  # Your Twilio phone number
EMERGENCY_CALL_TO = os.getenv("EMERGENCY_CALL_TO", "")  # Phone to call on HIGH threat

# Cooldown to avoid spamming calls
_last_call_time = 0.0
CALL_COOLDOWN = 60  # Don't call more than once per 60 seconds


def is_configured() -> bool:
    """Return True if Twilio credentials are set."""
    return bool(TWILIO_SID and TWILIO_TOKEN and ADMIN_TO)


def is_voice_configured() -> bool:
    """Return True if voice call settings are configured."""
    return bool(TWILIO_SID and TWILIO_TOKEN and TWILIO_PHONE_FROM and EMERGENCY_CALL_TO)


def send_whatsapp_alert(message: str) -> bool:
    """
    Send a WhatsApp alert to the admin.
    Returns True on success, False on failure.
    """
    if not is_configured():
        print("[ALERTS] Twilio not configured — skipping WhatsApp alert.")
        return False

    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_TOKEN)

        msg = client.messages.create(
            body=f"🚨 AEGIS ALERT 🚨\n\n{message}",
            from_=TWILIO_FROM,
            to=ADMIN_TO,
        )
        print(f"[ALERTS] WhatsApp sent — SID: {msg.sid}")
        return True

    except Exception as e:
        print(f"[ALERTS] Failed to send WhatsApp: {e}")
        return False


def make_emergency_voice_call(description: str) -> bool:
    """
    Make an automated voice call to the emergency contact.
    Uses Twilio TwiML <Say> to speak the alert message.
    Has a 60-second cooldown to prevent call spam.
    Returns True on success, False on failure.
    """
    global _last_call_time

    if not is_voice_configured():
        print("[ALERTS] Voice call not configured — skipping. Set TWILIO_PHONE_FROM and EMERGENCY_CALL_TO in .env")
        return False

    # Cooldown check
    if (time.time() - _last_call_time) < CALL_COOLDOWN:
        print(f"[ALERTS] Voice call cooldown active — skipping (wait {CALL_COOLDOWN}s between calls)")
        return False

    try:
        from twilio.rest import Client
        client = Client(TWILIO_SID, TWILIO_TOKEN)

        # TwiML spoken message
        twiml = f"""
        <Response>
            <Say voice="alice" language="en-IN">
                Aegis Security Alert! High threat detected at your home.
                {description}
                Please check your surveillance feed immediately.
                Repeating: {description}
            </Say>
        </Response>
        """

        call = client.calls.create(
            twiml=twiml,
            from_=TWILIO_PHONE_FROM,
            to=EMERGENCY_CALL_TO,
        )

        _last_call_time = time.time()
        print(f"[ALERTS] 📞 Emergency voice call placed — SID: {call.sid}")
        return True

    except Exception as e:
        print(f"[ALERTS] Failed to make voice call: {e}")
        return False


def send_high_threat_alert(description: str, description_telugu: str = "", action_needed: str = ""):
    """
    Send HIGH threat alerts via all channels:
    1. WhatsApp message
    2. Emergency voice call (if configured)
    """
    # WhatsApp alert
    lines = [
        f"⚠️ HIGH THREAT DETECTED",
        f"",
        f"📍 {description}",
    ]
    if description_telugu:
        lines.append(f"📍 {description_telugu}")
    if action_needed:
        lines.append(f"")
        lines.append(f"⚡ Action: {action_needed}")

    lines.extend([
        f"",
        f"🕐 Please check your surveillance feed immediately.",
    ])

    send_whatsapp_alert("\n".join(lines))

    # Voice call (with cooldown)
    make_emergency_voice_call(description)

