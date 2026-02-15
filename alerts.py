"""
alerts.py — Twilio WhatsApp integration for Home Guard AI.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
ADMIN_TO = os.getenv("ADMIN_WHATSAPP_TO", "")


def is_configured() -> bool:
    """Return True if Twilio credentials are set."""
    return bool(TWILIO_SID and TWILIO_TOKEN and ADMIN_TO)


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
            body=f"🚨 HOME GUARD AI ALERT 🚨\n\n{message}",
            from_=TWILIO_FROM,
            to=ADMIN_TO,
        )
        print(f"[ALERTS] WhatsApp sent — SID: {msg.sid}")
        return True

    except Exception as e:
        print(f"[ALERTS] Failed to send WhatsApp: {e}")
        return False


def send_high_threat_alert(description: str, description_telugu: str = ""):
    """
    Convenience wrapper for high-threat events.
    Sends both English and Telugu descriptions.
    """
    lines = [
        f"⚠️ HIGH THREAT DETECTED",
        f"",
        f"📍 {description}",
    ]
    if description_telugu:
        lines.append(f"📍 {description_telugu}")

    lines.extend([
        f"",
        f"🕐 Please check your surveillance feed immediately.",
    ])

    return send_whatsapp_alert("\n".join(lines))
