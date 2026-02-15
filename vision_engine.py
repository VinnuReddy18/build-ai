"""
vision_engine.py — OpenCV processing & Claude API logic for Home Guard AI.

Key optimizations:
  - Frames resized to 640x480 and JPEG-encoded at 70% quality before Base64 encoding.
  - Smart sampling: only captures frames where >5% pixels change (motion detection).
  - Frame throttling: max 1 Claude API call every 5 seconds.
"""

import os
import time
import base64
import json
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Frame pre-processing
# ---------------------------------------------------------------------------

TARGET_WIDTH = 640
TARGET_HEIGHT = 480
JPEG_QUALITY = 70
MOTION_THRESHOLD = 0.05          # 5% pixel change
THROTTLE_SECONDS = 5             # min gap between Claude calls

# Track last analysis time for throttling
_last_analysis_time = 0.0


def preprocess_frame(frame):
    """Resize to 640x480 and return the processed frame."""
    if frame is None:
        return None
    return cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))


def frame_to_base64(frame) -> str:
    """Encode a frame as JPEG (70% quality) → Base64 string."""
    if frame is None:
        return ""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    _, buffer = cv2.imencode(".jpg", frame, encode_params)
    return base64.b64encode(buffer).decode("utf-8")


# ---------------------------------------------------------------------------
# Motion detection (smart sampling)
# ---------------------------------------------------------------------------

def detect_motion(prev_frame, curr_frame) -> bool:
    """
    Compare two grayscale frames. Return True if more than MOTION_THRESHOLD
    fraction of pixels changed.
    """
    if prev_frame is None or curr_frame is None:
        return True  # first frame → always process

    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Absolute difference + threshold
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    changed_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    change_ratio = changed_pixels / total_pixels

    return change_ratio > MOTION_THRESHOLD


def get_motion_score(prev_frame, curr_frame) -> float:
    """Return the fraction of pixels that changed between frames."""
    if prev_frame is None or curr_frame is None:
        return 1.0

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    changed_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    return changed_pixels / total_pixels


# ---------------------------------------------------------------------------
# Frame overlay drawing
# ---------------------------------------------------------------------------

def draw_status_overlay(frame, status: str = "MONITORING", color=(0, 255, 0)):
    """Draw a status banner on the frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Semi-transparent banner at top
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Status text
    cv2.putText(
        frame, f"HOME GUARD AI | {status}",
        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )

    # Timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame, timestamp,
        (w - 230, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
    )

    return frame


def draw_motion_border(frame, motion_detected: bool):
    """Draw a green/red border to indicate motion status."""
    h, w = frame.shape[:2]
    color = (0, 0, 255) if motion_detected else (0, 255, 0)
    thickness = 3
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)
    return frame


# ---------------------------------------------------------------------------
# Claude 3 Haiku analysis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a high-security AI for an Indian household surveillance system called "Home Guard AI".

Analyze the provided image and respond ONLY with valid JSON (no markdown, no code fences) in this exact format:
{
    "threat_level": "low" | "medium" | "high",
    "description": "1-sentence description in English",
    "description_telugu": "Same description translated to Telugu",
    "category": "delivery" | "visitor" | "family" | "stranger" | "animal" | "vehicle" | "empty" | "other",
    "details": "Brief explanation of what you see and why you assigned this threat level"
}

Rules:
- If you see a delivery agent (Zomato/Swiggy/Amazon/Flipkart uniform or package), mark as LOW threat.
- If you see a known uniform (postman, police), mark as LOW threat.
- If you see a stranger loitering near the entrance or looking suspicious, mark as HIGH threat.
- If you see someone trying to peek, climb, or tamper with the door/gate, mark as HIGH threat.
- If the scene is calm with no unusual activity, mark as LOW threat.
- If unsure about a person's intent, mark as MEDIUM threat.
- Animals or vehicles passing by without stopping: LOW threat.
- Provide the Telugu translation naturally — this is for a Telugu-speaking household.
"""


def is_claude_configured() -> bool:
    """Return True if the Anthropic API key is set."""
    return bool(ANTHROPIC_API_KEY)


def can_analyze() -> bool:
    """Return True if enough time has passed since the last analysis (throttle)."""
    global _last_analysis_time
    return (time.time() - _last_analysis_time) >= THROTTLE_SECONDS


def analyze_frame(frame) -> dict:
    """
    Send a frame to Claude 3 Haiku for threat analysis.
    Returns a dict with threat_level, description, etc.
    Returns None if throttled or unconfigured.
    """
    global _last_analysis_time

    if not is_claude_configured():
        return {"error": "Anthropic API key not configured."}

    if not can_analyze():
        return None  # throttled

    try:
        import anthropic

        # Pre-process
        processed = preprocess_frame(frame)
        img_b64 = frame_to_base64(processed)

        client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            base_url="https://api.anthropic.com",
        )

        message = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Analyze this surveillance frame. What do you see? Assess the threat level.",
                        },
                    ],
                }
            ],
        )

        _last_analysis_time = time.time()

        # Parse response
        response_text = message.content[0].text.strip()

        # Try to extract JSON from the response
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(response_text[start:end])
            else:
                result = {
                    "threat_level": "medium",
                    "description": response_text[:200],
                    "description_telugu": "",
                    "category": "other",
                    "details": response_text,
                }

        return result

    except Exception as e:
        print(f"[VISION] Claude analysis error: {e}")
        return {"error": str(e)}


def analyze_frame_mock(frame) -> dict:
    """
    Mock analysis for testing without an API key.
    Uses basic motion/color heuristics.
    """
    if frame is None:
        return {
            "threat_level": "low",
            "description": "No frame available.",
            "description_telugu": "ఫ్రేమ్ అందుబాటులో లేదు.",
            "category": "empty",
            "details": "Camera returned no frame.",
        }

    # Simple heuristic: check for large dark blobs (potential person)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    dark_ratio = np.count_nonzero(thresh) / (thresh.shape[0] * thresh.shape[1])

    if dark_ratio > 0.3:
        return {
            "threat_level": "medium",
            "description": "Large object or person detected in frame.",
            "description_telugu": "ఫ్రేమ్‌లో పెద్ద వస్తువు లేదా వ్యక్తి గుర్తించబడింది.",
            "category": "stranger",
            "details": f"Dark region covers {dark_ratio:.1%} of frame.",
        }

    return {
        "threat_level": "low",
        "description": "Scene appears calm, no unusual activity.",
        "description_telugu": "దృశ్యం ప్రశాంతంగా కనిపిస్తోంది, అసాధారణ కార్యకలాపం లేదు.",
        "category": "empty",
        "details": f"Normal scene, dark region: {dark_ratio:.1%}",
    }
