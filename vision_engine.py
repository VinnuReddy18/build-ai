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
import threading
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

# HOG frame counter — only run HOG every Nth frame to save CPU
_hog_frame_counter = 0
_hog_run_every = 10  # run HOG every 10th frame
_last_hog_result = (False, [])  # cache last HOG result


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
# Smart Pre-Filtering (Layer 2 — FREE, no API calls)
# ---------------------------------------------------------------------------

# HOG (Histogram of Oriented Gradients) person detector
_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Minimum contour area to be considered "significant" (filters shadows/leaves)
MIN_CONTOUR_AREA = 3000  # pixels — roughly a person-sized blob at 640x480


def detect_person_hog(frame) -> tuple:
    """
    Use OpenCV HOG descriptor to detect human-shaped objects.
    Only runs every 10th frame to save CPU. Returns cached result otherwise.
    Returns (bool, list_of_bounding_boxes).
    FREE — no API call needed.
    """
    global _hog_frame_counter, _last_hog_result

    if frame is None:
        return False, []

    _hog_frame_counter += 1

    # Only run expensive HOG every Nth frame, return cached result otherwise
    if _hog_frame_counter % _hog_run_every != 0:
        return _last_hog_result

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes, weights = _hog.detectMultiScale(
        gray, winStride=(8, 8), padding=(4, 4), scale=1.05
    )

    # Filter by confidence
    confident_boxes = []
    for i, (x, y, w, h) in enumerate(boxes):
        if len(weights) > i and weights[i] > 0.3:
            confident_boxes.append((x, y, w, h))

    _last_hog_result = (len(confident_boxes) > 0, confident_boxes)
    return _last_hog_result


def detect_significant_contours(prev_frame, curr_frame) -> tuple:
    """
    Find large contours in the motion diff to filter out small/noise movements.
    Returns (bool, list_of_bounding_boxes).
    FREE — no API call needed.
    """
    if prev_frame is None or curr_frame is None:
        return True, []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    prev_blur = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    curr_blur = cv2.GaussianBlur(curr_gray, (21, 21), 0)

    diff = cv2.absdiff(prev_blur, curr_blur)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Dilate to fill gaps
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    significant_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            (x, y, w, h) = cv2.boundingRect(contour)
            significant_boxes.append((x, y, w, h))

    return len(significant_boxes) > 0, significant_boxes


def should_call_claude(prev_frame, curr_frame) -> tuple:
    """
    Smart 3-layer gating to decide if Claude should be called:
      Layer 1: Motion detection (pixel diff > 5%)
      Layer 2a: Significant contour detection (large moving blobs)
      Layer 2b: HOG person detection (human-shaped object)
      Layer 3: Throttle (5-second cooldown)

    Returns (should_analyze: bool, reason: str, bounding_boxes: list)
    """
    # Layer 1: Basic motion
    if not detect_motion(prev_frame, curr_frame):
        return False, "No motion", []

    # Layer 2a: Check for large contours (person-sized blobs)
    has_significant, contour_boxes = detect_significant_contours(prev_frame, curr_frame)
    if not has_significant:
        return False, "Motion too small (shadow/noise)", []

    # Layer 2b: HOG person detection (optional boost — if person found, definitely analyze)
    has_person, person_boxes = detect_person_hog(curr_frame)

    # Layer 3: Throttle
    if not can_analyze():
        return False, "Throttled (cooling down)", contour_boxes

    # Combine boxes for display
    all_boxes = person_boxes if person_boxes else contour_boxes
    reason = "Person detected (HOG)" if has_person else "Large movement detected"

    return True, reason, all_boxes


def draw_bounding_boxes(frame, boxes, color=(0, 255, 255), label="DETECTED"):
    """Draw bounding boxes on frame for detected objects."""
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


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

SYSTEM_PROMPT = """You are "Aegis", an Indian home surveillance AI. Analyze the camera frame and respond ONLY with valid JSON (no markdown, no fences):

{"threat_level":"low|medium|high","description":"2-3 sentences. Be SPECIFIC: clothing color, age, gender, exact action, location in frame.","description_telugu":"Same in conversational Telugu.","category":"delivery|visitor|family|stranger|animal|vehicle|empty|other","people_count":0,"action_needed":"Exact instruction for homeowner.","details":"Why this threat level."}

THREAT RULES:
- LOW: Empty/calm scene, delivery agents (Zomato/Swiggy/Amazon uniform), family, service workers, animals passing. Action: "No action needed"
- MEDIUM: Unknown person near house (not suspicious), unfamiliar group, parked vehicle, unclear intent. Action: "Monitor situation"
- HIGH: Stranger loitering/peeking/watching house, climbing walls/gates, tampering with locks/doors, masked person, fighting, weapons/tools, trespassing. Action: "Call police immediately" or "Alert family NOW"

HIGH ALERT RULE: For HIGH threats you MUST describe: (1) WHO — gender, clothing, build (2) WHAT — exact suspicious action (3) WHERE — location relative to house (4) WHY — why this is dangerous. Be urgent and clear.
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


# ---------------------------------------------------------------------------
# Background Analyzer — runs Claude in a separate thread (non-blocking)
# ---------------------------------------------------------------------------

class BackgroundAnalyzer:
    """
    Runs Claude analysis in a background thread so the video feed doesn't freeze.
    The main loop checks `get_result()` each frame — if a result is ready, it
    processes it; if not, the video keeps playing smoothly.
    """

    def __init__(self):
        self._result = None
        self._pending = False
        self._lock = threading.Lock()
        self._frame_b64 = None  # store b64 for logging

    @property
    def is_busy(self) -> bool:
        """True if a Claude analysis is currently running in the background."""
        with self._lock:
            return self._pending

    def submit(self, frame, use_mock: bool = False):
        """
        Submit a frame for analysis. Non-blocking — returns immediately.
        The analysis runs in a background thread.
        """
        with self._lock:
            if self._pending:
                return  # already analyzing, skip
            self._pending = True
            self._result = None

        # Store frame b64 for event logging later
        processed = preprocess_frame(frame)
        self._frame_b64 = frame_to_base64(processed)

        def _run():
            try:
                if use_mock:
                    result = analyze_frame_mock(processed)
                else:
                    result = analyze_frame(processed)

                with self._lock:
                    self._result = result
                    self._pending = False
            except Exception as e:
                print(f"[BG-ANALYZER] Error: {e}")
                with self._lock:
                    self._result = {"error": str(e)}
                    self._pending = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def get_result(self) -> tuple:
        """
        Check if a result is ready. Non-blocking.
        Returns (result_dict_or_None, frame_b64_or_None).
        """
        with self._lock:
            if self._result is not None:
                result = self._result
                b64 = self._frame_b64
                self._result = None
                self._frame_b64 = None
                return result, b64
        return None, None
