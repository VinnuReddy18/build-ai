"""
main.py — Streamlit UI & Frame capture for Home Guard AI.

Features:
  - Live webcam feed via cv2.VideoCapture(0)
  - Video file upload support
  - Real-time activity feed from database
  - Privacy toggle (Start/Stop monitoring)
  - Motion detection with visual indicators
  - Claude AI analysis with threat-level badges
"""

import os
import time
import base64
import tempfile
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv(override=True)

# Import our modules
from database import init_db, log_event, get_recent_events, get_event_stats
from vision_engine import (
    preprocess_frame,
    frame_to_base64,
    detect_motion,
    get_motion_score,
    draw_status_overlay,
    draw_motion_border,
    draw_bounding_boxes,
    analyze_frame,
    analyze_frame_mock,
    is_claude_configured,
    can_analyze,
    should_call_claude,
    BackgroundAnalyzer,
)
from alerts import send_high_threat_alert, is_configured as is_twilio_configured

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Home Guard AI — Surveillance System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS for premium dark UI
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: #a0a0b0;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    .stat-card .stat-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .stat-card .stat-label {
        color: #a0a0b0;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0;
    }
    .stat-high .stat-value { color: #e94560; }
    .stat-med .stat-value { color: #f5a623; }
    .stat-low .stat-value { color: #0be881; }
    .stat-total .stat-value { color: #4fc3f7; }

    /* Event cards */
    .event-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        transition: transform 0.2s ease;
    }
    .event-card:hover {
        transform: translateY(-2px);
        border-color: rgba(233,69,96,0.3);
    }

    /* Threat badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-high { background: rgba(233,69,96,0.2); color: #e94560; border: 1px solid rgba(233,69,96,0.3); }
    .badge-medium { background: rgba(245,166,35,0.2); color: #f5a623; border: 1px solid rgba(245,166,35,0.3); }
    .badge-low { background: rgba(11,232,129,0.2); color: #0be881; border: 1px solid rgba(11,232,129,0.3); }

    /* Status indicator */
    .status-active {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-monitoring {
        background: rgba(11,232,129,0.15);
        color: #0be881;
        border: 1px solid rgba(11,232,129,0.3);
    }
    .status-paused {
        background: rgba(245,166,35,0.15);
        color: #f5a623;
        border: 1px solid rgba(245,166,35,0.3);
    }

    /* Camera feed container */
    .camera-container {
        background: #0a0a1a;
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid rgba(255,255,255,0.05);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }

    /* Config badges */
    .config-ok { color: #0be881; }
    .config-miss { color: #e94560; }

    /* Hide Streamlit's default elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "monitoring": False,
        "camera": None,
        "prev_frame": None,
        "events_log": [],
        "total_analyses": 0,
        "motion_events": 0,
        "source_type": "webcam",
        "video_file_path": None,
        "last_analysis_result": None,
        "analyzer": BackgroundAnalyzer(),
        "frame_time": time.time(),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ---------------------------------------------------------------------------
# Camera management
# ---------------------------------------------------------------------------
def open_camera(source=0):
    """Open a camera or video file. Store in session state."""
    if st.session_state.camera is not None:
        st.session_state.camera.release()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error(f"❌ Failed to open video source: {source}")
        return False

    st.session_state.camera = cap
    st.session_state.prev_frame = None
    return True


def release_camera():
    """Release the camera."""
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
        st.session_state.prev_frame = None


def read_frame():
    """Read a frame from the active camera."""
    if st.session_state.camera is None:
        return None

    ret, frame = st.session_state.camera.read()
    if not ret:
        # If video file reached the end, loop
        if st.session_state.source_type == "video":
            st.session_state.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = st.session_state.camera.read()
            if not ret:
                return None
        else:
            return None

    return frame


# ---------------------------------------------------------------------------
# Helper: render threat badge
# ---------------------------------------------------------------------------
def threat_badge(level: str) -> str:
    level_lower = level.lower()
    css_class = f"badge-{level_lower}" if level_lower in ("high", "medium", "low") else "badge-low"
    emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(level_lower, "⚪")
    return f'<span class="badge {css_class}">{emoji} {level.upper()}</span>'


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🛡️ Home Guard AI</h1>
    <p>Intelligent Surveillance System — Powered by Claude AI & OpenCV</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")

    # Status indicators
    claude_status = "✅ Connected" if is_claude_configured() else "❌ Not configured"
    twilio_status = "✅ Connected" if is_twilio_configured() else "❌ Not configured"

    st.markdown(f"""
    | Service | Status |
    |---------|--------|
    | **Claude AI** | {claude_status} |
    | **Twilio** | {twilio_status} |
    | **Database** | ✅ Ready |
    """)

    st.divider()

    # Source selection
    st.markdown("### 📹 Video Source")
    source_type = st.radio(
        "Select source:",
        ["🎥 Live Webcam", "📁 Upload Video"],
        index=0 if st.session_state.source_type == "webcam" else 1,
        label_visibility="collapsed",
    )

    if "Live Webcam" in source_type:
        st.session_state.source_type = "webcam"
        camera_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
    else:
        st.session_state.source_type = "video"
        uploaded = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded:
            # Save to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            tfile.flush()
            st.session_state.video_file_path = tfile.name
            st.success(f"✅ Video loaded: {uploaded.name}")

    st.divider()

    # Monitoring controls
    st.markdown("### 🔒 Privacy Controls")
    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("▶️ Start", use_container_width=True, type="primary"):
            st.session_state.monitoring = True
            if st.session_state.source_type == "webcam":
                open_camera(camera_index)
            elif st.session_state.video_file_path:
                open_camera(st.session_state.video_file_path)
    with col_stop:
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.monitoring = False
            release_camera()

    # Monitoring status
    if st.session_state.monitoring:
        st.markdown('<div class="status-active status-monitoring">🟢 MONITORING ACTIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-active status-paused">🟡 SYSTEM PAUSED</div>', unsafe_allow_html=True)

    st.divider()

    # Analysis mode
    st.markdown("### 🧠 Analysis Mode")
    use_mock = st.checkbox("Use mock analysis (no API)", value=not is_claude_configured())

    st.divider()

    st.markdown("""
    ### 📋 About
    **Home Guard AI** uses computer vision and AI to provide
    intelligent home surveillance. Built with:
    - 🔍 OpenCV for motion detection
    - 🧠 Claude AI for threat analysis
    - 📱 Twilio for WhatsApp alerts
    - 🗄️ PostgreSQL for event logging
    """)


# ---------------------------------------------------------------------------
# STATS BAR
# ---------------------------------------------------------------------------
stats = get_event_stats()
st.markdown("### 📊 Dashboard")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="stat-card stat-total">
        <p class="stat-value">{stats['total']}</p>
        <p class="stat-label">Total Events</p>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="stat-card stat-high">
        <p class="stat-value">{stats['high']}</p>
        <p class="stat-label">High Threats</p>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="stat-card stat-med">
        <p class="stat-value">{stats['medium']}</p>
        <p class="stat-label">Medium Alerts</p>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="stat-card stat-low">
        <p class="stat-value">{stats['low']}</p>
        <p class="stat-label">Low Events</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")


# ---------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------
col_camera, col_feed = st.columns([3, 2])


# ----- LEFT: Camera Feed -----
with col_camera:
    st.markdown("### 📹 Live Feed")

    camera_placeholder = st.empty()
    motion_info = st.empty()
    analysis_result_box = st.empty()

    if st.session_state.monitoring:
        frame = read_frame()

        if frame is not None:
            # Pre-process
            processed = preprocess_frame(frame)

            # Smart 3-layer pre-filtering
            motion = detect_motion(st.session_state.prev_frame, processed)
            motion_score = get_motion_score(st.session_state.prev_frame, processed)
            should_analyze, filter_reason, bboxes = should_call_claude(
                st.session_state.prev_frame, processed
            )

            # Check for completed background analysis
            analyzer = st.session_state.analyzer
            bg_result, bg_b64 = analyzer.get_result()
            if bg_result and "error" not in bg_result:
                st.session_state.last_analysis_result = bg_result
                st.session_state.total_analyses += 1

                # Log to database
                log_event(
                    bg_result.get("threat_level", "low"),
                    bg_result.get("description", "No description"),
                    bg_b64 or "",
                )

                # Trigger WhatsApp + Voice Call for HIGH threats
                if bg_result.get("threat_level", "").lower() == "high":
                    send_high_threat_alert(
                        bg_result.get("description", ""),
                        bg_result.get("description_telugu", ""),
                        bg_result.get("action_needed", ""),
                    )
            elif bg_result and "error" in bg_result:
                analysis_result_box.warning(f"Analysis error: {bg_result['error']}")

            # Draw overlays
            display_frame = processed.copy()
            is_analyzing = analyzer.is_busy
            if is_analyzing:
                status_text = "🧠 ANALYZING (Claude)"
                status_color = (255, 165, 0)  # Orange — Claude is thinking
            elif should_analyze:
                status_text = f"🎯 {filter_reason}"
                status_color = (0, 0, 255)  # Red — about to send
            elif motion:
                status_text = f"⚡ MOTION ({filter_reason})"
                status_color = (0, 165, 255)  # Orange — motion but filtered
            else:
                status_text = "MONITORING"
                status_color = (0, 255, 0)  # Green — all clear

            display_frame = draw_status_overlay(display_frame, status_text, status_color)
            display_frame = draw_motion_border(display_frame, motion)

            # Draw bounding boxes on detected objects
            if bboxes:
                display_frame = draw_bounding_boxes(display_frame, bboxes)

            # Convert BGR → RGB for Streamlit
            display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

            # Motion info
            analyzing_label = '🟠 Claude analyzing...' if is_analyzing else ('🔴 Sending' if should_analyze else '🟡 Filtered' if motion else '🟢 Stable')
            motion_info.markdown(
                f"**Motion:** `{motion_score:.2%}` | "
                f"**Filter:** {filter_reason} | "
                f"**Status:** {analyzing_label} | "
                f"**Throttle:** {'🟢 Ready' if can_analyze() else '🟡 Cooling down...'}"
            )

            # Submit to background analyzer (non-blocking!)
            if should_analyze and not analyzer.is_busy:
                st.session_state.motion_events += 1
                analyzer.submit(processed, use_mock=use_mock)

            # Show last analysis result
            last = st.session_state.last_analysis_result
            if last:
                action = last.get('action_needed', '')
                action_line = f"\n                - **⚡ Action:** {action}" if action else ""
                people = last.get('people_count', '')
                people_line = f"\n                - **👥 People:** {people}" if people != '' else ""
                analysis_result_box.markdown(f"""
                **🧠 Latest Analysis:**
                - **Threat:** {threat_badge(last.get('threat_level', 'low'))}
                - **Category:** {last.get('category', 'N/A')}{people_line}
                - **EN:** {last.get('description', '')}
                - **TE:** {last.get('description_telugu', '')}{action_line}
                """, unsafe_allow_html=True)

            # Update previous frame
            st.session_state.prev_frame = processed.copy()

            # Frame pacing — target ~5 FPS (Streamlit's realistic limit)
            elapsed = time.time() - st.session_state.frame_time
            target_delay = 1.0 / 5.0  # 5 FPS
            sleep_time = max(0.05, target_delay - elapsed)
            time.sleep(sleep_time)
            st.session_state.frame_time = time.time()
            st.rerun()

        else:
            camera_placeholder.markdown("""
            <div class="camera-container" style="padding:80px; text-align:center;">
                <h3 style="color:#e94560;">⚠️ No Frame Available</h3>
                <p style="color:#a0a0b0;">Check your camera connection or video file.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        camera_placeholder.markdown("""
        <div class="camera-container" style="padding:80px; text-align:center;">
            <h3 style="color:#f5a623;">🔒 System Paused</h3>
            <p style="color:#a0a0b0;">Click <b>▶️ Start</b> in the sidebar to begin monitoring.</p>
        </div>
        """, unsafe_allow_html=True)


# ----- RIGHT: Activity Feed -----
with col_feed:
    st.markdown("### 📋 Activity Feed")

    # Manual analyze button
    if st.button("🔄 Refresh Feed", use_container_width=True):
        st.rerun()

    events = get_recent_events(limit=10)

    if events:
        for idx, event in enumerate(events):
            level = event.get("threat_level", "low")
            badge_html = threat_badge(level)
            desc = event.get("description", "No description")
            ts = event.get("timestamp", "")

            # Build event card
            card_html = f"""
            <div class="event-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                    {badge_html}
                    <span style="color:#666; font-size:0.75rem;">{ts}</span>
                </div>
                <p style="margin:0; color:#d0d0d0; font-size:0.85rem;">{desc}</p>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            # Show thumbnail for latest 5 events only (keeps media cache small)
            if idx < 5 and event.get("image_data"):
                try:
                    img_bytes = base64.b64decode(event["image_data"])
                    img = Image.open(BytesIO(img_bytes))
                    st.image(img, width=180, caption=f"Frame #{event.get('id', '')}")
                except Exception:
                    pass

    else:
        st.markdown("""
        <div class="event-card" style="text-align:center; padding:2rem;">
            <p style="color:#a0a0b0; margin:0;">No events recorded yet.</p>
            <p style="color:#666; margin:0.3rem 0 0 0; font-size:0.8rem;">Start monitoring to detect activity.</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SECURITY LOG TAB (full history)
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🗂️ Security Log")

with st.expander("View Full Event History", expanded=False):
    all_events = get_recent_events(limit=50)
    if all_events:
        # Build a table
        import pandas as pd
        df = pd.DataFrame(all_events)
        df = df[["id", "timestamp", "threat_level", "description"]]
        df.columns = ["ID", "Timestamp", "Threat Level", "Description"]

        # Color-code threat levels
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Threat Level": st.column_config.TextColumn(width="small"),
                "Description": st.column_config.TextColumn(width="large"),
            },
        )
    else:
        st.info("No events in the security log yet.")


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
fcol1, fcol2, fcol3 = st.columns(3)
with fcol1:
    st.caption(f"🧠 Analyses: {st.session_state.total_analyses}")
with fcol2:
    st.caption(f"⚡ Motion Events: {st.session_state.motion_events}")
with fcol3:
    st.caption("🛡️ Home Guard AI v1.0 — Built for Hackathon")
