"""
main.py — Aegis AI Surveillance System.
Premium Streamlit UI with glassmorphism, animations, and real-time monitoring.
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

from database import init_db, log_event, get_recent_events, get_event_stats
from vision_engine import (
    preprocess_frame, frame_to_base64, detect_motion, get_motion_score,
    draw_status_overlay, draw_motion_border, draw_bounding_boxes,
    analyze_frame, analyze_frame_mock, is_claude_configured,
    can_analyze, should_call_claude, BackgroundAnalyzer,
)
from alerts import send_high_threat_alert, is_configured as is_twilio_configured, is_voice_configured

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Aegis — AI Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — Glassmorphism + Glow + Animations
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #030712;
        --bg-card: rgba(15, 23, 42, 0.6);
        --bg-glass: rgba(15, 23, 42, 0.4);
        --border: rgba(56, 189, 248, 0.08);
        --border-hover: rgba(56, 189, 248, 0.2);
        --accent: #38bdf8;
        --accent2: #818cf8;
        --danger: #ef4444;
        --warning: #f59e0b;
        --success: #10b981;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #475569;
    }

    * { font-family: 'Inter', sans-serif; }

    /* ── Animated background ── */
    .stApp {
        background: var(--bg-primary);
    }

    /* ── Keyframes ── */
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 15px rgba(56,189,248,0.1); }
        50% { box-shadow: 0 0 30px rgba(56,189,248,0.2); }
    }
    @keyframes live-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes slide-up {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes border-pulse {
        0%, 100% { border-color: rgba(239,68,68,0.3); }
        50% { border-color: rgba(239,68,68,0.6); }
    }

    /* ── Header ── */
    .aegis-header {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 1rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        border: 1px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: space-between;
        animation: pulse-glow 4s ease-in-out infinite;
    }
    .aegis-header .logo-group {
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .aegis-header .shield {
        font-size: 2rem;
        filter: drop-shadow(0 0 8px rgba(56,189,248,0.4));
    }
    .aegis-header h1 {
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.6rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: 2px;
    }
    .aegis-header .tagline {
        color: var(--text-muted);
        margin: 0;
        font-size: 0.65rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        font-weight: 500;
    }
    .live-badge {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .live-badge.active {
        background: rgba(16,185,129,0.1);
        color: var(--success);
        border: 1px solid rgba(16,185,129,0.25);
    }
    .live-badge.active .dot {
        width: 6px; height: 6px;
        background: var(--success);
        border-radius: 50%;
        animation: live-dot 1.5s ease-in-out infinite;
    }
    .live-badge.paused {
        background: rgba(245,158,11,0.1);
        color: var(--warning);
        border: 1px solid rgba(245,158,11,0.25);
    }

    /* ── Stat cards ── */
    .stats-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.6rem;
        margin-bottom: 0.8rem;
    }
    .stat-card {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.8rem 0.6rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        border-color: var(--border-hover);
        transform: translateY(-2px);
    }
    .stat-card .val {
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-card .lbl {
        color: var(--text-muted);
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 0.15rem 0 0 0;
    }
    .c-cyan { color: var(--accent); }
    .c-red { color: var(--danger); }
    .c-yellow { color: var(--warning); }
    .c-green { color: var(--success); }

    /* ── Event cards ── */
    .event-card {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem;
        transition: all 0.25s ease;
        animation: slide-up 0.3s ease;
    }
    .event-card:hover {
        border-color: var(--border-hover);
        background: rgba(15, 23, 42, 0.7);
    }
    .event-card.high-event {
        border-color: rgba(239,68,68,0.25);
        animation: border-pulse 2s ease-in-out infinite, slide-up 0.3s ease;
    }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        font-size: 0.6rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-high {
        background: rgba(239,68,68,0.12);
        color: var(--danger);
        border: 1px solid rgba(239,68,68,0.25);
        box-shadow: 0 0 8px rgba(239,68,68,0.15);
    }
    .badge-medium {
        background: rgba(245,158,11,0.12);
        color: var(--warning);
        border: 1px solid rgba(245,158,11,0.25);
    }
    .badge-low {
        background: rgba(16,185,129,0.12);
        color: var(--success);
        border: 1px solid rgba(16,185,129,0.25);
    }

    /* ── Analysis card ── */
    .analysis-card {
        background: var(--bg-glass);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-top: 0.4rem;
        animation: slide-up 0.4s ease;
    }
    .analysis-card .a-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .analysis-card .a-title {
        color: var(--accent);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 0;
    }
    .analysis-card .a-desc {
        color: var(--text-primary);
        font-size: 0.82rem;
        line-height: 1.4;
        margin: 0.3rem 0;
    }
    .analysis-card .a-telugu {
        color: var(--text-secondary);
        font-size: 0.78rem;
        font-style: italic;
        margin: 0.2rem 0;
    }
    .action-alert {
        margin-top: 0.4rem;
        padding: 0.35rem 0.7rem;
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.2);
        border-radius: 8px;
        color: var(--danger);
        font-size: 0.75rem;
        font-weight: 600;
        animation: border-pulse 2s infinite;
    }

    /* ── Camera placeholder ── */
    .cam-off {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        border-radius: 14px;
        padding: 50px 20px;
        text-align: center;
        border: 1px solid var(--border);
    }
    .cam-off h3 { margin: 0; font-weight: 700; }
    .cam-off p { color: var(--text-muted); margin: 0.3rem 0 0 0; font-size: 0.85rem; }

    /* ── Motion bar ── */
    .motion-bar {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        border-radius: 8px;
        padding: 0.35rem 0.7rem;
        font-size: 0.7rem;
        color: var(--text-secondary);
        border: 1px solid var(--border);
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.3rem;
    }

    /* ── Feed header ── */
    .feed-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .feed-title {
        color: var(--text-primary);
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin: 0;
    }
    .feed-count {
        color: var(--text-muted);
        font-size: 0.65rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Section label ── */
    .section-label {
        color: var(--text-muted);
        font-size: 0.6rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    header[data-testid="stHeader"] { background: transparent; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(3, 7, 18, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .stRadio > label { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "monitoring": False, "camera": None, "prev_frame": None,
        "events_log": [], "total_analyses": 0, "motion_events": 0,
        "source_type": "webcam", "video_file_path": None,
        "last_analysis_result": None, "analyzer": BackgroundAnalyzer(),
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
    if st.session_state.camera is not None:
        st.session_state.camera.release()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error(f"❌ Failed to open: {source}")
        return False
    st.session_state.camera = cap
    st.session_state.prev_frame = None
    return True

def release_camera():
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
        st.session_state.prev_frame = None

def read_frame():
    if st.session_state.camera is None:
        return None
    ret, frame = st.session_state.camera.read()
    if not ret:
        if st.session_state.source_type == "video":
            st.session_state.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = st.session_state.camera.read()
            if not ret: return None
        else:
            return None
    return frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def threat_badge(level: str) -> str:
    l = level.lower()
    cls = f"badge-{l}" if l in ("high", "medium", "low") else "badge-low"
    emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(l, "⚪")
    return f'<span class="badge {cls}">{emoji} {level.upper()}</span>'


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
if st.session_state.monitoring:
    badge = '<div class="live-badge active"><span class="dot"></span>LIVE</div>'
else:
    badge = '<div class="live-badge paused">⏸ STANDBY</div>'

st.markdown(f"""
<div class="aegis-header">
    <div class="logo-group">
        <span class="shield">🛡️</span>
        <div>
            <h1>AEGIS</h1>
            <p class="tagline">AI-Powered Surveillance</p>
        </div>
    </div>
    {badge}
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    # Source
    st.markdown('<p class="section-label">📹 Video Source</p>', unsafe_allow_html=True)
    source_type = st.radio(
        "Source:",
        ["🎥 Live Webcam", "📁 Upload Video"],
        index=0 if st.session_state.source_type == "webcam" else 1,
        label_visibility="collapsed",
    )

    if "Live Webcam" in source_type:
        st.session_state.source_type = "webcam"
        camera_index = st.number_input("Camera", min_value=0, max_value=10, value=0, step=1)
    else:
        st.session_state.source_type = "video"
        uploaded = st.file_uploader("Upload", type=["mp4", "avi", "mov", "mkv"])
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            tfile.flush()
            st.session_state.video_file_path = tfile.name
            st.success(f"✅ {uploaded.name}")

    st.markdown("")

    # Controls
    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("▶ START", use_container_width=True, type="primary"):
            st.session_state.monitoring = True
            if st.session_state.source_type == "webcam":
                open_camera(camera_index)
            elif st.session_state.video_file_path:
                open_camera(st.session_state.video_file_path)
    with col_stop:
        if st.button("■ STOP", use_container_width=True):
            st.session_state.monitoring = False
            release_camera()

    st.markdown("---")

    # Analysis mode
    use_mock = st.checkbox("Mock mode (no API)", value=not is_claude_configured())

    st.markdown("---")
    st.caption("🛡️ Aegis v2.0")
    st.caption("Claude AI • OpenCV • Twilio")


# ---------------------------------------------------------------------------
# STATS
# ---------------------------------------------------------------------------
stats = get_event_stats()
st.markdown(f"""
<div class="stats-row">
    <div class="stat-card"><p class="val c-cyan">{stats['total']}</p><p class="lbl">Total</p></div>
    <div class="stat-card"><p class="val c-red">{stats['high']}</p><p class="lbl">Critical</p></div>
    <div class="stat-card"><p class="val c-yellow">{stats['medium']}</p><p class="lbl">Warnings</p></div>
    <div class="stat-card"><p class="val c-green">{stats['low']}</p><p class="lbl">Normal</p></div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------
col_camera, col_feed = st.columns([3, 2])


# ----- LEFT: Camera Feed -----
with col_camera:
    camera_placeholder = st.empty()
    motion_info = st.empty()
    analysis_result_box = st.empty()

    if st.session_state.monitoring:
        frame = read_frame()

        if frame is not None:
            processed = preprocess_frame(frame)

            motion = detect_motion(st.session_state.prev_frame, processed)
            motion_score = get_motion_score(st.session_state.prev_frame, processed)
            should_analyze, filter_reason, bboxes = should_call_claude(
                st.session_state.prev_frame, processed
            )

            # Background analysis
            analyzer = st.session_state.analyzer
            bg_result, bg_b64 = analyzer.get_result()
            if bg_result and "error" not in bg_result:
                st.session_state.last_analysis_result = bg_result
                st.session_state.total_analyses += 1
                log_event(
                    bg_result.get("threat_level", "low"),
                    bg_result.get("description", "No description"),
                    bg_b64 or "",
                )
                if bg_result.get("threat_level", "").lower() == "high":
                    send_high_threat_alert(
                        bg_result.get("description", ""),
                        bg_result.get("description_telugu", ""),
                        bg_result.get("action_needed", ""),
                    )
            elif bg_result and "error" in bg_result:
                analysis_result_box.warning(f"Analysis error: {bg_result['error']}")

            # Overlays
            display_frame = processed.copy()
            is_analyzing = analyzer.is_busy
            if is_analyzing:
                status_text = "ANALYZING..."
                status_color = (248, 189, 56)
            elif should_analyze:
                status_text = f"DETECTED: {filter_reason}"
                status_color = (68, 68, 239)
            elif motion:
                status_text = f"MOTION ({filter_reason})"
                status_color = (11, 158, 245)
            else:
                status_text = "MONITORING"
                status_color = (16, 185, 129)

            display_frame = draw_status_overlay(display_frame, status_text, status_color)
            display_frame = draw_motion_border(display_frame, motion)
            if bboxes:
                display_frame = draw_bounding_boxes(display_frame, bboxes)

            display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

            # Motion bar
            if is_analyzing:
                s_icon, s_text = "🟠", "Analyzing..."
            elif should_analyze:
                s_icon, s_text = "🔴", "Sending"
            elif motion:
                s_icon, s_text = "🟡", "Filtered"
            else:
                s_icon, s_text = "🟢", "Stable"

            motion_info.markdown(
                f'<div class="motion-bar">'
                f'{s_icon} {s_text} &nbsp;│&nbsp; '
                f'Motion: {motion_score:.1%} &nbsp;│&nbsp; '
                f'{filter_reason}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Submit to analyzer
            if should_analyze and not analyzer.is_busy:
                st.session_state.motion_events += 1
                analyzer.submit(processed, use_mock=use_mock)

            # Analysis result
            last = st.session_state.last_analysis_result
            if last:
                action = last.get('action_needed', '')
                people = last.get('people_count', '')
                threat_lvl = last.get('threat_level', 'low')

                action_html = ""
                if action and threat_lvl.lower() in ("high", "medium"):
                    action_html = f'<div class="action-alert">⚡ {action}</div>'

                people_html = f' &nbsp;│&nbsp; 👥 {people}' if people else ''

                analysis_result_box.markdown(f"""
                <div class="analysis-card">
                    <div class="a-header">
                        <span class="a-title">Analysis</span>
                        {threat_badge(threat_lvl)}
                        <span style="color:var(--text-muted); font-size:0.7rem; font-family:'JetBrains Mono',monospace;">{last.get('category', '').upper()}{people_html}</span>
                    </div>
                    <p class="a-desc">{last.get('description', '')}</p>
                    <p class="a-telugu">{last.get('description_telugu', '')}</p>
                    {action_html}
                </div>
                """, unsafe_allow_html=True)

            st.session_state.prev_frame = processed.copy()

            # Frame pacing — ~5 FPS
            elapsed = time.time() - st.session_state.frame_time
            sleep_time = max(0.05, 0.2 - elapsed)
            time.sleep(sleep_time)
            st.session_state.frame_time = time.time()
            st.rerun()

        else:
            camera_placeholder.markdown("""
            <div class="cam-off">
                <h3 style="color:var(--danger);">⚠️ No Frame</h3>
                <p>Check your camera or video file.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        camera_placeholder.markdown("""
        <div class="cam-off">
            <h3 style="color:var(--accent);">🛡️ Aegis Ready</h3>
            <p>Press ▶ START to begin monitoring.</p>
        </div>
        """, unsafe_allow_html=True)


# ----- RIGHT: Activity Feed -----
with col_feed:
    event_list = get_recent_events(limit=10)
    count = len(event_list) if event_list else 0

    st.markdown(f"""
    <div class="feed-header">
        <p class="feed-title">📋 Activity Feed</p>
        <span class="feed-count">{count} events</span>
    </div>
    """, unsafe_allow_html=True)

    if event_list:
        for idx, event in enumerate(event_list):
            level = event.get("threat_level", "low")
            badge_html = threat_badge(level)
            desc = event.get("description", "No description")
            ts = event.get("timestamp", "")
            extra_cls = "high-event" if level.lower() == "high" else ""

            st.markdown(f"""
            <div class="event-card {extra_cls}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
                    {badge_html}
                    <span style="color:var(--text-muted); font-size:0.65rem; font-family:'JetBrains Mono',monospace;">{ts}</span>
                </div>
                <p style="margin:0; color:var(--text-primary); font-size:0.78rem; line-height:1.35;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

            # Thumbnail for latest 3
            if idx < 3 and event.get("image_data"):
                try:
                    img_bytes = base64.b64decode(event["image_data"])
                    img = Image.open(BytesIO(img_bytes))
                    st.image(img, width=150)
                except Exception:
                    pass
    else:
        st.markdown("""
        <div class="cam-off" style="padding:2rem;">
            <p style="color:var(--text-muted); margin:0; font-size:0.85rem;">No events yet. Start monitoring to begin.</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SECURITY LOG
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("🗂️ Security Log", expanded=False):
    all_events = get_recent_events(limit=50)
    if all_events:
        import pandas as pd
        df = pd.DataFrame(all_events)
        df = df[["id", "timestamp", "threat_level", "description"]]
        df.columns = ["ID", "Timestamp", "Threat", "Description"]
        st.dataframe(df, use_container_width=True, hide_index=True, column_config={
            "Threat": st.column_config.TextColumn(width="small"),
            "Description": st.column_config.TextColumn(width="large"),
        })
    else:
        st.info("No events yet.")


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
f1, f2, f3 = st.columns(3)
with f1:
    st.caption(f"🧠 Analyses: {st.session_state.total_analyses}")
with f2:
    st.caption(f"⚡ Detections: {st.session_state.motion_events}")
with f3:
    st.caption("🛡️ Aegis v2.0")
