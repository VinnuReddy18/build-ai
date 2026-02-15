# 🛡️ Home Guard AI — Intelligent Surveillance System

AI-powered home surveillance system built with **OpenCV**, **Claude AI**, **Streamlit**, and **Twilio**. Designed for Indian households with Telugu/Hindi regional alerts.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT UI (main.py)                │
│  ┌──────────────────┐    ┌────────────────────────────┐ │
│  │  📹 Live Camera  │    │  📋 Activity Feed          │ │
│  │  - Webcam (0)    │    │  - Real-time events        │ │
│  │  - Video Upload  │    │  - Threat-level badges     │ │
│  │  - Motion overlay│    │  - Event thumbnails        │ │
│  └────────┬─────────┘    └──────────┬─────────────────┘ │
└───────────┼──────────────────────────┼──────────────────┘
            │                          │
     ┌──────▼──────────┐       ┌───────▼────────┐
     │ vision_engine.py│       │  database.py   │
     │ - Motion detect │       │ - PostgreSQL   │
     │ - Frame resize  │       │ - SQLite local │
     │ - Claude API    │       │ - Event CRUD   │
     └──────┬──────────┘       └────────────────┘
            │
     ┌──────▼──────────┐
     │   alerts.py     │
     │ - Twilio SDK    │
     │ - WhatsApp msg  │
     └─────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Claude API key | For AI analysis |
| `TWILIO_ACCOUNT_SID` | Twilio SID | For WhatsApp alerts |
| `TWILIO_AUTH_TOKEN` | Twilio token | For WhatsApp alerts |
| `TWILIO_WHATSAPP_FROM` | Twilio sandbox number | For WhatsApp alerts |
| `ADMIN_WHATSAPP_TO` | Your WhatsApp number | For WhatsApp alerts |
| `DATABASE_URL` | PostgreSQL URL | Optional (SQLite fallback) |

### 3. Run

```bash
streamlit run main.py
```

Open `http://localhost:8501` in your browser.

## 🎯 Features

- **🎥 Live Webcam Feed** — `cv2.VideoCapture(0)` for real-time monitoring
- **📁 Video Upload** — Upload `.mp4` / `.avi` files for analysis
- **⚡ Smart Sampling** — Only processes frames with >5% pixel change
- **🧠 Claude AI Analysis** — Threat identification with contextual awareness
- **🌐 Regional Alerts** — Bilingual descriptions in English + Telugu
- **📱 WhatsApp Alerts** — Instant HIGH threat notifications via Twilio
- **🗄️ Event Logging** — PostgreSQL with SQLite fallback
- **🔒 Privacy Mode** — One-click start/stop monitoring

## 🏆 Demo Checklist

1. ✅ Walk in front of camera → show WhatsApp alert on phone
2. ✅ Open Security Log → show AI-identified past events
3. ✅ Explain: OpenCV saves tokens, Claude only for reasoning
4. ✅ Show motion detection overlay (green = stable, red = motion)
5. ✅ Show threat-level badges (🟢 Low / 🟡 Medium / 🔴 High)
