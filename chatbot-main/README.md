# دكتور استقبال (Doctor Reception) — Smart Clinic AI Receptionist

An AI-powered medical receptionist chatbot that has natural, bilingual (Arabic/English) conversations with patients, reasons about their symptoms using clinical frameworks, and — when warranted — generates a structured medical report they can bring to a doctor.

Built with **FastAPI**, **Google Gemini** (via `google-genai`), and a single-file HTML/JS chat frontend.

---

## ✨ Features

- **Natural bilingual conversation** — responds in Arabic or English depending on how the patient writes, with correct Arabic gender agreement.
- **Real-time streaming responses** over Server-Sent Events (SSE), token-by-token.
- **Clinical reasoning frameworks** used silently in the background: OPQRST, SAMPLE, demographics, and "Worst-Case-First" thinking to avoid missing dangerous presentations.
- **Hidden clinical scratchpad** — the model maintains a structured JSON state (age, symptoms, red flags, differential diagnosis, etc.) across the conversation without exposing it to the patient.
- **Opt-in medical report generation** — a structured report (JSON) is produced when the patient asks for one, or automatically for emergencies/urgent cases, and can be rendered as a printable PDF from the frontend.
- **Google Search grounding & URL context tools** — the model can ground its answers in trusted medical sources (Mayo Clinic, WHO, NHS, MedlinePlus, CDC, etc.) when it matters.
- **Bilingual content moderation** — a two-layer filter (regex + LLM classifier) blocks self-harm, drug, sexual, or off-topic content and responds with an appropriate safety message.
- **Multi-key rotation with cooldown & backoff** — API calls rotate across multiple Gemini API keys, with automatic cooldown for rate-limited keys, removal of invalid keys, and exponential backoff retries for transient (5xx) errors.
- **Automatic model fallback** — falls back from `gemini-2.5-flash` to `gemini-2.5-flash-lite` if the primary model keeps failing with transient errors.
- **Conversation summarisation** — long conversations are automatically summarised in the background to keep the model's context window manageable without losing key clinical facts.
- **Session management** — in-memory sessions with TTL-based expiry and automatic cleanup.
- **Basic rate limiting** — per-IP sliding-window rate limiting on chat endpoints.
- **Optional cross-session patient memory** — if the calling app provides a stable `user_id`, the bot remembers a patient's recent visits (chief complaint, differential, worst-case considered) and carries that context into their next session. Fully optional and off by default.

---

## 🏗️ Architecture

```
┌─────────────────┐      HTTP / SSE      ┌──────────────────┐
│   test_ui.html   │ ───────────────────► │     main.py       │
│ (chat frontend)  │ ◄─────────────────── │   (FastAPI app)   │
└─────────────────┘                       └─────────┬────────┘
                                                      │
                                    ┌─────────────────┼─────────────────┐
                                    ▼                                   ▼
                          ┌──────────────────┐              ┌────────────────────┐
                          │ chatbot_engine.py │              │ content_filter.py   │
                          │ (chat sessions,   │─────────────►│ (regex + LLM        │
                          │  streaming, state,│              │  moderation)        │
                          │  reports)         │              └─────────┬──────────┘
                          └─────────┬─────────┘                        │
                                    │                                  │
                                    ▼                                  ▼
                          ┌───────────────────────────────────────────────┐
                          │                key_manager.py                  │
                          │  Gemini API key rotation, cooldown, retries    │
                          └───────────────────────────┬─────────────────────┘
                                                        ▼
                                              Google Gemini API
                                        (gemini-2.5-flash / -flash-lite)
```

---

## 📁 Project Structure

```
.
├── main.py               # FastAPI app: routes, sessions, rate limiting, SSE streaming
├── chatbot_engine.py      # Core chat logic: system prompt, clinical reasoning, streaming, reports
├── key_manager.py         # Gemini API key rotation, cooldown, retry & fallback logic
├── content_filter.py      # Bilingual content moderation (regex + LLM classifier)
├── memory_store.py         # Optional cross-session patient memory (keyed by user_id)
├── test_ui.html            # Standalone HTML/JS chat client for local testing
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔧 Requirements

- Python 3.10+
- One or more **Google Gemini API keys** ([Google AI Studio](https://aistudio.google.com/))

Dependencies (see `requirements.txt`):

```
fastapi==0.115.0
uvicorn==0.30.0
google-genai==1.16.1
python-multipart==0.0.9
python-dotenv==1.0.1
```

---

## ⚙️ Setup

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
# One or more Gemini API keys, numbered sequentially.
# The app rotates across all of them automatically.
GEMINI_KEY_1=your_gemini_api_key_here
GEMINI_KEY_2=your_second_gemini_api_key_here

# Comma-separated list of allowed frontend origins (CORS).
# If omitted, defaults to localhost + file:// for local development only.
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com

# Optional: where per-patient memory JSON files are stored on disk.
# Defaults to ./patient_memory_data if not set.
PATIENT_MEMORY_DIR=patient_memory_data
```

> ⚠️ **Never commit your `.env` file.** It's already excluded via `.gitignore`.

Adding more keys (`GEMINI_KEY_3`, `GEMINI_KEY_4`, ...) increases your effective rate limit, since `key_manager.py` rotates across all configured keys and puts a key in cooldown only when *it* gets rate-limited.

### 3. Run the API server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### 4. Open the test frontend

Simply open `test_ui.html` in a browser (double-click it, or serve it with any static file server). By default it points to `http://localhost:8000` — you can change the API URL from the UI itself if your backend runs elsewhere.

---

## 🔌 API Reference

| Method   | Endpoint              | Description                                                  |
|----------|-----------------------|----------------------------------------------------------------|
| `POST`   | `/chat/start`          | Creates a new chat session, returns a `session_id`.             |
| `POST`   | `/chat/message`        | Sends a message, returns the full response (non-streaming).     |
| `POST`   | `/chat/stream`         | Sends a message, streams the response back over SSE.            |
| `GET`    | `/chat/{session_id}/report` | Retrieves the medical report generated for the session (if any). |
| `DELETE` | `/chat/{session_id}`   | Ends and removes a session.                                       |
| `GET`    | `/health`              | Health check — reports loaded keys and active session count.     |

### `POST /chat/start` — form fields

| Field     | Type   | Required | Description                                                                 |
|-----------|--------|----------|-------------------------------------------------------------------------------|
| `user_id` | string | No       | Stable patient identity for cross-session memory. Omit for a stateless session. |

### `POST /chat/message` and `/chat/stream` — form fields

| Field        | Type   | Description                  |
|--------------|--------|-------------------------------|
| `session_id` | string | Session ID from `/chat/start` |
| `message`    | string | The patient's message         |

### SSE event format (`/chat/stream`)

The stream sends plain `data:` lines with the following possible prefixes:

```
data: <plain text chunk>            → a piece of the assistant's reply
data: [SOURCES]<json_array>         → grounding sources used in the reply
data: [REPORT]<json_object>         → a generated medical report
data: [ERROR]<error_message>        → an error occurred
data: [DONE]                        → the turn is complete
```

---

## 🛡️ Safety & Content Moderation

`content_filter.py` runs on every incoming patient message before it reaches the model:

1. **Hard regex block** — instantly blocks messages matching self-harm, illegal drugs, or sexual content patterns (Arabic + English).
2. **Scope check + LLM classification** — messages that touch clearly off-topic areas (politics, finance, hacking, etc.) are additionally classified by a lightweight Gemini call to confirm whether they should be blocked, failing open to "safe" if the classifier itself is unavailable.

Appropriate bilingual messages are returned to the patient for each block reason (self-harm, drugs, sexual, off-topic), including a self-harm message that encourages reaching out to a mental health professional or crisis line.

---

## 🩺 Clinical Behaviour Notes

The system prompt in `chatbot_engine.py` enforces several hard boundaries on the model:

- Gives a *reasoned clinical impression*, never a confirmed/final diagnosis.
- Never names a specific medication, brand, or dosage for the patient to personally take.
- Immediately flags emergency red flags (chest pain, stroke signs, loss of consciousness, etc.) and tells the patient to go to the nearest ER, generating a report automatically to bring with them.
- Always closes clinical impressions with a caveat that a doctor should confirm the assessment.

This project is a **decision-support / triage assistant**, not a replacement for professional medical diagnosis or care.

---

## 🔑 Key Rotation, Retries & Fallback

`key_manager.py` provides:

- **Round-robin key rotation** across all configured `GEMINI_KEY_*` values.
- **Cooldown with exponential backoff** for keys that hit rate limits (up to `MAX_COOLDOWN_SECONDS`, default 300s).
- **Automatic removal** of keys that turn out to be invalid.
- **Transient error retries** (502/503/504/"overloaded") with exponential backoff, separate from key-rotation attempts.
- **Model fallback**: if `gemini-2.5-flash` keeps failing transiently, the session automatically switches to `gemini-2.5-flash-lite` for the remainder of the conversation, preserving chat history.

---

## 🧠 Patient Memory — Integration Guide for the App/Website Team

The bot can remember a patient across separate visits (different sessions, different days) — but **only if the app sends a stable `user_id`**. Without it, every session is stateless, exactly like before. This section is what the app/website side needs to do.

### What you need to send

Add one field, `user_id`, to your existing call to `POST /chat/start`:

```http
POST /chat/start
Content-Type: application/x-www-form-urlencoded

user_id=<your_id_for_this_patient>
```

Response is unchanged — you still get back a `session_id` and use it for `/chat/message` / `/chat/stream` / etc. exactly as before. `user_id` is only needed on `/chat/start`; you don't need to send it again on later calls in the same session.

### Requirements for `user_id`

- **It must be stable per patient** — the same patient must get the same `user_id` every time they open a new chat, or the bot won't recognise them as a returning patient.
- **It must NOT change between visits** — don't generate a new random id per session (that's what `session_id` is already for).
- **Any format is fine** — it doesn't have to be a phone number. Whatever unique, stable identifier you already have for a patient/account in your system works: an internal DB id, an account UUID, a national ID hash, etc. Just be consistent.
- **If you don't have patient identity/login sorted out yet**, don't send `user_id` at all — the bot works completely normally without it. There's no rush to wire this up before launch; it's an enhancement, not a dependency.

### What happens once you send it

- On the patient's first-ever session, nothing changes — there's no history yet.
- On any later session with the same `user_id`, the bot silently reviews the patient's last few visits (chief complaint, what was discussed, the most serious possibility considered) before responding to their first message. The patient never sees this — it's just background context for the model.
- After every reply, the bot updates that patient's record with the current visit's clinical picture, so the "memory" always reflects the latest visit.

### Data considerations

- Patient memory is stored as one JSON file per `user_id` on the server's disk, under `PATIENT_MEMORY_DIR` (defaults to `patient_memory_data/`, configurable via `.env` — see Setup section). It is **not** sent anywhere external besides the Gemini API call itself (as part of the context for that patient's own conversation).
- Each patient's file keeps their last 20 visits; older ones are dropped automatically.
- There's currently no deletion/right-to-be-forgotten endpoint exposed over the API — if you need one (e.g. for GDPR-style data requests), flag it and it can be added.

---

## 📌 Notes on Production Deployment

- Sessions are stored **in-memory** (`_sessions` dict in `main.py`) — they will not survive a server restart and won't work across multiple processes/instances without a shared session store (e.g. Redis).
- Set `ALLOWED_ORIGINS` explicitly before deploying — the default includes `null` (file:// origin) for local development only, which you should remove in production.
- Basic in-memory, per-IP rate limiting is included, but for production use behind a real gateway/WAF is recommended.
- Consider persisting reports and conversation logs to a database if you need audit trails or multi-device access.

---

## 📄 License

Add your license of choice here.
