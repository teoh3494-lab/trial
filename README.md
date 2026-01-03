# Tube Trends Buddy (Streamlit)

Local YouTube trends research tool inspired by “Tube Trends Buddy”.

## Features
- Super Search Engine with YouTube Data API v3 (search + videos.list).
- Save results to local SQLite bank, with category and notes.
- Track videos + snapshots, VPH (Views per Hour) detection.
- Heatmap for best upload time by channel.
- Market dashboard with revenue estimation + Viral Rate Score.
- Gemini AI idea generator (titles, hooks, thumbnail concept).
- Snapshot CLI runner for Windows Task Scheduler.

## Requirements
- Python 3.10+
- YouTube Data API v3 key
- Gemini API key (optional, for Idea Generator)

## Setup (Windows)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` from `.env.example`:
```bash
copy .env.example .env
```
Edit `.env` and add your API keys.

## Run the App
```bash
streamlit run app.py
```

## Optional: Local Transcription (Captions)
```bash
pip install -r requirements-extra.txt
```

## Snapshot Runner (for scheduling)
The Streamlit app cannot reliably run background jobs. Use `snapshot_runner.py` in Task Scheduler.

### Manual Run
```bash
python snapshot_runner.py
```

### Task Scheduler (Example)
1. Open **Task Scheduler** → **Create Basic Task**.
2. Trigger: Daily, Repeat every 30 or 60 minutes.
3. Action: **Start a program**.
4. Program/script: `C:\Path\To\Python\python.exe`
5. Add arguments: `C:\Path\To\Project\snapshot_runner.py`
6. Start in: `C:\Path\To\Project`

## Notes
- `search.list` is costly; `videos.list` is cheaper. The app caches search & details for 300s.
- Do not share your API keys.

## Project Structure
- `app.py` – Streamlit UI
- `yt_api.py` – YouTube API wrapper + helpers
- `db.py` – SQLite schema and CRUD
- `scoring.py` – Viral rate score heuristic
- `gemini_ai.py` – Gemini prompt + generation
- `snapshot_runner.py` – CLI snapshot runner
- `requirements.txt` – Dependencies
- `.env.example` – Environment variable template
