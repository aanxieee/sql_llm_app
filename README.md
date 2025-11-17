# NL → SQL Streamlit App (Gemini + LangChain)

This app lets you:
- Ask natural-language questions about your database.
- Convert them to SQL using Google Gemini (via LangChain).
- Run the SQL safely against your DB.
- See results, quick charts, and an insight summary.
- Log feedback (correct / incorrect / partial) for later tuning.

## Setup

1. **Create a virtual environment** (optional but recommended):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

## Environment Variables (create `.env`)

Create a file named `.env` in the project root (same folder as `app.py`) using the template below or copy from `.env.example`:

```env
GOOGLE_API_KEY=your_google_api_key_here
DATABASE_URL=sqlite:///data.db
FEEDBACK_LOG_PATH=feedback_log.jsonl
```

Notes:
- Get a Google API key from Google AI Studio (Gemini). The variable name must be `GOOGLE_API_KEY` for `langchain_google_genai`.
- For a different DB file or location, change `DATABASE_URL`. Example absolute path: `sqlite:///D:/learning/projects/sql_llm_app/data.db`.
- If `GOOGLE_API_KEY` is missing, the Streamlit app will show an error on startup. Add the key and reload.

## Initialize the SQLite database

Run the lightweight scaffolding to create a sample `sales` table and seed data:

```powershell
& .\.venv\Scripts\python.exe database.py
```

This will create `data.db` if it does not exist and insert a few sample rows. You can then ask questions like:

- "Total sales per customer"
- "Average sale amount"
