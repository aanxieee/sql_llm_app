# sql_llm_app — Environment setup

This workspace contains helper files to create a Python virtual environment for a small SQL + LLM application.

Files added:
- `requirements.txt` — suggested dependencies for an SQL + LLM app (openai, langchain, sqlalchemy, etc.)
- `setup.ps1` — PowerShell helper to create the venv and optionally install dependencies
- `.gitignore` — ignores `.venv` and common Python artifacts

Quick steps (PowerShell, Windows):

1) Allow script execution for this session and run the setup script (creates `.venv`):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\setup.ps1
```

2) To also create the venv and install dependencies in one step:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\setup.ps1 -install
```

3) Activate the environment in your current PowerShell session (example):

```powershell
& .\.venv\Scripts\Activate.ps1
# then verify
python -V
pip list
```

Notes
- The script assumes `python` is on your PATH and is Python 3.10+. Adjust `requirements.txt` as needed.
- If you prefer `venv` directory name `venv` instead of `.venv`, pass `-venvPath "venv"` to `setup.ps1`.

Next steps I can do for you
- Run the setup script now and install dependencies (I will not run it without your permission).
- Change the dependency list or pin exact versions.
- Create a starter `app.py` / sample to verify the environment.
