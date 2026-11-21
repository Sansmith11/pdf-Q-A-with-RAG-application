@echo off
cd /d "%~dp0"
echo Starting PDF Q&A RAG System...
".venv\Scripts\streamlit.exe" run app.py
pause
