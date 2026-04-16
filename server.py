from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import os
import sys

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PY  = os.path.join(BASE_DIR, "newsflow", "src", "newsflow", "main.py")
DIGEST   = os.path.join(BASE_DIR, "output", "digest.html")

# ── Scheduler ────────────────────────────────────────────────────────────────
def run_pipeline_job():
    subprocess.Popen([sys.executable, MAIN_PY])

scheduler = BackgroundScheduler()
scheduler.add_job(run_pipeline_job, "cron", hour=7, minute=0)
scheduler.start()

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_digest():
    if not os.path.exists(DIGEST):
        return "<h1>No digest yet — run the pipeline first</h1>"
    with open(DIGEST, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/run")
def run_pipeline():
    subprocess.Popen([sys.executable, MAIN_PY])
    return {"status": "pipeline started"}