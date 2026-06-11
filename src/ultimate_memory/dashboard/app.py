from __future__ import annotations

import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from ..config import load_settings
from ..router import MemoryRouter

app = FastAPI(title="Ultimate Memory Dashboard")
router = MemoryRouter()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    state = router.dashboard_state()
    health = state["health"]
    candidates = state["candidates"]
    audit = state["recent_audit"]
    indexed = state["indexed_sources"]
    return f"""
    <!doctype html>
    <html>
      <head>
        <title>Ultimate Memory</title>
        <style>
          body {{ font-family: Segoe UI, sans-serif; margin: 32px; color: #1f2933; }}
          h1, h2 {{ margin-bottom: 8px; }}
          code, pre {{ background: #f4f6f8; padding: 8px; border-radius: 6px; display: block; overflow: auto; }}
          .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }}
          .panel {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 16px; }}
          .ok {{ color: #047857; }}
          .bad {{ color: #b42318; }}
          input {{ width: 70%; padding: 8px; }}
          button {{ padding: 8px 12px; }}
        </style>
      </head>
      <body>
        <h1>Ultimate Memory</h1>
        <form method="post" action="/search">
          <input name="query" placeholder="Search memory" />
          <button>Search</button>
        </form>
        <h2>Health</h2>
        <div class="grid">
          {''.join(f'<div class="panel"><strong>{key}</strong><br><span class="{_cls(value)}">{value}</span></div>' for key, value in health.items())}
        </div>
        <h2>Staged Candidates</h2>
        <pre>{_safe(candidates)}</pre>
        <h2>Recent Audit</h2>
        <pre>{_safe(audit)}</pre>
        <h2>Indexed Sources</h2>
        <pre>{_safe(indexed)}</pre>
      </body>
    </html>
    """


@app.post("/search", response_class=HTMLResponse)
def search(query: str = Form(...)) -> str:
    result = router.search(query=query, limit=10)
    return f"""
    <!doctype html>
    <html>
      <head><title>Ultimate Memory Search</title></head>
      <body style="font-family: Segoe UI, sans-serif; margin: 32px;">
        <a href="/">Back</a>
        <h1>Search</h1>
        <pre style="background:#f4f6f8;padding:16px;border-radius:8px;white-space:pre-wrap;">{_safe(result)}</pre>
      </body>
    </html>
    """


def _cls(value: object) -> str:
    if value is True or value == "full":
        return "ok"
    if value is False or value == "fallback":
        return "bad"
    return ""


def _safe(value: object) -> str:
    import html
    import json

    return html.escape(json.dumps(value, indent=2, ensure_ascii=True, default=str))


def main() -> None:
    settings = load_settings()
    uvicorn.run(app, host=settings.dashboard.host, port=settings.dashboard.port)

