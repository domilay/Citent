"""
JUROR · FastAPI backend.

Exposes the real agent (trained LR triage + real NLI + real LLM calls)
to the demo front-end.  Also serves the static demo HTML so a single
`uvicorn server:app` is enough to bring the whole product up.

Run (dev):
    cd server/
    uvicorn server:app --reload --host 0.0.0.0 --port 8000

Run (prod):
    gunicorn -k uvicorn.workers.UvicornWorker server:app \
             --bind 0.0.0.0:8000 --workers 2 --timeout 120

Endpoints:
    GET  /                       → the demo HTML (auto-detects live backend)
    GET  /api/status             → which providers / NLI are wired
    POST /api/run-case           → run the real agent on one case
    GET  /api/cases              → built-in sample cases (the same 3 as the demo)
"""

from __future__ import annotations

import os
import json
import logging
import pathlib
from typing import Optional, Dict

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s · %(message)s",
)
log = logging.getLogger("juror.server")

# When this module runs from outside the package, make sibling-imports work.
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import run_case             # noqa: E402
from llm_clients import available_providers, DEFAULTS  # noqa: E402
from tools import get_triage, get_nli  # noqa: E402


HERE     = pathlib.Path(__file__).parent.resolve()
DEMO_DIR = (HERE.parent / "demo").resolve()

load_dotenv(HERE / ".env")


# ───────────────────────────────────────────────────────────────────
#  App
# ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="JUROR — citation-intent agent",
    description=(
        "Real agent backend (trained LR + DeBERTa NLI + LLM providers). "
        "When this backend is reachable the front-end runs in **server-real** "
        "mode; otherwise it transparently falls back to the in-browser demo."
    ),
    version="1.0.0",
)

# Allow the front-end to call /api/* from any origin.  Adjust in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ───────────────────────────────────────────────────────────────────
#  Built-in sample cases (same 3 the demo ships)
# ───────────────────────────────────────────────────────────────────
SAMPLE_CASES = {
    1: {
        "case_id": "ID_183878",
        "paragraph": (
            "In general, simultaneous confidence bands for a function f are constructed "
            "by studying the asymptotic distribution of the sup. The approach of "
            "Bickel and Rosenblatt (1973) relates this to a study of the distribution "
            "of a Gaussian process. This approach to constructing confidence bands has "
            "been used in the context of nonparametric estimation by, among others, "
            "Hardle (1989) for M-estimators and Claeskens and Van Keilegom (2003) for "
            "local polynomial likelihood estimators."
        ),
        "ref_clean_citation": "Hardle (1989)",
        "ground_truth": "Background",
    },
    2: {
        "case_id": "ID_99812",
        "paragraph": (
            "We follow the boosting framework of Buehlmann and Hothorn (2007), in which "
            "regularization is achieved indirectly via the application of penalized "
            "base learners. We extend their componentwise gradient boosting to handle "
            "conditional transformation models, but the iterative scheme and the "
            "stopping criterion remain as in the original work."
        ),
        "ref_clean_citation": "Buehlmann and Hothorn (2007)",
        "ground_truth": "Technical basis",
    },
    3: {
        "case_id": "ID_77541",
        "paragraph": (
            "Our proposed kernel estimator achieves a parametric rate of convergence "
            "on subsets, which is faster than the non-parametric rate reported by "
            "Yao et al. (2005) under the sparsely observed regime. In contrast to "
            "their approach, our method does not require functional principal "
            "component analysis as a preprocessing step."
        ),
        "ref_clean_citation": "Yao et al. (2005)",
        "ground_truth": "Comparison",
    },
}


# ───────────────────────────────────────────────────────────────────
#  Models
# ───────────────────────────────────────────────────────────────────
class CaseInput(BaseModel):
    case_id: str
    paragraph: str
    ref_clean_citation: str
    ground_truth: Optional[str] = None


class RunCaseRequest(BaseModel):
    case: CaseInput
    provider: str = "anthropic"
    model: Optional[str] = None


# ───────────────────────────────────────────────────────────────────
#  Health & status
# ───────────────────────────────────────────────────────────────────
@app.get("/api/status")
def api_status() -> Dict:
    """Tell the frontend what's actually wired on the server.

    Used to decide whether to show 'server-real' or 'browser-fallback' badge.
    """
    triage_ready = False
    triage_meta  = None
    try:
        t = get_triage()
        triage_ready = True
        # surface the held-out metrics next to the badge
        metrics_path = HERE / "models" / "metrics.json"
        if metrics_path.exists():
            triage_meta = json.loads(metrics_path.read_text("utf-8"))
    except Exception as e:                   # pragma: no cover
        log.warning("triage not loaded: %s", e)

    nli = get_nli()
    return {
        "ok": True,
        "triage": {
            "ready":   triage_ready,
            "model":   "tfidf_lr",
            "metrics": triage_meta and {
                "accuracy":   triage_meta.get("accuracy"),
                "macro_f1":   triage_meta.get("macro_f1"),
                "n_train":    triage_meta.get("n_train"),
            },
        },
        "fact_checker": {
            "nli_available": nli.available,
            "model":         nli.model_name if nli.available else "bleu_only",
        },
        "llm_providers": available_providers(),
        "version":       app.version,
    }


# ───────────────────────────────────────────────────────────────────
#  Cases & runs
# ───────────────────────────────────────────────────────────────────
@app.get("/api/cases")
def api_cases() -> Dict:
    return {"cases": SAMPLE_CASES}


@app.post("/api/run-case")
async def api_run_case(req: RunCaseRequest):
    if req.provider not in DEFAULTS:
        raise HTTPException(400, f"unknown provider: {req.provider}")
    if not available_providers().get(req.provider, False):
        raise HTTPException(
            400,
            f"server has no API key configured for '{req.provider}'. "
            f"Set the {DEFAULTS[req.provider][0]} environment variable.",
        )
    try:
        trace = await run_case(req.case.model_dump(), provider=req.provider,
                               model=req.model)
        return JSONResponse(trace)
    except Exception as e:                   # pragma: no cover
        log.exception("run-case failed")
        raise HTTPException(500, f"agent failure: {e}")


# ───────────────────────────────────────────────────────────────────
#  Static demo
# ───────────────────────────────────────────────────────────────────
if DEMO_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DEMO_DIR)), name="static")

    @app.get("/")
    def root() -> FileResponse:
        idx = DEMO_DIR / "index.html"
        if not idx.exists():
            raise HTTPException(404, "demo/index.html missing")
        return FileResponse(idx)

    @app.get("/style.css")
    def style_css() -> FileResponse:
        return FileResponse(DEMO_DIR / "style.css")

    @app.get("/animation.js")
    def animation_js() -> FileResponse:
        return FileResponse(DEMO_DIR / "animation.js")

    @app.get("/live-agent.js")
    def live_agent_js() -> FileResponse:
        return FileResponse(DEMO_DIR / "live-agent.js")

    @app.get("/traces/{name}")
    def trace_file(name: str) -> FileResponse:
        path = (DEMO_DIR / "traces" / name).resolve()
        # prevent escaping
        if not str(path).startswith(str((DEMO_DIR / "traces").resolve())):
            raise HTTPException(400, "bad path")
        if not path.exists():
            raise HTTPException(404, name)
        return FileResponse(path)
