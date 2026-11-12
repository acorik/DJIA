"""API local del AI DJ Set Copilot + servidor de la interfaz web estática."""
from __future__ import annotations
import dataclasses
import os

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config
from .analysis_service import scan_folder
from .llm_brief import build_set_brief, brief_to_dict
from .set_planner import plan_order
from .transition_designer import design_transition

app = FastAPI(title="AI DJ Set Copilot")

FRONTEND_DIR = os.path.join(config.BASE_DIR, "frontend")


class AnalyzeRequest(BaseModel):
    folder_path: str


class PlanRequest(BaseModel):
    folder_path: str
    style_prompt: str


def _index_tracks_by_path(tracks: list[dict]) -> dict[str, dict]:
    return {t["file_path"]: t for t in tracks if "error" not in t}


@app.post("/api/analyze")
def api_analyze(req: AnalyzeRequest):
    if not os.path.isdir(req.folder_path):
        raise HTTPException(status_code=400, detail=f"La carpeta no existe: {req.folder_path}")
    tracks = scan_folder(req.folder_path)
    if not any("error" not in t for t in tracks):
        raise HTTPException(status_code=400, detail="No se encontraron pistas de audio soportadas en la carpeta.")
    return {"tracks": tracks}


@app.post("/api/plan")
def api_plan(req: PlanRequest):
    if not os.path.isdir(req.folder_path):
        raise HTTPException(status_code=400, detail=f"La carpeta no existe: {req.folder_path}")

    tracks = [t for t in scan_folder(req.folder_path) if "error" not in t]
    if len(tracks) < 2:
        raise HTTPException(status_code=400, detail="Se necesitan al menos 2 pistas analizables para planificar un set.")

    brief = build_set_brief(req.style_prompt)
    planned = plan_order(tracks, brief)
    by_path = _index_tracks_by_path(tracks)

    transitions = []
    for i in range(len(planned) - 1):
        outgoing = by_path[planned[i].file_path]
        incoming = by_path[planned[i + 1].file_path]
        relation = planned[i + 1].camelot_relation_to_prev or "misma_clave"
        pitch_adjust = planned[i + 1].pitch_adjust_percent
        plan = design_transition(outgoing, incoming, relation, pitch_adjust, brief)
        transitions.append({
            **dataclasses.asdict(plan),
            "actions": [dataclasses.asdict(a) | {"t_mmss": a.t_mmss} for a in plan.actions],
        })

    return {
        "brief": brief_to_dict(brief),
        "tracklist": [dataclasses.asdict(p) for p in planned],
        "tracks_by_path": by_path,
        "transitions": transitions,
    }


# --- Frontend estático ---
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
