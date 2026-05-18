from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from . import personas
from .db import init_db

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="inkprint", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "personas/list.html",
        {"personas": personas.list_all()},
    )


@app.get("/personas/new", response_class=HTMLResponse)
async def new_persona_form(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "personas/new.html", {})


@app.post("/personas")
async def create_persona(
    name: str = Form(...),
    description: str = Form(""),
) -> RedirectResponse:
    if not name.strip():
        raise HTTPException(status_code=400, detail="name is required")
    persona_id = personas.create(name, description)
    return RedirectResponse(url=f"/personas/{persona_id}", status_code=303)


@app.get("/personas/{persona_id}", response_class=HTMLResponse)
async def show_persona(request: Request, persona_id: int) -> HTMLResponse:
    persona = personas.get(persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail="persona not found")
    return templates.TemplateResponse(
        request,
        "personas/show.html",
        {"persona": persona, "sources": personas.list_sources(persona_id)},
    )


@app.post("/personas/{persona_id}/sources")
async def add_source(
    persona_id: int,
    platform: str = Form(...),
    identifier: str = Form(...),
) -> RedirectResponse:
    if personas.get(persona_id) is None:
        raise HTTPException(status_code=404, detail="persona not found")
    result = personas.add_source(persona_id, platform, identifier)
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return RedirectResponse(url=f"/personas/{persona_id}", status_code=303)
