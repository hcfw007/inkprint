import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from . import crawler_zhihu, personas, samples, voice_profile
from .db import init_db

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _purge_persona_files(persona_id: int) -> None:
    """Wipe on-disk artifacts (samples + profile versions) for a persona."""
    for platform_dir in (ROOT / "samples").glob("*"):
        target = platform_dir / str(persona_id)
        if target.exists():
            shutil.rmtree(target)
    profile_dir = ROOT / "profiles" / str(persona_id)
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    legacy_profile = ROOT / "profiles" / f"{persona_id}.md"
    legacy_profile.unlink(missing_ok=True)


def _purge_source_files(persona_id: int, platform: str) -> None:
    target = ROOT / "samples" / platform / str(persona_id)
    if target.exists():
        shutil.rmtree(target)


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
async def show_persona(
    request: Request,
    persona_id: int,
    added: int | None = None,
    updated: int | None = None,
    unchanged: int | None = None,
    total: int | None = None,
    profile: str | None = None,
) -> HTMLResponse:
    persona = personas.get(persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail="persona not found")
    return templates.TemplateResponse(
        request,
        "personas/show.html",
        {
            "persona": persona,
            "sources": personas.list_sources(persona_id),
            "bound_platforms": personas.bound_platforms(persona_id),
            "sync_delta": (
                {"added": added, "updated": updated, "unchanged": unchanged, "total": total}
                if added is not None
                else None
            ),
            "profile_generated": profile == "ok",
            "profile_md": voice_profile.read_existing(persona_id),
        },
    )


@app.post("/personas/{persona_id}/delete")
async def delete_persona(persona_id: int) -> RedirectResponse:
    if personas.get(persona_id) is None:
        raise HTTPException(status_code=404, detail="persona not found")
    personas.delete(persona_id)
    _purge_persona_files(persona_id)
    return RedirectResponse(url="/", status_code=303)


@app.post("/personas/{persona_id}/sources/{source_id}/delete")
async def unbind_source(persona_id: int, source_id: int) -> RedirectResponse:
    source = personas.get_source(source_id)
    if source is None or source["persona_id"] != persona_id:
        raise HTTPException(status_code=404, detail="source not found")
    personas.delete_source(source_id)
    _purge_source_files(persona_id, source["platform"])
    return RedirectResponse(url=f"/personas/{persona_id}", status_code=303)


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


@app.get("/personas/{persona_id}/samples", response_class=HTMLResponse)
async def list_samples(request: Request, persona_id: int) -> HTMLResponse:
    persona = personas.get(persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail="persona not found")
    return templates.TemplateResponse(
        request,
        "samples/list.html",
        {"persona": persona, "items": samples.list_for(persona_id)},
    )


@app.get("/personas/{persona_id}/samples/{idx}", response_class=HTMLResponse)
async def show_sample(request: Request, persona_id: int, idx: int) -> HTMLResponse:
    persona = personas.get(persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail="persona not found")
    item = samples.get_for(persona_id, idx)
    if item is None:
        raise HTTPException(status_code=404, detail="sample not found")
    return templates.TemplateResponse(
        request,
        "samples/show.html",
        {"persona": persona, "item": item, "idx": idx},
    )


@app.post("/personas/{persona_id}/profile")
async def generate_profile(persona_id: int) -> RedirectResponse:
    if personas.get(persona_id) is None:
        raise HTTPException(status_code=404, detail="persona not found")
    try:
        voice_profile.generate(persona_id)
    except voice_profile.ProfileError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return RedirectResponse(url=f"/personas/{persona_id}?profile=ok", status_code=303)


@app.get("/personas/{persona_id}/profile/versions", response_class=HTMLResponse)
async def list_profile_versions(request: Request, persona_id: int) -> HTMLResponse:
    persona = personas.get(persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail="persona not found")
    return templates.TemplateResponse(
        request,
        "profiles/versions.html",
        {"persona": persona, "versions": voice_profile.list_versions(persona_id)},
    )


@app.get("/personas/{persona_id}/profile/{name}", response_class=HTMLResponse)
async def show_profile_version(request: Request, persona_id: int, name: str) -> HTMLResponse:
    persona = personas.get(persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail="persona not found")
    content = voice_profile.read_version(persona_id, name)
    if content is None:
        raise HTTPException(status_code=404, detail="profile version not found")
    return templates.TemplateResponse(
        request,
        "profiles/show.html",
        {
            "persona": persona,
            "name": name,
            "created_at": voice_profile.format_ts(name),
            "content": content,
        },
    )


@app.post("/personas/{persona_id}/sources/{source_id}/sync")
async def sync_source(persona_id: int, source_id: int) -> RedirectResponse:
    source = personas.get_source(source_id)
    if source is None or source["persona_id"] != persona_id:
        raise HTTPException(status_code=404, detail="source not found")
    if source["platform"] != "zhihu":
        raise HTTPException(status_code=400, detail=f"unsupported platform: {source['platform']}")
    try:
        result = crawler_zhihu.sync(persona_id, source["identifier"])
    except crawler_zhihu.CrawlerError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    personas.mark_source_synced(source_id, result.total_count, str(result.sample_path))
    return RedirectResponse(
        url=(
            f"/personas/{persona_id}"
            f"?added={result.added_count}"
            f"&updated={result.updated_count}"
            f"&unchanged={result.unchanged_count}"
            f"&total={result.total_count}"
        ),
        status_code=303,
    )
