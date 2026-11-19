from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
from starlette.middleware.base import BaseHTTPMiddleware

from app.db import SessionLocal, engine, maybe_prepare_schema
from app import models  # noqa: F401
from app.middleware_auth import AuthGateMiddleware
from app.models import AnalyticsEvent
from app.routers import admin, auth, expert, most, qa, registry, workspace
from app.services.corpus import ensure_corpus_layout

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    maybe_prepare_schema(engine)
    ensure_corpus_layout()
    yield


app = FastAPI(
    title="Эксперт МОСТ (УКЗ-15)",
    description="RAG-консультант по корпоративной базе знаний",
    version="0.1.0",
    lifespan=app_lifespan,
)


class AnalyticsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/api/") and not path.endswith("/health"):
            db = SessionLocal()
            try:
                db.add(
                    AnalyticsEvent(
                        event_type="api_call",
                        payload={"path": path, "status": response.status_code},
                    )
                )
                db.commit()
            except Exception:
                db.rollback()
            finally:
                db.close()
        return response


app.add_middleware(AnalyticsMiddleware)
app.add_middleware(AuthGateMiddleware)
app.include_router(auth.router)
app.include_router(most.ui_router)
app.include_router(most.router)
app.include_router(expert.router)
app.include_router(qa.router)
app.include_router(registry.router)
app.include_router(admin.ui_router)
app.include_router(admin.router)
app.include_router(workspace.router)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/ui", response_class=HTMLResponse)
def chat_ui(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/api/v1/health")
def health_check():
    db_name = None
    try:
        with engine.connect() as conn:
            db_name = conn.execute(text("SELECT current_database()")).scalar()
    except Exception:
        db_name = None
    return {"status": "ok", "project": "most-expert", "version": "0.1.0", "database": db_name}
