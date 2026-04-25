from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from chesspoint72.web.api.engines import router as engines_router
from chesspoint72.web.api.hvse import router as hvse_router
from chesspoint72.web.api.eve import router as eve_router


_STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="Chesspoint72 Web", version="0.1.0")

    # API
    app.include_router(engines_router, prefix="/api")
    app.include_router(hvse_router, prefix="/api")
    app.include_router(eve_router, prefix="/api")

    # Static
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/eve", response_class=HTMLResponse)
    def eve() -> FileResponse:
        return FileResponse(_STATIC_DIR / "eve.html")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"ok": "true"}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = os.environ.get("CHESSPOINT72_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("CHESSPOINT72_WEB_PORT", "8000"))
    uvicorn.run("chesspoint72.web.server:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()

