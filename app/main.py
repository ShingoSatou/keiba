from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import ui

app = FastAPI(title="競馬予測システム API")

# フロントエンド開発時の CORS 許可（Vite デフォルトポート）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ui.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
