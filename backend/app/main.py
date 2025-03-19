
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import punto1_router, punto2_router, punto3_router, punto4_router

app = FastAPI(
    title="API Optimización",
    description="API para el aprendizaje de optimización",
    version="0.1",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(punto1_router)
app.include_router(punto2_router)
app.include_router(punto3_router)
app.include_router(punto4_router)


@app.get("/")
async def root():
    return {"message": "Hello World from API Optimización"}
