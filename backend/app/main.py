from fastapi import FastAPI

app = FastAPI(
    title="API Optimización",
    description="API para el aprendizaje de optimización",
    version="0.1",
)

@app.get("/")
async def root():
    return {"message": "Hello World"}