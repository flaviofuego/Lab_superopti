from typing import Optional
from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.core.graficos import plot_region_max
from app.core.funciones import costo

router = APIRouter(prefix="/punto1", tags=["Punto 1"])

MIN_X = 0
MIN_Y = 0
C = 10

@router.get("/funcion", response_model=str)
def get_funcion_costo():
    return f"f(x, y) = (x - 2)^2 + (y - 3)^2"

@router.get("/resticciones")
def get_restricciones():
    global MIN_X, MIN_Y, C
    return [f"x >= {MIN_X}", f"y >= {MIN_Y}", f"x + y <= {C}"]
    
@router.get("/costo", response_model=float)
def get_costo(x: float, y: float):
    return costo(x, y)

@router.get("/region_factible")
async def get_region_factible(mx: Optional[float] = None, my: Optional[float] = None, c: Optional[float] = None):
    global MIN_X, MIN_Y, C
    if mx is None:
        mx = MIN_X
    if my is None:
        my = MIN_Y
    if c is None:
        c = C
    plot_region_max(c, mx, my)
    return FileResponse("region_factible.png", media_type="image/png", filename="region_factible.png", content_disposition_type="inline")
