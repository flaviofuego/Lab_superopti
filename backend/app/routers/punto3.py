from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
import sympy as sp

from app.core.graficos import crear_grafico
from app.schemas.schemas import Funcion

router = APIRouter(prefix="/punto3", tags=["Punto 3"])

@router.get("/taylor_dinamico", 
  description="Obtiene la aproximación de la serie de Taylor de una función cualquiera en un punto dado.",
  response_description="Imagen PNG con el gráfico de la aproximación de Taylor.",
)
async def get_polinomio_taylor_dinamico(expresion: str, a: float, n: int = 2):
    x = sp.symbols('x')
    funcion = None
    try:
        funcion = sp.sympify(expresion, locals={'e': sp.E, 'pi': sp.pi, 'ln': sp.log})
    except:
        return {"error": "La expresión no es válida"}

    if n <= 1:
        return JSONResponse(status_code=400, content={"error": "El número de términos debe ser mayor a 1"})
    
    crear_grafico(funcion, a, n, x)

    return FileResponse("Aproximación de Taylor.png", media_type="image/png", filename="Aproximación de Taylor.png", content_disposition_type="inline")


@router.get("/taylor_preestablecido", 
  description="Obtiene la aproximación de la serie de Taylor de una serie de funciones preestablecidas en un punto dado.",
  response_description="Imagen PNG con el gráfico de la aproximación de Taylor.",
)
async def get_polinomio_taylor_preestablecido(expresion: Funcion, a: float, n: int = 2):
    x = sp.symbols('x')

    funciones = {
        'exp': sp.exp(x),
        'sin': sp.sin(x),
        'cos': sp.cos(x),
        'ln': sp.log(1+x),   # dominio: x > -1
        'atan': sp.atan(x),
        'sqrt': sp.sqrt(x),
        'x**2': sp.Pow(x, 2),
        'x**3': sp.Pow(x, 3),
        'x**4': sp.Pow(x, 4),
        'x**5': sp.Pow(x, 5),
        '1/(1+x**2)': sp.Pow(1+x**2, -1),
    }

    funcion = None
    try:
        funcion = funciones[expresion.value]
    except:
        return {"error": "La función no es válida"}

    if n <= 1:
        return JSONResponse(status_code=400, content={"error": "El número de términos debe ser mayor a 1"})
    
    crear_grafico(funcion, a, n, x)

    return FileResponse("Aproximación de Taylor.png", media_type="image/png", filename="Aproximación de Taylor.png", content_disposition_type="inline")

@router.get("/funciones")
async def get_funciones():
    return list(Funcion.__members__.values())