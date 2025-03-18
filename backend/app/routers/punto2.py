from typing import Optional
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.matrices import generar_matriz_dispersa, comparar, visualizacion
from app.schemas.schemas import Metodo, Operacion

router = APIRouter(prefix="/punto2", tags=["Punto 2"])

@router.get("/", 
  summary="Comparar tiempo de suma de dos matrices dispersas",
  description="Compara el tiempo de suma de dos matrices dispersas con implementación propia.",
  response_description="Tiempo de suma de matrices dispersas (propia, scipy, densa).",
)
def get_test(tamaño: Optional[int] = 200, dispersion: Optional[float] = 0.95):
  if dispersion < 0 or dispersion > 1:
    return {"error": "La dispersión debe estar en el rango [0, 1]"}
  elif tamaño > 10000 and tamaño <= 0:
    return {"error": "El tamaño de la matriz no puede ser mayor a 10000"}
  
  dense1 = generar_matriz_dispersa(tamaño, tamaño, dispersion=dispersion)
  dense2 = generar_matriz_dispersa(tamaño, tamaño, dispersion=dispersion)

  results = comparar(dense1, dense2)

  return JSONResponse(content={
    "tiempos": results[0],
    "tamaño": tamaño,
    "dispersión": dispersion,
    "resultados": str(results[1][0])
  }, status_code=200)


@router.get("/tiempo", 
  summary="Obtener tiempo de ejecución de una operación",
  description="Obtiene el tiempo de ejecución de una operación en matrices dispersas. Metodo: 1 (COO), 2 (CSR), 3 (CSC). Operación: 1 (suma), 2 (multiplicación)",
  response_description="Tiempo de ejecución de la operación.",
)
def get_tiempo_ejecucion(metodo: Metodo, operacion: Operacion,escalar: Optional[int] = 1,  tamaño: Optional[int] = 200, dispersion: Optional[float] = 0.95):
  if dispersion < 0 or dispersion > 1:
    return {"error": "La dispersión debe estar en el rango [0, 1]"}
  elif tamaño > 10000 and tamaño <= 0:
    return {"error": "El tamaño de la matriz no puede ser mayor a 10000"}
  elif escalar == 0:
    return {"error": "El escalar debe ser diferente a 0"}

  results = visualizacion(metodo.value, operacion.value, escalar, tamaño, tamaño, dispersion)

  return JSONResponse(content={
    "tiempo": float(results[0]),
    "tamaño": tamaño,
    "dispersión": dispersion,
    "resultado": str(results[1])
  }, status_code=200)