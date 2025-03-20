from typing import Optional
from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.core.graficos import comparar_convergencia
from app.core.metodos_minimizacion import gradiente_descendente, newton_method, bfgs_method
from app.core.funciones import f_opt, grad_f_opt, hess_f_opt

router = APIRouter(prefix="/punto4", tags=["Punto 4"])

@router.get("/", 
  response_description="Compara la convergencia de los métodos de Gradiente Descendente, Newton y BFGS para la función f(x, y) = (x - 2)^2 + (y - 3)^2"
)
def get_comvergencia(
    x0: float = 0, 
    y0: float = 0, 
    tol: Optional[float] = 1e-6, 
    max_iter: Optional[int] = 1000,
    tasa_aprendizaje: Optional[float] = 0.1
  ):

    if max_iter < 1:
        return {"error": "El número máximo de iteraciones debe ser mayor o igual a 1."}
    if tol <= 0:
        return {"error": "La tolerancia debe ser un número positivo."}
    if tasa_aprendizaje <= 0:
        return {"error": "La tasa de aprendizaje debe ser un número positivo."}

    f_grad = gradiente_descendente(f_opt, grad_f_opt, [x0, y0], tol=tol, max_iter=max_iter, lr=tasa_aprendizaje)[1]
    f_newton = newton_method(f_opt, grad_f_opt, hess_f_opt, [x0, y0], tol=tol, max_iter=max_iter)[1]
    f_bfgs = bfgs_method(f_opt,  [x0, y0], tol=tol, max_iter=max_iter)[1]
  
    comparar_convergencia(f_grad, f_newton, f_bfgs)
    return FileResponse("convergencia.png", media_type="image/png", filename="convergencia.png", content_disposition_type="inline")