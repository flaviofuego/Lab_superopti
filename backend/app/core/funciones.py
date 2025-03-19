import numpy as np

def costo(x: float, y: float) -> float:
    return (x - 2)**2 + (y - 3)**2 #(x**2 - 2)**2 + (y - 3)**3


def f_opt(x):
    """
    Función objetivo: f(x,y) = (x-2)^2 + (y-3)^2
    Parámetros:
      - x: array con dos componentes [x, y]
    Retorna:
      - Valor de la función evaluada en x
    """
    return (x[0] - 2)**2 + (x[1] - 3)**2

def grad_f_opt(x):
    """
    Gradiente de la función f_opt.
    Calcula ∇f(x,y) = [2*(x-2), 2*(y-3)]
    Parámetros:
      - x: array [x, y]
    Retorna:
      - Array con las derivadas parciales respecto a x e y
    """
    return np.array([2*(x[0] - 2), 2*(x[1] - 3)])

def hess_f_opt(x):
    """
    Hessiana de la función f_opt.
    Para f(x,y) = (x-2)^2+(y-3)^2, la Hessiana es constante:
      H = [[2, 0],
           [0, 2]]
    Parámetros:
      - x: array [x, y] (no se utiliza en este caso, ya que es constante)
    Retorna:
      - Matriz Hessiana 2x2
    """
    return np.array([[2, 0],
                     [0, 2]])
