import numpy as np
from scipy.optimize import minimize

"""### Gradiente Descendente"""

def gradiente_descendente(f, grad_f, x0, lr=0.1, tol=1e-6, max_iter=1000):
    """
    Optimización mediante Gradiente Descendente.

    Parámetros:
      - f: función objetivo
      - grad_f: función que calcula el gradiente de f
      - x0: punto inicial (array)
      - lr: tasa de aprendizaje
      - tol: tolerancia para la norma del gradiente
      - max_iter: número máximo de iteraciones

    Retorna:
      - x: solución final
      - iterates: lista de puntos (soluciones) en cada iteración
      - f_values: lista de valores de f en cada iteración
    """
    x = np.array(x0, dtype=float)   # Inicializar la solución
    iterates = [x.copy()]           # Almacenar el punto inicial
    f_values = [f(x)]               # Almacenar el valor inicial de la función

    for i in range(max_iter):
        grad = grad_f(x)            # Calcular el gradiente en el punto x
        # Si la norma del gradiente es menor que la tolerancia, detener el algoritmo
        if np.linalg.norm(grad) < tol:
            break
        x = x - lr * grad           # Actualizar x usando la regla del gradiente descendente
        iterates.append(x.copy())   # Guardar el nuevo punto
        f_values.append(f(x))       # Guardar el nuevo valor de la función

    return x, iterates, f_values

"""### Método de Newton"""

def newton_method(f, grad_f, hess_f, x0, tol=1e-6, max_iter=100):
    """
    Optimización mediante el Método de Newton.

    Parámetros:
      - f: función objetivo
      - grad_f: función que calcula el gradiente de f
      - hess_f: función que calcula la Hessiana de f
      - x0: punto inicial (array)
      - tol: tolerancia para la norma del gradiente
      - max_iter: número máximo de iteraciones

    Retorna:
      - x: solución final
      - iterates: lista de puntos en cada iteración
      - f_values: lista de valores de f en cada iteración
    """
    x = np.array(x0, dtype=float)
    iterates = [x.copy()]           # Almacenar el punto inicial
    f_values = [f(x)]               # Almacenar el valor inicial de f

    for i in range(max_iter):
        grad = grad_f(x)            # Calcular el gradiente
        # Si la norma del gradiente es menor que la tolerancia, se alcanza la convergencia
        if np.linalg.norm(grad) < tol:
            break
        H = hess_f(x)               # Calcular la Hessiana
        delta = np.linalg.solve(H, grad)  # Resolver el sistema H * delta = gradiente
        x = x - delta               # Actualizar x
        iterates.append(x.copy())   # Guardar el nuevo punto
        f_values.append(f(x))       # Guardar el nuevo valor de f

    return x, iterates, f_values

"""### BFGS utilizando SciPy"""

def bfgs_method(f, x0, tol=1e-6, max_iter=1000):
    """
    Optimización mediante el método BFGS (cuasi-Newton) usando la función minimize de SciPy.

    Parámetros:
      - f: función objetivo
      - x0: punto inicial (array)
      - tol: tolerancia para la convergencia
      - max_iter: número máximo de iteraciones

    Retorna:
      - res: objeto resultado de SciPy (contiene la solución y otros datos)
      - iterates: lista de puntos generados en cada iteración (usando callback)
      - f_values: lista de valores de f en cada iteración
    """
    iterates = []   # Lista para almacenar los puntos en cada iteración
    f_values = []    # Lista para almacenar los valores de f correspondientes

    def callback(xk):
        """
        Función callback que se llama en cada iteración de BFGS.
        Almacena el punto actual y el valor de f en ese punto.
        """
        iterates.append(np.copy(xk))
        f_values.append(f(xk))

    # Ejecutar la optimización con método BFGS
    res = minimize(f, x0, method='BFGS', tol=tol, options={'maxiter': max_iter, 'disp': False}, callback=callback)

    # Si el callback no se llamó (por ejemplo, convergió en 0 iteraciones), se agrega el punto inicial.
    if len(iterates) == 0:
        iterates.append(np.array(x0))
        f_values.append(f(x0))

    return res, iterates, f_values
