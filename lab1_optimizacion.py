# -*- coding: utf-8 -*-
"""lab1 - Optimizacion.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1iJitrn2rO3cJfQ984BYjVw9cdeQmYpDz

# Punto 1
"""

import numpy as np
import matplotlib.pyplot as plt

# Definir la función de costo
def costo(x, y):
    return (x - 2)**2 + (y - 3)**2 #(x**2 - 2)**2 + (y - 3)**3

# Función para graficar la región factible con colores basados en el costo y marcar el máximo
def plot_region_max(c=10, x_min=0, y_min=0):
    # Crear un grid de puntos
    x = np.linspace(x_min, c, 400)
    y = np.linspace(y_min, c, 400)
    X, Y = np.meshgrid(x, y)

    Z = costo(X, Y) # Evaluar la función de costo en cada punto del grid

    # Definir la región factible: x>=x_min, y>=y_min y x+y<=c
    factible = (X >= x_min) & (Y >= y_min) & (X + Y <= c) # & (Y - 1 > X**2)

    # Para los puntos no factibles, asignamos NaN para que no se coloreen
    Z_feasible = np.where(factible, Z, np.nan)

    # Calcular el máximo en la región factible
    try:
        max_idx = np.nanargmax(Z_feasible)
    except ValueError:
        print("No se encontró un máximo en la región factible.")
        return
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z_feasible.flatten()
    x_max_val = X_flat[max_idx]
    y_max_val = Y_flat[max_idx]
    max_val = Z_flat[max_idx]

    # Graficar la región factible coloreada por el costo
    plt.figure(figsize=(6,6))
    im = plt.imshow(
        Z_feasible,
        extent=(x_min, c, y_min, c),
        origin='lower', cmap='Blues', interpolation='nearest'
    )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Región Factible (color ~ costo) con Máximo')

    # Agregar la barra de color
    cbar = plt.colorbar(im)
    cbar.set_label('Costo (mayor = más oscuro)')

    # Marcar el máximo en la gráfica
    plt.scatter(
        x_max_val, y_max_val,
        color='red', marker='.',
        s=150,
        label=f'Máximo: ({x_max_val:.2f}, {y_max_val:.2f})\nCosto = {max_val:.2f}'
    )
    plt.legend()
    plt.savefig("region_factible.png", dpi=300, bbox_inches='tight')
    plt.show()


# Función para evaluar la función de costo en un punto dado
def evaluar_punto(x, y):
    valor = costo(x, y)
    print(f"f({x}, {y}) = {valor}")

# Ejemplo de uso:
plot_region_max(c=10)
evaluar_punto(4, 5)

"""# Punto 2a"""

import numpy as np

class SparseCOO:
    def __init__(self, matrix: np.ndarray = []):
        self.shape = None
        self.rows = []
        self.cols = []
        self.data = []

        if len(matrix) <= 0:
            self.shape = (0, 0)
            return

        if not isinstance(matrix, np.ndarray):
            return "Error: matrix must be a numpy array"
        self.shape = matrix.shape

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.add_element(i, j, matrix[i, j])

    def add_element(self, i, j, value):
        if value != 0:
            self.rows.append(i)
            self.cols.append(j)
            self.data.append(value)

    def get_element(self, i):
        if i < self.shape[0]:
            return self.data[i]
        else:
            return None

    def to_dense(self):
        dense = np.zeros(self.shape)
        for r, c, v in zip(self.rows, self.cols, self.data):
            dense[r, c] = v
        return dense

    def __add__(self, other):
        # Suma de dos matrices sparse: se suma elemento a elemento.
        result = SparseCOO()
        result.shape = self.shape
        result.rows = self.rows.copy()
        result.cols = self.cols.copy()
        result.data = self.data.copy()

        # Agregar elementos de la segunda matriz
        for i, j, v in zip(other.rows, other.cols, other.data):
            data = result.get_element(i) # Buscar si el elemento ya existe en result
            if data is not None:
                result.add_element(i, j, data + v)
            else:
                result.add_element(i, j, v)
        return result

    def __str__(self):
        string = "Coords\tValues\n"

        for i in range(self.shape[0]):
            row = self.rows[i]
            col = self.cols[i]
            string += f"({row}, {col})\t{self.get_element(i)}\n"
        return string


import scipy.sparse as sp
import time

# Crear una matriz densa aleatoria y hacerla sparse
dense = generar_matriz_dispersa(200, 200, dispersion=0.95, rango=(-1000, 1000))

# Implementación propia: SparseCOO
t0 = time.time()
sparse_custom = SparseCOO(dense)
t1 = time.time()

# Usando SciPy (COO)
t2 = time.time()
sparse_scipy = sp.coo_matrix(dense)
t3 = time.time()

print("Tiempo de creación - Implementación propia:", t1 - t0)
print("Tiempo de creación - SciPy COO:", t3 - t2)

# Operación: suma de dos matrices sparse
# Crear otra matriz densa y convertirla a sparse
dense2 = generar_matriz_dispersa(200, 200, dispersion=0.95, rango=(-1000, 1000))

# Con implementación propia
sparse_custom2 = SparseCOO(dense2)
t4 = time.time()
sparse_sum_custom = sparse_custom + sparse_custom2
t5 = time.time()

# Con SciPy
sparse_scipy2 = sp.coo_matrix(dense2)
t6 = time.time()
sparse_sum_scipy = sparse_scipy + sparse_scipy2
t7 = time.time()

# Con matrix densa
t8 = time.time()
dense_sum = dense.tolist() + dense2.tolist()
t9 = time.time()

print("Tiempo de suma - Implementación propia:", t5 - t4)
print("Tiempo de suma - SciPy COO:", t7 - t6)
print("Tiempo de suma - NumPy:", t9 - t8)

"""# Punto 2b

Métodos de representación de matrices sparse
1. COO (SciPy)
2. CSR (SciPy)
3. CSC (SciPy)

Operaciones
1. Suma
2. Multiplicación por escalar
3. Multiplicacion entre matrices
"""

def visualizacion(metodo, operacion, escalar = 1, n = 200, m = 200, dispersion = 0.95, rango = (-1000, 1000)):
    dense1 = generar_matriz_dispersa(n, m, dispersion, rango)
    dense2 = generar_matriz_dispersa(n, m, dispersion, rango)

    # Crear matrices sparse según el método seleccionado
    if metodo == 1:
        sparse1 = sp.coo_matrix(dense1)
        sparse2 = sp.coo_matrix(dense2)
    elif metodo == 2:
        sparse1 = sp.csr_matrix(dense1)
        sparse2 = sp.csr_matrix(dense2)
    elif metodo == 3:
        sparse1 = sp.csc_matrix(dense1)
        sparse2 = sp.csc_matrix(dense2)

    # Realizar la operación y medir el tiempo
    inicio = time.time()
    if operacion == 1:
        resultado = sparse1 + sparse2
    elif operacion == 2:
        resultado = sparse1 * escalar
    elif operacion == 3:
        resultado = sparse1.dot(sparse2)
    fin = time.time()

    tiempo_ejecucion = fin - inicio

    return tiempo_ejecucion, resultado


visualizacion(1, 3)

"""# Punto 3"""

def taylor_series(func, a, n, x):
    """
    Calcula la serie de Taylor de una función en torno a un punto dado.

    Parámetros:
    -----------
    func : sympy.Expr
        Función simbólica a aproximar con la serie de Taylor.
    a : float o sympy.Symbol
        Punto alrededor del cual se expande la serie de Taylor.
    n : int
        Número de términos en la expansión de Taylor.
    x : sympy.Symbol
        Variable simbólica con respecto a la cual se deriva.

    Retorna:
    --------
    function
        Función numérica generada a partir de la serie de Taylor,
        que puede evaluarse con valores numéricos usando NumPy.

    """

    serie = 0
    for i in range(n):
        derivada = sp.diff(func, x, i)  # Calcula la i-ésima derivada de func respecto a x
        serie += derivada.subs(x, a) / sp.factorial(i) * (x - a)**i  # Agrega el término a la serie

    return sp.lambdify(x, serie, 'numpy')  # Convierte la serie simbólica en una función numérica

def crear_grafico(funcion, a:float, n: int, x):
    """
    Crea un gráfico mostrando la función original y su aproximación con la serie de Taylor.

    Parámetros:
    -----------
    funcion : sympy.Expr
        Función simbólica a aproximar con la serie de Taylor.

    a : float o sympy.Symbol
        Punto alrededor del cual se expande la serie de Taylor.
    n : int
        Número de términos en la expansión de Taylor.
    x : sympy.Symbol
        Variable simbólica con respecto a la cual se deriva.

    """
    # Función original y su aproximación
    f_original = sp.lambdify(x, funcion, 'numpy')
    f_taylor = taylor_series(funcion, a, n, x)

    # Graficar en un intervalo
    label = rf"${sp.latex(funcion)}$"
    x_vals = np.linspace(a-10, a+10, 400)
    plt.figure(figsize=(10,6))
    plt.plot(x_vals, f_original(x_vals),  label=label)
    plt.plot(x_vals, f_taylor(x_vals), label=f'Taylor ({n} términos)', linestyle='--')
    plt.scatter(a, f_original(a), color='red', label=f'Punto de Expansión: ({a}, {f_original(a)})') # grafica el punto donde se centra la grafica
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Aproximación de Taylor de {label}')
    plt.legend()
    plt.savefig("Aproximación de Taylor.png", dpi=300, bbox_inches='tight')
    plt.show()

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Definir funciones disponibles
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

# Usuario elige función, punto de expansión y número de términos
nombre_func = input(f"ingresa {[fun for fun in funciones.keys()]}: ")   # por ejemplo, 'sin'
a = float(input("ingrese centro: "))                                    # punto de expansión
n = int(input("ingrese la cantidad de terminos: "))                     # número de términos

crear_grafico(funciones[nombre_func], a, n, x)

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x = sp.symbols('x')

expresion = input("ingrese la funcion: ")
funcion = sp.sympify(expresion, locals={'e': sp.E, 'pi': sp.pi, 'ln': sp.log})
a = float(input("ingrese centro: "))                                # punto de expansión
n = int(input("ingrese la cantidad de terminos: "))                 # número de términos

crear_grafico(funcion, a, n, x)

"""# Punto 4

## Metodo 1
"""

def gradiente_descendente(f, grad_f, x0, lr=0.1, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    iter_count = 0
    while iter_count < max_iter:
        grad = np.array(grad_f(x))
        if np.linalg.norm(grad) < tol:
            break
        x = x - lr * grad
        iter_count += 1
    return x, iter_count

# Función y gradiente
def f_opt(x):
    return (x[0]-2)**2 + (x[1]-3)**2

def grad_f_opt(x):
    return [2*(x[0]-2), 2*(x[1]-3)]

# Ejecución con punto inicial y tasa de aprendizaje modificable
x0 = [0, 0]
sol, iteraciones = gradiente_descendente(f_opt, grad_f_opt, x0, lr=0.1)
print("Gradiente Descendente -> Solución:", sol, "Iteraciones:", iteraciones)

def newton_method(f, grad_f, hess_f, x0, tol=1e-6, max_iter=100):
    x = np.array(x0, dtype=float)
    iter_count = 0
    while iter_count < max_iter:
        grad = np.array(grad_f(x))
        if np.linalg.norm(grad) < tol:
            break
        H = np.array(hess_f(x))
        # Resolver H * delta = grad
        delta = np.linalg.solve(H, grad)
        x = x - delta
        iter_count += 1
    return x, iter_count

def hess_f_opt(x):
    return [[2, 0],
            [0, 2]]

sol_newton, iter_newton = newton_method(f_opt, grad_f_opt, hess_f_opt, x0)
print("Método de Newton -> Solución:", sol_newton, "Iteraciones:", iter_newton)

from scipy.optimize import minimize

res = minimize(f_opt, x0, method='BFGS', options={'disp': True})
print("BFGS -> Solución:", res.x, "Iteraciones:", res.nit)

"""## Metodo 2"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

"""### Funcion Objetivo"""

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

"""### Main"""

x0 = np.array([-2.0, 0.0]) # Definir el punto inicial común para todos los métodos

x_grad, iter_grad, f_grad = gradiente_descendente(f_opt, grad_f_opt, x0, lr=0.1) # Método del Gradiente Descendente
x_newton, iter_newton, f_newton = newton_method(f_opt, grad_f_opt, hess_f_opt, x0) # Método de Newton
res_bfgs, iter_bfgs, f_bfgs = bfgs_method(f_opt, x0) # Método BFGS (usando SciPy)

# Imprimir las soluciones y número de iteraciones para cada método
print("Gradiente Descendente -> Solución:", x_grad, "Iteraciones:", len(iter_grad))
print("Newton -> Solución:", x_newton, "Iteraciones:", len(iter_newton))
print("BFGS -> Solución:", res_bfgs.x, "Iteraciones:", len(iter_bfgs))

# =============================================================================
# Gráfica de comparación de la convergencia de los tres métodos
# =============================================================================

plt.figure(figsize=(10,6))
plt.plot(f_grad, label='Gradiente Descendente', marker='o') # Graficar la evolución del valor de f en cada iteración para Gradiente Descendente
plt.plot(f_newton, label='Newton', marker='s') # Graficar la evolución del valor de f en cada iteración para el Método de Newton
plt.plot(f_bfgs, label='BFGS', marker='^') # Graficar la evolución del valor de f en cada iteración para BFGS
plt.xlabel('Iteración')
plt.ylabel('Valor de f(x, y)')
plt.title('Comparación de Convergencia de Métodos de Optimización')
plt.legend()
plt.grid(True)
plt.show()