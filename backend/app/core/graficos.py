from matplotlib import pyplot as plt
import numpy as np
import sympy as sp

from app.core.funciones import costo
from app.core.taylor import taylor_series

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
    #plt.show()

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
    plt.scatter(a, f_original(a), color='red', label=f'Punto de Expansión: ({a}, {f_original(a):4f})') # grafica el punto donde se centra la grafica
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Aproximación de Taylor de {label}')
    plt.legend()
    plt.savefig("Aproximación de Taylor.png", dpi=300, bbox_inches='tight')
    #plt.show()


def comparar_convergencia(f_grad, f_newton, f_bfgs):
    """
    Compara la convergencia de tres métodos de optimización.

    Parámetros:
    -----------
    f_grad : list
        Lista con los valores de la función objetivo en cada iteración para Gradiente Descendente.
    f_newton : list
        Lista con los valores de la función objetivo en cada iteración para el Método de Newton.
    f_bfgs : list
        Lista con los valores de la función objetivo en cada iteración para BFGS.

    """
    plt.figure(figsize=(10,6))
    plt.plot(f_grad, label=f'Gradiente Descendente ({len(f_grad)})', marker='o') # Graficar la evolución del valor de f en cada iteración para Gradiente Descendente
    plt.plot(f_newton, label=f'Newton ({len(f_newton)})', marker='s') # Graficar la evolución del valor de f en cada iteración para el Método de Newton
    plt.plot(f_bfgs, label=f'BFGS ({len(f_bfgs)})', marker='^') # Graficar la evolución del valor de f en cada iteración para BFGS
    plt.xlabel('Iteración')
    plt.ylabel('Valor de f(x, y)')
    plt.title('Comparación de Convergencia de Métodos de Optimización')
    plt.legend()
    plt.grid(True)
    plt.savefig("convergencia.png", dpi=300, bbox_inches='tight')
    #plt.show()