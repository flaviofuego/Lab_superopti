from matplotlib import pyplot as plt
import numpy as np

from app.core.funciones import costo

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
