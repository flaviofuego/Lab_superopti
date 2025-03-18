import numpy as np


def generar_matriz_dispersa(n, m, dispersion=0.95, rango=(-1000, 1000)) -> np.ndarray:
    """
    Genera una matriz de tamaño (n, m) con un porcentaje de elementos en cero.

    Parámetros:
    - n: Número de filas.
    - m: Número de columnas.
    - dispersion: Proporción de elementos en cero (por defecto 95%).
    - rango: Tupla con el rango (mín, máx) de los valores no nulos.

    Retorna:
    - Matriz numpy de tamaño (n, m).
    """
    matriz = np.zeros((n, m))  # Matriz inicial con solo ceros
    num_no_ceros = int((1 - dispersion) * n * m) # Número de elementos no nulos a insertar

    # Generar índices aleatorios donde se ubicarán los valores no nulos
    indices = np.random.choice(n * m, num_no_ceros, replace=False)

    # Generar valores aleatorios para los elementos no nulos
    valores = np.random.randint(rango[0], rango[1], num_no_ceros)

    np.put(matriz, indices, valores) # Insertar los valores en la matriz

    return matriz
