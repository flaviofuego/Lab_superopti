import time
import numpy as np
import scipy.sparse as sp


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
        string = f"Coords\tValues ({len(self.data)})\n"

        for i in range(self.shape[0]):
            row = self.rows[i]
            col = self.cols[i]
            string += f"({row}, {col})\t{self.get_element(i)}\n"
        return string


def comparar(matris_densa1: np.ndarray, matris_densa2: np.ndarray) -> tuple:
    """
    Compara el tiempo de suma de dos matrices densas, dispersas y dispersas con implementación propia.

    Parámetros:
    - matris_densa1: Matriz densa 1.
    - matris_densa2: Matriz densa 2.

    Retorna:
    - Tupla con los tiempos de ejecución y los resultados de las sumas.
    """

    # Con implementación propia
    sparse_custom = SparseCOO(matris_densa1)
    sparse_custom2 = SparseCOO(matris_densa2)

    t4 = time.time()
    sparse_sum_custom = sparse_custom + sparse_custom2
    t5 = time.time()

    # Con SciPy
    sparse_scipy = sp.coo_matrix(matris_densa1)
    sparse_scipy2 = sp.coo_matrix(matris_densa2)
    t6 = time.time()
    sparse_sum_scipy = sparse_scipy + sparse_scipy2
    t7 = time.time()

    # Con matrix densa
    t8 = time.time()
    dense_sum = matris_densa1.tolist() + matris_densa2.tolist()
    t9 = time.time()

    return (t5 - t4, t7 - t6, t9 - t8), (sparse_sum_custom, sparse_sum_scipy, dense_sum)


def visualizacion(metodo, operacion, escalar = 1, n = 200, m = 200, dispersion = 0.95, rango = (-1000, 1000)):
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