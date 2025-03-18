# Proyecto de Optimización y Representación de Datos

Este proyecto contiene una serie de implementaciones relacionadas con optimización matemática, representación de matrices dispersas y visualización de funciones. Está dividido en varios puntos que abordan diferentes aspectos de la optimización y el análisis numérico.

## Estructura del Proyecto

El proyecto está organizado en los siguientes archivos y directorios:

- **`lab1_optimizacion.py`**: Archivo principal que contiene todas las implementaciones del proyecto.
- **`lab1.ipynb`** y **`lab1 copy.ipynb`**: Notebooks que pueden contener ejemplos o análisis adicionales.
- **`otra.py`**: Archivo auxiliar (contenido no especificado).
- **`backend/`**: Directorio que podría contener código adicional relacionado con el proyecto.
- **`README.md`**: Este archivo, que describe el proyecto.
- **`LICENSE`**: Archivo de licencia del proyecto.

## Contenido del Archivo Principal (`lab1_optimizacion.py`)

### Punto 1: Visualización de Regiones Factibles

Se implementa una función para graficar regiones factibles de una función de costo, coloreando las áreas según el valor del costo y marcando el punto máximo.

- **Funciones principales**:
  - `costo(x, y)`: Define la función de costo.
  - `plot_region_max(c, x_min, y_min)`: Genera un gráfico de la región factible y marca el máximo.

### Punto 2: Representación de Matrices Dispersas

Se implementa una clase personalizada para representar matrices dispersas en formato COO (Coordinate List) y se compara su rendimiento con la implementación de SciPy.

- **Clases y funciones principales**:
  - `SparseCOO`: Clase para representar matrices dispersas.
  - `generar_matriz_dispersa(n, m, dispersion, rango)`: Genera matrices dispersas aleatorias.
  - Comparación de tiempos entre la implementación personalizada y SciPy para creación y suma de matrices dispersas.

### Punto 3: Aproximación de Series de Taylor

Se implementa una función para calcular la aproximación de una función mediante la serie de Taylor y graficar la función original junto con su aproximación.

- **Funciones principales**:
  - `taylor_series(func, a, n, x)`: Calcula la serie de Taylor de una función.
  - `crear_grafico(funcion, a, n, x)`: Genera un gráfico comparativo entre la función original y su aproximación.

### Punto 4: Métodos de Optimización

Se implementan y comparan diferentes métodos de optimización para minimizar una función objetivo.

- **Métodos implementados**:
  - **Gradiente Descendente**: Implementación desde cero.
  - **Método de Newton**: Utiliza el gradiente y la Hessiana.
  - **BFGS**: Método cuasi-Newton utilizando SciPy.

- **Funciones principales**:
  - `gradiente_descendente(f, grad_f, x0, lr, tol, max_iter)`: Implementación del gradiente descendente.
  - `newton_method(f, grad_f, hess_f, x0, tol, max_iter)`: Implementación del método de Newton.
  - `bfgs_method(f, x0, tol, max_iter)`: Implementación del método BFGS utilizando SciPy.

- **Comparación**:
  Se grafican las curvas de convergencia de los tres métodos para analizar su rendimiento.

## Ejecución del Proyecto

1. **Requisitos**:
   - Python 3.7+
   - Bibliotecas: `numpy`, `matplotlib`, `scipy`, `sympy`

2. **Ejemplo de ejecución**:
   - Para visualizar la región factible:

     ```python
     plot_region_max(c=10)
     ```

   - Para generar una matriz dispersa y convertirla a formato COO:

     ```python
     dense = generar_matriz_dispersa(200, 200, dispersion=0.95, rango=(-1000, 1000))
     sparse = SparseCOO(dense)
     ```

   - Para aproximar una función con la serie de Taylor:

     ```python
     crear_grafico(funciones['sin'], a=0, n=5, x=sp.symbols('x'))
     ```

   - Para comparar métodos de optimización:

     ```python
     x0 = [0, 0]
     gradiente_descendente(f_opt, grad_f_opt, x0, lr=0.1)
     newton_method(f_opt, grad_f_opt, hess_f_opt, x0)
     bfgs_method(f_opt, x0)
     ```

## Resultados

- **Visualización**: Se generan gráficos que muestran regiones factibles, aproximaciones de Taylor y curvas de convergencia.
- **Matrices dispersas**: Comparación de tiempos entre la implementación personalizada y SciPy.
- **Optimización**: Comparación de métodos de optimización en términos de convergencia y rendimiento.

## Licencia

Este proyecto está licenciado bajo los términos especificados en el archivo [LICENSE](http://_vscodecontentref_/0).

## Contacto

Para preguntas o sugerencias, por favor contacta al autor del proyecto.
