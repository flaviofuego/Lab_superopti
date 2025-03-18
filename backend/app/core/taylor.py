import sympy as sp

def taylor_series(func, a, n, x) -> 'function':
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
