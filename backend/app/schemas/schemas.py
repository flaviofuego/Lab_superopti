from pydantic import BaseModel
from enum import Enum, IntEnum
class Base(BaseModel):
    pass

class Funcion(Enum):
    exp = 'exp'
    sin = 'sin'
    cos = 'cos'
    ln = 'ln'
    atan = 'atan'
    sqrt = 'sqrt'
    x2 = 'x**2'
    x3 = 'x**3'
    x4 = 'x**4'
    x5 = 'x**5'
    inv1x2 = '1/(1+x**2)'

class Metodo(IntEnum):
    COO = 1
    CSR = 2
    CSC = 3

class Operacion(IntEnum):
    suma = 1
    multiplicacion = 2