from pydantic import BaseModel, Field

class Base(BaseModel):
    pass

class Punto(Base):
    x: float = Field(..., example=1.0, title="Coordenada x")
    y: float = Field(..., example=2.3, title="Coordenada y")
    