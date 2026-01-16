from pydantic import BaseModel
from typing import Optional

class BFResponse(BaseModel):
    body_fat_percentage: float
    muscle_mass_estimate: float
    confidence: float
    message: str
