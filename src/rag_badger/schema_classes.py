# All the pydantic's DataModel Schema classes here
from pydantic import BaseModel
from typing import Literal

class IntScore(BaseModel):
    score:Literal[0, 1, 2]