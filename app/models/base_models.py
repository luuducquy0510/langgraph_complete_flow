import uuid
from datetime import datetime
from typing import TypeVar, Generic, Literal

from pydantic import BaseModel, Field, field_validator


DataT = TypeVar("DataT")


class BaseResponse(BaseModel):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    created_by: uuid.UUID
    updated_by: uuid.UUID

