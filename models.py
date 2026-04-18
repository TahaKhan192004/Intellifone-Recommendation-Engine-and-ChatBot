from pydantic import BaseModel
from typing import Optional
import typing
from typing import List

class ChatRequest(BaseModel):
    user_id: str
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatHistoryResponse(BaseModel):
    conversation_id: str
    messages: List[ChatMessage]

class UsedMobile(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None
    ram: Optional[str] = None
    storage: Optional[str] = None

    condition: Optional[int] = None               # subjective from OLX
    condition_score: Optional[float] = None       # AI score (not used here)

    pta_approved: Optional[bool] = None        

    is_panel_changed: Optional[bool] = None
    screen_crack: Optional[bool] = None
    panel_dot: Optional[bool] = None
    panel_line: Optional[bool] = None
    panel_shade: Optional[bool] = None
    camera_lens_ok: Optional[bool] = None
    fingerprint_ok: Optional[bool] = None

    with_box: Optional[bool] = None
    with_charger: Optional[bool] = None

    price: Optional[int] = None
    city: Optional[str] = None

    listing_source: Optional[str] = None     
    images: Optional[list[str]] = None
    post_date: Optional[str] = None