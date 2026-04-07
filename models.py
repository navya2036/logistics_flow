from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional

class LogisticsAction(BaseModel):
    action_type: Literal["fulfill", "restock", "noop"] = Field(..., description="The action to take")
    order_id: Optional[str] = Field(None, description="Required for 'fulfill'")
    item_id: Optional[str] = Field(None, description="Required for 'restock'")
    quantity: int = Field(1, description="Amount to restock or fulfill")

class Order(BaseModel):
    id: str
    item: str
    quantity: int
    due_day: int
    priority: int  # 1 (Low) to 3 (High)
    penalty: float = 2.0


class IncomingRestock(BaseModel):
    item_id: str
    quantity: int
    arrival_day: int

class LogisticsObservation(BaseModel):
    inventory: Dict[str, int]
    pending_orders: List[Order]
    current_day: int
    task_id: str
    restock_lead_time: int
    incoming_restock: List[IncomingRestock]
    log: str
    done: bool