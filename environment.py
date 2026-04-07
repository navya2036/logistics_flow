import json
from pathlib import Path
from models import LogisticsAction, LogisticsObservation, Order, IncomingRestock


DEFAULT_TASK_ID = "easy_fulfillment"

class WarehouseEnv:
    def __init__(self):
        scenarios_path = Path(__file__).with_name("task_scenarios.json")
        with scenarios_path.open("r", encoding="utf-8") as file:
            self.scenarios = json.load(file)
        self.reset(DEFAULT_TASK_ID)

    def reset(self, task_id: str = DEFAULT_TASK_ID):
        if task_id not in self.scenarios:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Available tasks: {', '.join(self.scenarios.keys())}"
            )

        scenario = self.scenarios[task_id]
        self.task_id = task_id
        self.max_days = int(scenario.get("max_days", 10))
        self.restock_lead_time = int(scenario.get("restock_lead_time", 0))
        self.current_day = 1
        self.inventory = dict(scenario.get("inventory", {}))
        self.orders = [
            Order(**order_data) for order_data in scenario.get("orders", [])
        ]
        self.incoming_restock = []
        self.log = (
            f"Warehouse initialized for {task_id}. "
            f"Restock lead time: {self.restock_lead_time} day(s)."
        )
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        return LogisticsObservation(
            inventory=self.inventory,
            pending_orders=self.orders,
            current_day=self.current_day,
            task_id=self.task_id,
            restock_lead_time=self.restock_lead_time,
            incoming_restock=[IncomingRestock(**evt) for evt in self.incoming_restock],
            log=self.log,
            done=self.done
        )

    def _apply_arrivals(self):
        arrivals = [evt for evt in self.incoming_restock if evt["arrival_day"] <= self.current_day]
        if not arrivals:
            return

        for evt in arrivals:
            self.inventory[evt["item_id"]] = self.inventory.get(evt["item_id"], 0) + evt["quantity"]

        self.incoming_restock = [evt for evt in self.incoming_restock if evt["arrival_day"] > self.current_day]
        arrivals_text = ", ".join([f"{evt['quantity']} {evt['item_id']}" for evt in arrivals])
        self.log += f" Restock arrived: {arrivals_text}."

    def step(self, action: LogisticsAction):
        reward = 0.0
        self.log = ""
        self._apply_arrivals()

        if action.action_type == "fulfill":
            order = next((o for o in self.orders if o.id == action.order_id), None)
            if order and self.inventory.get(order.item, 0) >= order.quantity:
                self.inventory[order.item] -= order.quantity
                self.orders.remove(order)
                reward += float(order.priority)
                self.log = f"Fulfilled {order.id}."
            else:
                reward -= 0.5
                self.log = "Fulfillment failed: insufficient stock or invalid ID."

        elif action.action_type == "restock":
            item_id = action.item_id or "unknown"
            quantity = max(1, int(action.quantity))
            arrival_day = self.current_day + self.restock_lead_time
            self.incoming_restock.append(
                {"item_id": item_id, "quantity": quantity, "arrival_day": arrival_day}
            )
            reward -= 0.1  # Cost of restocking
            self.log = (
                f"Restock placed for {quantity} {item_id}. "
                f"Expected arrival day: {arrival_day}."
            )
        else:
            self.log = "No-op action taken."

        # Advance Time
        self.current_day += 1

        # Check for expired orders
        expired = [o for o in self.orders if o.due_day < self.current_day]
        for o in expired:
            reward -= float(o.penalty)
            self.orders.remove(o)
            self.log += f" Order {o.id} expired (penalty {o.penalty})."

        if self.current_day > self.max_days or not self.orders:
            self.done = True

        return self._get_obs(), reward, self.done, {}