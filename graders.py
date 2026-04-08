import json
from typing import Dict, List, Tuple

from environment import WarehouseEnv
from models import LogisticsAction


KNOWN_TASKS = ["easy_fulfillment", "medium_restock", "hard_peak_season"]


def to_strict_unit_interval(value: float) -> float:
    # Hackathon requirement: each task score must be strictly between 0 and 1.
    return min(0.99, max(0.01, float(value)))


def choose_rule_based_action(obs: Dict) -> Dict:
    inventory = obs.get("inventory", {})
    orders = obs.get("pending_orders", [])
    incoming = obs.get("incoming_restock", [])

    if not orders:
        return {"action_type": "noop"}

    incoming_by_item: Dict[str, int] = {}
    for evt in incoming:
        item = evt.get("item_id")
        qty = int(evt.get("quantity", 0))
        incoming_by_item[item] = incoming_by_item.get(item, 0) + qty

    def order_rank(order: Dict) -> Tuple:
        return (
            order.get("due_day", 999),
            -order.get("priority", 0),
            -float(order.get("penalty", 0.0)),
        )

    sorted_orders = sorted(orders, key=order_rank)

    for order in sorted_orders:
        item = order["item"]
        need = int(order["quantity"])
        have = int(inventory.get(item, 0))
        if have >= need:
            return {"action_type": "fulfill", "order_id": order["id"]}

    target = sorted_orders[0]
    item = target["item"]
    need = int(target["quantity"])
    have = int(inventory.get(item, 0))
    incoming_qty = int(incoming_by_item.get(item, 0))
    shortage = need - have

    if shortage <= 0 or incoming_qty >= shortage:
        return {"action_type": "noop"}

    return {
        "action_type": "restock",
        "item_id": item,
        "quantity": shortage - incoming_qty,
    }


def run_episode(task_id: str, max_steps: int = 12) -> Dict:
    env = WarehouseEnv()
    obs_model = env.reset(task_id)
    obs = obs_model.model_dump()

    done = bool(obs.get("done", False))
    steps = 0
    total_reward = 0.0
    actions: List[Dict] = []

    while not done and steps < max_steps:
        action_dict = choose_rule_based_action(obs)
        actions.append(action_dict)

        action = LogisticsAction(**action_dict)
        next_obs_model, reward, done, _ = env.step(action)

        obs = next_obs_model.model_dump()
        total_reward += float(reward)
        steps += 1

    return {
        "task_id": task_id,
        "steps": steps,
        "total_reward": total_reward,
        "actions": actions,
        "remaining_orders": len(obs.get("pending_orders", [])),
        "done": done,
    }


def grade_easy(result: Dict) -> float:
    if not result["done"]:
        return 0.01  # near-zero: episode did not complete
    if result["remaining_orders"] == 0 and result["total_reward"] >= 4.0:
        return 0.99  # near-perfect: all orders fulfilled with full reward
    if result["remaining_orders"] == 0:
        return 0.8
    return 0.2


def grade_medium(result: Dict) -> float:
    actions = result["actions"]
    first_fulfill_idx = next((i for i, a in enumerate(actions) if a.get("action_type") == "fulfill"), None)
    first_restock_idx = next((i for i, a in enumerate(actions) if a.get("action_type") == "restock"), None)

    restock_before_fulfill = first_restock_idx is not None and (
        first_fulfill_idx is None or first_restock_idx < first_fulfill_idx
    )

    if result["remaining_orders"] == 0 and restock_before_fulfill:
        return 0.99  # near-perfect: correct restock-first strategy and all orders done
    if result["remaining_orders"] == 0:
        return 0.6
    if restock_before_fulfill:
        return 0.4
    return 0.01  # near-zero: wrong strategy and orders unfulfilled


def grade_hard(result: Dict) -> float:
    fulfilled_ids = {a.get("order_id") for a in result["actions"] if a.get("action_type") == "fulfill"}
    high_priority_done = "ORD201" in fulfilled_ids

    if result["remaining_orders"] == 0 and result["total_reward"] >= 3.0:
        return 0.99  # near-perfect: all orders done with sufficient reward
    if high_priority_done and result["total_reward"] >= 2.0:
        return 0.8
    if high_priority_done:
        return 0.5
    return 0.01  # near-zero: high-priority order was not fulfilled


def grade_task(task_id: str) -> Dict:
    result = run_episode(task_id)

    if task_id == "easy_fulfillment":
        score = grade_easy(result)
    elif task_id == "medium_restock":
        score = grade_medium(result)
    elif task_id == "hard_peak_season":
        score = grade_hard(result)
    else:
        raise ValueError(f"Unknown task id: {task_id}")

    score = to_strict_unit_interval(score)

    return {
        "task_id": task_id,
        "score": round(float(score), 2),
        "total_reward": round(float(result["total_reward"]), 2),
        "steps": result["steps"],
        "remaining_orders": result["remaining_orders"],
    }


def grade_all_tasks() -> Dict:
    task_results = [grade_task(task_id) for task_id in KNOWN_TASKS]
    avg_score = sum(item["score"] for item in task_results) / len(task_results)
    return {
        "tasks": task_results,
        "average_score": round(float(avg_score), 2),
    }


if __name__ == "__main__":
    print(json.dumps(grade_all_tasks(), indent=2))
