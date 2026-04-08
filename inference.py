import json
import os
import re

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_BASE_URL = (os.getenv("MODEL_BASE_URL") or "").strip()
API_BASE_URL = (os.getenv("API_BASE_URL") or "").strip()
RESOLVED_MODEL_BASE_URL = API_BASE_URL or MODEL_BASE_URL
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
API_KEY = (os.getenv("API_KEY") or "").strip()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()
RESOLVED_API_KEY = API_KEY or OPENAI_API_KEY or HF_TOKEN
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
TASK_ID = os.getenv("TASK_ID", "easy_fulfillment")
BENCHMARK_ENV = os.getenv("BENCHMARK_ENV", "logistics_flow")
AGENT_MODE = os.getenv("AGENT_MODE", "hybrid").lower()
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))

if API_BASE_URL and not API_KEY:
    raise ValueError(
        "API_KEY environment variable is required when API_BASE_URL is set "
        "(use the injected LiteLLM proxy credentials)"
    )

if not RESOLVED_API_KEY:
    raise ValueError("API_KEY (or OPENAI_API_KEY / HF_TOKEN) environment variable is required")

if not RESOLVED_MODEL_BASE_URL:
    raise ValueError(
        "MODEL_BASE_URL or API_BASE_URL environment variable is required "
        "(set it to the provided LiteLLM proxy URL)"
    )

client = OpenAI(base_url=RESOLVED_MODEL_BASE_URL, api_key=RESOLVED_API_KEY)


def parse_json_response(response: requests.Response, context: str):
    if not response.ok:
        raise RuntimeError(f"{context} status={response.status_code}")
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise RuntimeError(f"{context} returned non-json") from exc


def parse_action_payload(raw_content: str):
    content = (raw_content or "").strip()
    if not content:
        raise ValueError("empty model response")

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content, re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    inline = re.search(r"(\{[\s\S]*\})", content)
    if inline:
        return json.loads(inline.group(1))

    raise ValueError("invalid action json")


def choose_rule_based_action(obs):
    inventory = obs.get("inventory", {})
    orders = obs.get("pending_orders", [])
    incoming = obs.get("incoming_restock", [])

    if not orders:
        return {"action_type": "noop"}

    incoming_by_item = {}
    for evt in incoming:
        item = evt.get("item_id")
        qty = int(evt.get("quantity", 0))
        incoming_by_item[item] = incoming_by_item.get(item, 0) + qty

    def order_rank(order):
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


def sanitize_error(err: Exception):
    msg = str(err).replace("\n", " ").replace("\r", " ").strip()
    if not msg:
        return "error"
    return msg[:140]


def build_prompt(obs):
    return f"""
You are a Warehouse Manager maximizing total reward.
Task: {obs.get('task_id')}
Current Day: {obs.get('current_day')}
Inventory: {obs.get('inventory')}
Pending Orders: {obs.get('pending_orders')}
Incoming Restock: {obs.get('incoming_restock')}
Restock Lead Time: {obs.get('restock_lead_time')} days
Last Log: {obs.get('log')}

Choose exactly one action and return only raw JSON:
- {{"action_type": "fulfill", "order_id": "ID"}}
- {{"action_type": "restock", "item_id": "ITEM", "quantity": N}}
- {{"action_type": "noop"}}
""".strip()


def choose_action(obs):
    if AGENT_MODE == "rule":
        return choose_rule_based_action(obs), None

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": build_prompt(obs)}],
            temperature=0.2,
        )
        return parse_action_payload(completion.choices[0].message.content), None
    except Exception as llm_error:
        if AGENT_MODE == "llm":
            raise
        return choose_rule_based_action(obs), sanitize_error(llm_error)


def run_inference():
    print(f"[START] task={TASK_ID} env={BENCHMARK_ENV} model={MODEL_NAME}")

    rewards = []
    success = False
    step_count = 0

    try:
        reset_resp = requests.post(
            f"{ENV_BASE_URL}/reset",
            params={"task_id": TASK_ID},
            timeout=20,
        )
        obs = parse_json_response(reset_resp, "reset")
        done = bool(obs.get("done", False))

        while not done and step_count < MAX_STEPS:
            action, model_error = choose_action(obs)
            reward_value = 0.0
            step_error = model_error

            try:
                step_resp = requests.post(
                    f"{ENV_BASE_URL}/step",
                    json=action,
                    timeout=20,
                )
                result = parse_json_response(step_resp, "step")
                reward_value = float(result.get("reward", 0.0))
                obs = result.get("observation", {})
                done = bool(result.get("done", False))
            except Exception as step_exc:
                done = True
                if step_error is None:
                    step_error = sanitize_error(step_exc)

            step_count += 1
            rewards.append(reward_value)

            action_str = json.dumps(action, separators=(",", ":"))
            error_str = "null" if step_error is None else json.dumps(step_error)
            print(
                f"[STEP] step={step_count} action={action_str} "
                f"reward={reward_value:.2f} done={str(done).lower()} error={error_str}"
            )

        success = done and not obs.get("pending_orders")
    except Exception:
        success = False
    finally:
        rewards_str = ",".join(f"{value:.2f}" for value in rewards)
        print(f"[END] success={str(success).lower()} steps={step_count} rewards={rewards_str}")


if __name__ == "__main__":
    run_inference()