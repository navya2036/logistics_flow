# # import json
# # import os
# # import re

# # import requests
# # from dotenv import load_dotenv
# # from openai import OpenAI

# # load_dotenv()

# # API_BASE_URL = (os.getenv("API_BASE_URL") or "").strip()
# # MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
# # API_KEY = (os.getenv("API_KEY") or "").strip()
# # ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
# # BENCHMARK_ENV = os.getenv("BENCHMARK_ENV", "logistics_flow")
# # AGENT_MODE = os.getenv("AGENT_MODE", "hybrid").lower()
# # MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))
# # LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "10"))
# # HYBRID_LLM_CALLS_PER_TASK = int(os.getenv("HYBRID_LLM_CALLS_PER_TASK", "1"))

# # # If TASK_ID is set, run only that task; otherwise run all 3.
# # _env_task_id = (os.getenv("TASK_ID") or "").strip()
# # ALL_TASK_IDS = ["easy_fulfillment", "medium_restock", "hard_peak_season"]
# # TASK_IDS = [_env_task_id] if _env_task_id else ALL_TASK_IDS

# # if not API_BASE_URL:
# #     raise ValueError(
# #         "API_BASE_URL environment variable is required "
# #         "(use the injected LiteLLM proxy URL)"
# #     )

# # if not API_KEY:
# #     raise ValueError(
# #         "API_KEY environment variable is required "
# #         "(use the injected LiteLLM proxy credentials)"
# #     )

# # client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
# import json
# import os
# import re
# import requests

# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()

# API_BASE_URL = os.environ["API_BASE_URL"]
# API_KEY = os.environ["API_KEY"]
# MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")

# print(f"[CONFIG] base_url={API_BASE_URL}")

# client = OpenAI(
#     api_key=API_KEY,
#     base_url=API_BASE_URL
# )


# def parse_json_response(response: requests.Response, context: str):
#     if not response.ok:
#         raise RuntimeError(f"{context} status={response.status_code}")
#     try:
#         return response.json()
#     except requests.exceptions.JSONDecodeError as exc:
#         raise RuntimeError(f"{context} returned non-json") from exc


# def parse_action_payload(raw_content: str):
#     content = (raw_content or "").strip()
#     if not content:
#         raise ValueError("empty model response")

#     try:
#         return json.loads(content)
#     except json.JSONDecodeError:
#         pass

#     fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content, re.IGNORECASE)
#     if fenced:
#         return json.loads(fenced.group(1))

#     inline = re.search(r"(\{[\s\S]*\})", content)
#     if inline:
#         return json.loads(inline.group(1))

#     raise ValueError("invalid action json")


# def choose_rule_based_action(obs):
#     inventory = obs.get("inventory", {})
#     orders = obs.get("pending_orders", [])
#     incoming = obs.get("incoming_restock", [])

#     if not orders:
#         return {"action_type": "noop"}

#     incoming_by_item = {}
#     for evt in incoming:
#         item = evt.get("item_id")
#         qty = int(evt.get("quantity", 0))
#         incoming_by_item[item] = incoming_by_item.get(item, 0) + qty

#     def order_rank(order):
#         return (
#             order.get("due_day", 999),
#             -order.get("priority", 0),
#             -float(order.get("penalty", 0.0)),
#         )

#     sorted_orders = sorted(orders, key=order_rank)

#     for order in sorted_orders:
#         item = order["item"]
#         need = int(order["quantity"])
#         have = int(inventory.get(item, 0))
#         if have >= need:
#             return {"action_type": "fulfill", "order_id": order["id"]}

#     target = sorted_orders[0]
#     item = target["item"]
#     need = int(target["quantity"])
#     have = int(inventory.get(item, 0))
#     incoming_qty = int(incoming_by_item.get(item, 0))
#     shortage = need - have

#     if shortage <= 0 or incoming_qty >= shortage:
#         return {"action_type": "noop"}

#     return {
#         "action_type": "restock",
#         "item_id": item,
#         "quantity": shortage - incoming_qty,
#     }


# def sanitize_error(err: Exception):
#     msg = str(err).replace("\n", " ").replace("\r", " ").strip()
#     if not msg:
#         return "error"
#     return msg[:140]


# def build_prompt(obs):
#     return f"""
# You are a Warehouse Manager maximizing total reward.
# Task: {obs.get('task_id')}
# Current Day: {obs.get('current_day')}
# Inventory: {obs.get('inventory')}
# Pending Orders: {obs.get('pending_orders')}
# Incoming Restock: {obs.get('incoming_restock')}
# Restock Lead Time: {obs.get('restock_lead_time')} days
# Last Log: {obs.get('log')}

# Choose exactly one action and return only raw JSON:
# - {{"action_type": "fulfill", "order_id": "ID"}}
# - {{"action_type": "restock", "item_id": "ITEM", "quantity": N}}
# - {{"action_type": "noop"}}
# """.strip()


# def choose_action(obs, allow_llm: bool):
#     if AGENT_MODE == "rule":
#         return choose_rule_based_action(obs), None, False

#     if not allow_llm:
#         return choose_rule_based_action(obs), None, False

#     try:
#         completion = client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=[{"role": "user", "content": build_prompt(obs)}],
#             temperature=0.2,
#             timeout=LLM_TIMEOUT_SECONDS,
#         )
#         return parse_action_payload(completion.choices[0].message.content), None, True
#     except Exception as llm_error:
#         if AGENT_MODE == "llm":
#             raise
#         return choose_rule_based_action(obs), sanitize_error(llm_error), True


# def run_inference(task_id: str):
#     """Run one full episode for a given task_id, emitting structured logs."""
#     print(f"[START] task={task_id} env={BENCHMARK_ENV} model={MODEL_NAME}")

#     rewards = []
#     success = False
#     step_count = 0
#     llm_calls_remaining = max(0, HYBRID_LLM_CALLS_PER_TASK)

#     try:
#         reset_resp = requests.post(
#             f"{ENV_BASE_URL}/reset",
#             params={"task_id": task_id},
#             timeout=20,
#         )
#         obs = parse_json_response(reset_resp, "reset")
#         done = bool(obs.get("done", False))

#         while not done and step_count < MAX_STEPS:
#             allow_llm = AGENT_MODE == "llm" or (
#                 AGENT_MODE == "hybrid" and llm_calls_remaining > 0
#             )
#             action, model_error, used_llm = choose_action(obs, allow_llm=allow_llm)
#             if AGENT_MODE == "hybrid" and used_llm and llm_calls_remaining > 0:
#                 llm_calls_remaining -= 1

#             reward_value = 0.0
#             step_error = model_error

#             try:
#                 step_resp = requests.post(
#                     f"{ENV_BASE_URL}/step",
#                     json=action,
#                     timeout=20,
#                 )
#                 result = parse_json_response(step_resp, "step")
#                 # Support both typed reward model {"value": ...} and bare float
#                 raw_reward = result.get("reward", 0.0)
#                 if isinstance(raw_reward, dict):
#                     reward_value = float(raw_reward.get("value", 0.0))
#                 else:
#                     reward_value = float(raw_reward)
#                 obs = result.get("observation", {})
#                 done = bool(result.get("done", False))
#             except Exception as step_exc:
#                 done = True
#                 if step_error is None:
#                     step_error = sanitize_error(step_exc)

#             step_count += 1
#             rewards.append(reward_value)

#             action_str = json.dumps(action, separators=(",", ":"))
#             error_str = "null" if step_error is None else json.dumps(step_error)
#             print(
#                 f"[STEP] step={step_count} action={action_str} "
#                 f"reward={reward_value:.2f} done={str(done).lower()} error={error_str}"
#             )

#         success = done and not obs.get("pending_orders")
#     except Exception:
#         success = False
#     finally:
#         rewards_str = ",".join(f"{value:.2f}" for value in rewards)
#         print(f"[END] success={str(success).lower()} steps={step_count} rewards={rewards_str}")


# if __name__ == "__main__":
#     for task_id in TASK_IDS:
#         run_inference(task_id)






import json
import os
import re
import subprocess
import sys
import time
from urllib.parse import urlparse

import requests

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------- CONFIG ----------------
API_BASE_URL = os.environ["API_BASE_URL"]   # must use injected proxy exactly
API_KEY = os.environ["API_KEY"]             # must use injected key exactly
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
BENCHMARK_ENV = os.getenv("BENCHMARK_ENV", "logistics_flow")
AGENT_MODE = os.getenv("AGENT_MODE", "hybrid").lower()
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "10"))
HYBRID_LLM_CALLS_PER_TASK = int(os.getenv("HYBRID_LLM_CALLS_PER_TASK", "1"))
AUTOSTART_LOCAL_ENV = os.getenv("AUTOSTART_LOCAL_ENV", "true").lower() in {"1", "true", "yes"}

_env_task_id = (os.getenv("TASK_ID") or "").strip()
ALL_TASK_IDS = ["easy_fulfillment", "medium_restock", "hard_peak_season"]
TASK_IDS = [_env_task_id] if _env_task_id else ALL_TASK_IDS

print(f"[CONFIG] base_url={API_BASE_URL}")

# IMPORTANT: do not modify API_BASE_URL
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
    timeout=LLM_TIMEOUT_SECONDS
)

_LOCAL_ENV_PROC = None

# ---------------- HELPERS ----------------
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
    return msg[:140] if msg else "error"


def _is_env_server_reachable() -> bool:
    try:
        resp = requests.get(f"{ENV_BASE_URL}/", timeout=1.5)
        return resp.ok
    except Exception:
        return False


def _start_local_env_server_if_needed():
    global _LOCAL_ENV_PROC

    if _is_env_server_reachable():
        return

    parsed = urlparse(ENV_BASE_URL)
    host = (parsed.hostname or "").lower()
    is_local = host in {"127.0.0.1", "localhost"}
    port = parsed.port or 8000

    if not is_local:
        raise RuntimeError(
            f"ENV_BASE_URL '{ENV_BASE_URL}' is not reachable. "
            "Set ENV_BASE_URL to a live environment endpoint."
        )

    if not AUTOSTART_LOCAL_ENV:
        raise RuntimeError(
            f"Local environment server is not running at {ENV_BASE_URL}. "
            "Start it with: python -m uvicorn app:app --host 127.0.0.1 --port 8000"
        )

    _LOCAL_ENV_PROC = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + 12
    while time.time() < deadline:
        if _is_env_server_reachable():
            return
        time.sleep(0.3)

    raise RuntimeError(
        f"Failed to start local environment server at {ENV_BASE_URL}. "
        "Run manually: python -m uvicorn app:app --host 127.0.0.1 --port 8000"
    )


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


def choose_action(obs, allow_llm: bool):
    if AGENT_MODE == "rule":
        return choose_rule_based_action(obs), None, False

    if not allow_llm:
        return choose_rule_based_action(obs), None, False

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": build_prompt(obs)}],
            temperature=0.2,
        )
        action = parse_action_payload(completion.choices[0].message.content)
        return action, None, True

    except Exception as llm_error:
        if AGENT_MODE == "llm":
            raise
        return choose_rule_based_action(obs), sanitize_error(llm_error), True


# ---------------- MAIN LOOP ----------------
def run_inference(task_id: str):
    print(f"[START] task={task_id} env={BENCHMARK_ENV} model={MODEL_NAME}")

    rewards = []
    success = False
    step_count = 0
    llm_calls_remaining = max(0, HYBRID_LLM_CALLS_PER_TASK)

    try:
        _start_local_env_server_if_needed()

        reset_resp = requests.post(
            f"{ENV_BASE_URL}/reset",
            params={"task_id": task_id},
            timeout=20,
        )
        obs = parse_json_response(reset_resp, "reset")
        done = bool(obs.get("done", False))

        while not done and step_count < MAX_STEPS:
            allow_llm = AGENT_MODE == "llm" or (
                AGENT_MODE == "hybrid" and llm_calls_remaining > 0
            )

            action, model_error, used_llm = choose_action(obs, allow_llm)

            if AGENT_MODE == "hybrid" and used_llm and llm_calls_remaining > 0:
                llm_calls_remaining -= 1

            reward_value = 0.0
            step_error = model_error

            try:
                step_resp = requests.post(
                    f"{ENV_BASE_URL}/step",
                    json=action,
                    timeout=20,
                )
                result = parse_json_response(step_resp, "step")

                raw_reward = result.get("reward", 0.0)
                reward_value = float(raw_reward.get("value", 0.0)) if isinstance(raw_reward, dict) else float(raw_reward)

                obs = result.get("observation", {})
                done = bool(result.get("done", False))

            except Exception as step_exc:
                done = True
                if step_error is None:
                    step_error = sanitize_error(step_exc)

            step_count += 1
            rewards.append(reward_value)

            print(
                f"[STEP] step={step_count} "
                f"action={json.dumps(action)} "
                f"reward={reward_value:.2f} "
                f"done={str(done).lower()} "
                f"error={json.dumps(step_error) if step_error else 'null'}"
            )

        success = done and not obs.get("pending_orders")

    except Exception as e:
        print(f"[ERROR] {e}")
        success = False

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={step_count} rewards={rewards_str}")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    for task_id in TASK_IDS:
        run_inference(task_id)