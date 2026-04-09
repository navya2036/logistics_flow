# logistics_flow

An intelligent warehouse operations environment built for [OpenEnv](https://openenv.dev)-compatible agent evaluation. It simulates real-world inventory management decisions — fulfillment timing, stock restocking, and priority resolution under constraints.

---

## Project Structure

```
logistics_flow/
├── app.py               # FastAPI server exposing the environment as a REST API
├── environment.py       # Core WarehouseEnv simulation logic
├── models.py            # Typed Pydantic models (action, observation, reward, step result)
├── graders.py           # Deterministic task graders (score strictly in (0, 1))
├── grade_tasks.py       # Entry point to run all graders and print JSON results
├── inference.py         # LLM/rule-based agent that interacts with the environment API
├── task_scenarios.json  # Scenario definitions for all three tasks
├── openenv.yaml         # OpenEnv metadata (task list, difficulty, goals)
├── Dockerfile           # Container config — exposes port 7860
├── .env.example         # Template for required environment variables
├── requirements.txt     # Runtime dependencies
└── requirements-dev.txt # Dev/validation-only dependencies
```

---

## How It Works

### Environment (`environment.py`)

`WarehouseEnv` is a discrete-time simulation. At each step:
1. Pending restock orders that have arrived are applied to inventory.
2. The agent takes one action (fulfill, restock, or noop).
3. Time advances by one day.
4. Expired orders are penalized and removed.
5. The episode ends when all orders are resolved or `max_days` is reached.

### API Server (`app.py`)

A FastAPI app wraps `WarehouseEnv` and exposes four endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check — returns `{"status": "ok"}` |
| `POST` | `/reset?task_id=<id>` | Reset environment to a named scenario |
| `POST` | `/step` | Submit an action, get observation + reward |
| `GET`  | `/state` | Get the current observation without stepping |

### Inference Agent (`inference.py`)

Supports three agent modes (set via `AGENT_MODE` env var):

| Mode | Behaviour |
|------|-----------|
| `rule` | Pure rule-based: sort orders by urgency, fulfill if stock available, else restock |
| `llm` | Pure LLM: all decisions via the configured model |
| `hybrid` | LLM for the first N steps (default 1), then falls back to rule-based |

The agent auto-starts a local uvicorn server if `ENV_BASE_URL` is localhost and the server isn't running.

Runs all three tasks by default unless `TASK_ID` is set.

---

## Data Models (`models.py`)

All models are typed Pydantic v2 schemas.

### `LogisticsAction`
```python
action_type: Literal["fulfill", "restock", "noop"]
order_id: Optional[str]   # required for "fulfill"
item_id: Optional[str]    # required for "restock"
quantity: int             # default 1
```

### `LogisticsObservation`
```python
inventory: Dict[str, int]          # current stock per item
pending_orders: List[Order]        # active orders not yet fulfilled or expired
current_day: int                   # simulation time
task_id: str                       # active scenario name
restock_lead_time: int             # days until a restock arrives
incoming_restock: List[IncomingRestock]  # queued arrivals
log: str                           # last system event message
done: bool                         # True when episode is finished
```

### `LogisticsReward`
```python
value: float                       # total reward for this step
breakdown: Dict[str, float]        # per-component breakdown
```

Reward components:
- `fulfill`: `+priority` (1–3) on successful fulfillment
- `restock_cost`: `−0.1` per restock action
- `invalid_action`: `−0.5` for failed fulfill attempts
- `expiry_penalty`: `−penalty` (order-defined) when order expires

### `StepResult`
```python
observation: LogisticsObservation
reward: LogisticsReward
done: bool
info: Dict[str, Any]
```

---

## Task Scenarios (`task_scenarios.json`)

Three scenarios of increasing difficulty:

| Task ID | Difficulty | Key Challenge |
|---------|-----------|---------------|
| `easy_fulfillment` | Easy | Ample stock, 2 orders, no lead time — straightforward fulfillment |
| `medium_restock` | Medium | Low initial stock, 2-day restock lead time, must plan ahead |
| `hard_peak_season` | Hard | Insufficient stock for all orders; must prioritize high-value orders under strict deadlines |

**`easy_fulfillment`** — inventory: `electronics×10, appliances×6`, lead time: 0 days  
**`medium_restock`** — inventory: `electronics×2, appliances×0`, lead time: 2 days  
**`hard_peak_season`** — inventory: `electronics×6, appliances×1`, lead time: 2 days, 3 competing orders all due day 2

---

## Graders (`graders.py` / `grade_tasks.py`)

Each task has a deterministic grader. All scores are clamped strictly inside `(0.0, 1.0)` — never exactly 0 or 1 — using `to_strict_unit_interval()`.

### `easy_fulfillment`
| Condition | Score |
|-----------|-------|
| Not done | 0.05 |
| All orders fulfilled, reward ≥ 4.0 | **0.95** |
| All orders fulfilled | 0.80 |
| Otherwise | 0.30 |

### `medium_restock`
| Condition | Score |
|-----------|-------|
| All orders done + restock happened before any fulfill | **0.95** |
| All orders done (wrong strategy) | 0.70 |
| Restock first, but orders remain | 0.50 |
| Wrong strategy + unfulfilled | 0.10 |

### `hard_peak_season`
| Condition | Score |
|-----------|-------|
| All orders done + total reward ≥ 3.0 | **0.95** |
| High-priority order (ORD201) done + reward ≥ 2.0 | 0.80 |
| High-priority order done | 0.55 |
| High-priority order missed | 0.10 |

Run all graders:
```bash
python grade_tasks.py
```

---

## Baseline Scores

Rule-based agent results (run `python grade_tasks.py`):

| Task | Score | Total Reward | Steps | Remaining Orders |
|------|-------|-------------|-------|-----------------|
| `easy_fulfillment` | **0.95** | 5.0 | 2 | 0 |
| `medium_restock` | **0.95** | 2.9 | 4 | 0 |
| `hard_peak_season` | **0.95** | 3.0 | 2 | 0 |
| **Average** | **0.95** | — | — | — |

> Scores are strictly within `(0.0, 1.0)`. `0.95` = near-perfect. The LLM-backed agent (`AGENT_MODE=llm`) may vary per model.

---

## Local Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

**Required for inference:**

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LiteLLM proxy URL injected by the evaluation platform |
| `API_KEY` | LiteLLM API key injected by the evaluation platform |
| `MODEL_NAME` | Model identifier (default: `gpt-4.1-mini`) |

**Optional inference controls:**

| Variable | Default | Description |
|----------|---------|-------------|
| `TASK_ID` | *(none — runs all)* | Run a single task instead of all three |
| `AGENT_MODE` | `hybrid` | `rule`, `llm`, or `hybrid` |
| `MAX_STEPS` | `12` | Max steps per episode |
| `LLM_TIMEOUT_SECONDS` | `10` | LLM call timeout |
| `HYBRID_LLM_CALLS_PER_TASK` | `1` | LLM calls before falling back to rules |
| `AUTOSTART_LOCAL_ENV` | `true` | Auto-launch uvicorn if `ENV_BASE_URL` is localhost |
| `ENV_BASE_URL` | `http://127.0.0.1:8000` | Environment server address |

### 3. Start the environment server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

> **Note:** If `AUTOSTART_LOCAL_ENV=true` (the default), `inference.py` will start this automatically.

### 4. Run baseline inference

```bash
python inference.py
```

### 5. Grade all tasks

```bash
python grade_tasks.py
```

---

## Structured Inference Logs

`inference.py` emits strictly structured log lines:

```
[CONFIG] base_url=...
[START] task=<id> env=logistics_flow model=<model>
[STEP]  step=N action={...} reward=X.XX done=false error=null
[END]   success=true steps=N score=0.XXX rewards=X.XX,...
```

---

## Docker

Build image:
```bash
docker build -t logistics-flow .
```

Run container:
```bash
docker run -p 7860:7860 logistics-flow
```

Health checks:
```bash
curl http://localhost:7860/
curl -X POST "http://localhost:7860/reset?task_id=easy_fulfillment"
```

---

## Hugging Face Spaces Deployment

This project is designed to run as a **Docker Space** on Hugging Face.

1. Create a new Space → select **Docker** as SDK.
2. Push this repository to the Space.
3. In **Space Settings → Variables and Secrets**, set:
   - `API_BASE_URL` *(required)*
   - `API_KEY` *(required)*
   - `MODEL_NAME` *(optional, default: `gpt-4.1-mini`)*
   - `AGENT_MODE` *(optional, default: `hybrid`)*
4. Ensure the Space metadata includes the `openenv` tag.
5. After build completes, verify:
   - `GET /` → `{"status": "ok"}`
   - `POST /reset?task_id=easy_fulfillment` → observation JSON

---