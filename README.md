# logistics_flow

Intelligent warehouse operations environment built for OpenEnv-style agent evaluation.

## Motivation

This environment simulates a real operations workflow: inventory management, fulfillment timing, and priority decisions under stock constraints.

## Observation Space

`LogisticsObservation` fields:

- `inventory`: item-to-stock dictionary
- `pending_orders`: list of orders with due day, priority, and penalty
- `current_day`: simulation day index
- `task_id`: current task scenario
- `restock_lead_time`: days for restock arrival
- `incoming_restock`: queued restocks with arrival day
- `log`: latest system event
- `done`: episode finished flag

## Action Space

`LogisticsAction` supports:

- `{"action_type": "fulfill", "order_id": "..."}`
- `{"action_type": "restock", "item_id": "...", "quantity": N}`
- `{"action_type": "noop"}`

## Reward Signal

Trajectory reward is shaped to provide useful learning signal:

- Positive reward for successful fulfillment weighted by order priority
- Small penalty for restocking cost
- Penalty for invalid fulfillment attempts
- Order-specific penalty when orders expire

## Tasks

Defined in `task_scenarios.json`:

- `easy_fulfillment` (easy): fulfill simple orders with ample stock
- `medium_restock` (medium): account for 2-day lead time before fulfillment
- `hard_peak_season` (hard): prioritize high-value orders under stock shortage

## Deterministic Graders (0.0-1.0)

Run:

```bash
python grade_tasks.py
```

Output includes per-task deterministic score in `[0.0, 1.0]` and average score.

## API Endpoints

- `GET /` (health)
- `POST /reset?task_id=<id>`
- `POST /step`
- `GET /state`

## Local Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment variables:

Copy `.env.example` to `.env` and fill in tokens.

Required for baseline inference:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` (or `OPENAI_API_KEY`)

3. Start environment server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

4. Run baseline inference:

```bash
python inference.py
```

5. Run deterministic graders (0.0-1.0):

```bash
python grade_tasks.py
```

## Structured Inference Logs

The inference script emits strict lines only:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

## Docker

Build:

```bash
docker build -t logistics-flow .
```

Run:

```bash
docker run -p 7860:7860 logistics-flow
```

Health checks:

```bash
curl http://localhost:7860/
curl -X POST "http://localhost:7860/reset?task_id=easy_fulfillment"
```

## Hugging Face Spaces Deployment

Use a Docker Space.

1. Create a new Space on Hugging Face.
2. Select `Docker` as SDK.
3. Push this repository to the Space.
4. In Space Settings -> Variables and secrets, set:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
   - (optional) `OPENAI_API_KEY` (same value as `HF_TOKEN`)
5. Ensure the Space metadata includes the `openenv` tag.
6. After build is complete, verify:
   - `GET /` returns 200
   - `POST /reset` returns 200

## Pre-Submission Checklist

- Space deploys and root ping returns 200.
- `POST /reset` works on deployed URL.
- `openenv.yaml` includes task metadata.
- `step/reset/state` endpoints respond correctly.
- Docker image builds with `docker build -t logistics-flow .`.
- `python inference.py` runs and emits strict `[START]`, `[STEP]`, `[END]` logs.
- `python grade_tasks.py` returns task scores in `[0.0, 1.0]`.

## OpenEnv Metadata

`openenv.yaml` defines task metadata (easy/medium/hard).
