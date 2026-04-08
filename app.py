from fastapi import FastAPI
from environment import WarehouseEnv
from models import LogisticsAction, LogisticsReward, StepResult

app = FastAPI()
env = WarehouseEnv()


@app.get("/")
async def health():
    return {"status": "ok", "service": "logistics_flow"}


@app.post("/reset")
async def reset(task_id: str = "easy_fulfillment"):
    return env.reset(task_id=task_id)


@app.post("/step", response_model=StepResult)
async def step(action: LogisticsAction):
    obs, reward, done, info = env.step(action)
    reward_breakdown = info.get("reward_breakdown", {})
    typed_reward = LogisticsReward(value=float(reward), breakdown=reward_breakdown)
    return StepResult(observation=obs, reward=typed_reward, done=done, info=info)


@app.get("/state")
async def state():
    return env._get_obs()