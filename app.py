from fastapi import FastAPI
from environment import WarehouseEnv
from models import LogisticsAction

app = FastAPI()
env = WarehouseEnv()

@app.get("/")
async def health():
    return {"status": "ok", "service": "logistics_flow"}

@app.post("/reset")
async def reset(task_id: str = "easy_fulfillment"):
    return env.reset(task_id=task_id)

@app.post("/step")
async def step(action: LogisticsAction):
    obs, reward, done, info = env.step(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.get("/state")
async def state():
    return env._get_obs()