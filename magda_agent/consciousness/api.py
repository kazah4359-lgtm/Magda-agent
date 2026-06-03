import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from magda_agent.llm_client import LLMClient
from magda_agent.emotions.engine import EmotionalEngine
from magda_agent.memory.storage import MemorySystem
from magda_agent.skills import initialize_skills
from magda_agent.consciousness.core import Consciousness
from magda_agent.subconsciousness.reflection import Subconsciousness

# Global Instances
llm_client = LLMClient()
emotional_engine = EmotionalEngine()
memory_system = MemorySystem()
skill_registry = initialize_skills()

consciousness = Consciousness(
    llm=llm_client,
    emotions=emotional_engine,
    memory=memory_system,
    skills=skill_registry
)

subconsciousness = Subconsciousness(
    llm=llm_client,
    emotions=emotional_engine,
    memory=memory_system,
    interval=300 # Reflect every 5 minutes in production
)

subconsciousness_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global subconsciousness_task
    # Startup
    logging.info("Starting Consciousness Microservice...")
    subconsciousness_task = asyncio.create_task(subconsciousness.start())
    yield
    # Shutdown
    logging.info("Shutting down Consciousness Microservice...")
    await subconsciousness.stop()
    if subconsciousness_task:
        subconsciousness_task.cancel()
        try:
            await subconsciousness_task
        except asyncio.CancelledError:
            pass

app = FastAPI(lifespan=lifespan, title="Magda Consciousness Service")

class ProcessRequest(BaseModel):
    text: str

class ProcessResponse(BaseModel):
    response: str

@app.post("/process", response_model=ProcessResponse)
async def process_input(req: ProcessRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        response_text = await consciousness.process_input(req.text)
        return ProcessResponse(response=response_text)
    except Exception as e:
        logging.error(f"Error processing input: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
async def get_state() -> Dict[str, Any]:
    try:
        state_text = consciousness.get_internal_state()
        return {"state": state_text}
    except Exception as e:
        logging.error(f"Error getting state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("magda_agent.consciousness.api:app", host="0.0.0.0", port=8000, reload=True)
