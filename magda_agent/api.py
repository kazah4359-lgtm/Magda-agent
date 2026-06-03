import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from magda_agent.llm_client import LLMClient
from magda_agent.emotions.engine import EmotionalEngine
from magda_agent.memory.storage import MemorySystem
from magda_agent.skills import initialize_skills
from magda_agent.consciousness.core import Consciousness
from magda_agent.subconsciousness.reflection import Subconsciousness

logging.basicConfig(level=logging.INFO)

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
    interval=300
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    asyncio.create_task(subconsciousness.start())
    yield
    # Shutdown
    await subconsciousness.stop()

app = FastAPI(title="Magda Consciousness API", lifespan=lifespan)

class ProcessInputRequest(BaseModel):
    text: str

class ProcessInputResponse(BaseModel):
    response: str

@app.post("/process", response_model=ProcessInputResponse)
async def process_input(req: ProcessInputRequest):
    resp = await consciousness.process_input(req.text)
    return ProcessInputResponse(response=resp)

class StateResponse(BaseModel):
    state: str

@app.get("/state", response_model=StateResponse)
async def get_state():
    return StateResponse(state=consciousness.get_internal_state())
