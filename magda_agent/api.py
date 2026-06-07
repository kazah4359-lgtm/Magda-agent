import asyncio
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from magda_agent.visualization.server import CanvasServer
from pydantic import BaseModel

from typing import Any, Dict, List, Optional

from magda_agent.llm_client import LLMClient
from magda_agent.safety.policy import PolicyLayer
from magda_agent.action.selector import BasalGanglia
from magda_agent.emotions.engine import EmotionalEngine
from magda_agent.emotions.attachment import AttachmentModel
from magda_agent.emotions.style_adapter import StyleAdapter
from magda_agent.memory.storage import MemorySystem
from magda_agent.memory.procedural import ProceduralMemory
from magda_agent.memory.semantic import SemanticMemory
from magda_agent.learning.skill_creator import SkillCreator
from magda_agent.learning.skill_versioning import SkillVersioning
from magda_agent.skills import initialize_skills
from magda_agent.planning.planner import Planner
from magda_agent.consciousness.core import Consciousness
from magda_agent.subconsciousness.reflection import Subconsciousness
from magda_agent.scheduler.cron import CronScheduler
from magda_agent.autonomy.task_store import TaskStore, TaskStatus
from magda_agent.autonomy.executor import AutonomousExecutor
from magda_agent.memory.long_term import LongTermMemory
from magda_agent.metacognition.evaluator import Evaluator
from magda_agent.metacognition.assert_evaluator import AssertEvaluator
from magda_agent.metacognition.confidence import ConfidenceCalibrator
from magda_agent.metacognition.tracker import QualityTracker
from magda_agent.learning.habits import HabitTracker
from magda_agent.learning.online import OnlineLearner
from magda_agent.learning.openclaw_rl import OpenClawInteractiveLearner
from magda_agent.user_model.model import UserModel
from magda_agent.reflexes.brainstem import Brainstem
from magda_agent.thalamus.router import Thalamus
from magda_agent.drives.hypothalamus import Hypothalamus
from magda_agent.emotions.insula import Insula
from magda_agent.rhythms.pineal_gland import PinealGland
from magda_agent.emotions.mirror_neurons import MirrorNeurons
from magda_agent.attention.salience import SalienceNetwork
from magda_agent.attention.workspace import GlobalWorkspace
from magda_agent.safety.guardrails import RealtimeGuardrail
from magda_agent.context.engine import ContextEngine
from magda_agent.context.default_plugin import DefaultContextPlugin
from magda_agent.tracing.tracer import ThoughtChainTracer

logging.basicConfig(level=logging.INFO)

llm_client = LLMClient()
emotional_engine = EmotionalEngine()

# Initialize ContextEngine with default plugin
context_engine = ContextEngine(plugins=[DefaultContextPlugin(llm=llm_client)])

memory_system = MemorySystem(llm=llm_client, context_engine=context_engine)
procedural_memory = ProceduralMemory(persist_directory="./procedural_memory_db")
semantic_memory = SemanticMemory(persist_directory="./semantic_memory_db")
skill_creator = SkillCreator(procedural_memory=procedural_memory, llm=llm_client)
skill_versioning = SkillVersioning(procedural_memory=procedural_memory)
policy_layer = PolicyLayer()
skill_registry = initialize_skills(policy_layer=policy_layer)
basal_ganglia = BasalGanglia(policy_layer=policy_layer)
habit_tracker = HabitTracker()
mirror_neurons = MirrorNeurons()
quality_tracker = QualityTracker()
confidence_calibrator = ConfidenceCalibrator(llm=llm_client, tracker=quality_tracker)
online_learner = OnlineLearner(
    habit_tracker=habit_tracker,
    memory=memory_system,
    mirror_neurons=mirror_neurons,
)
style_adapter = StyleAdapter()
user_model = UserModel(llm=llm_client)

openclaw_rl = OpenClawInteractiveLearner(
    habit_tracker=habit_tracker,
    mirror_neurons=mirror_neurons,
    user_model=user_model,
)
brainstem = Brainstem()
planner = Planner(llm=llm_client, skills=skill_registry, habit_tracker=habit_tracker)
long_term_memory = LongTermMemory()
evaluator = Evaluator(llm=llm_client, memory=memory_system)
assert_evaluator = AssertEvaluator(llm=llm_client, memory=memory_system)
attachment_model = AttachmentModel()
thalamus = Thalamus()
hypothalamus = Hypothalamus()
insula = Insula()
pineal_gland = PinealGland()
salience_network = SalienceNetwork()
global_workspace = GlobalWorkspace(salience_network=salience_network)
guardrail = RealtimeGuardrail(policy_layer=policy_layer)
thought_chain_tracer = ThoughtChainTracer()

consciousness = Consciousness(
    llm=llm_client,
    emotions=emotional_engine,
    memory=memory_system,
    skills=skill_registry,
    planner=planner,
    long_term_memory=long_term_memory,
    evaluator=evaluator,
    assert_evaluator=assert_evaluator,
    confidence_calibrator=confidence_calibrator,
    habit_tracker=habit_tracker,
    attachment=attachment_model,
    thalamus=thalamus,
    basal_ganglia=basal_ganglia,
    hypothalamus=hypothalamus,
    insula=insula,
    brainstem=brainstem,
    pineal_gland=pineal_gland,
    mirror_neurons=mirror_neurons,
    salience=salience_network,
    global_workspace=global_workspace,
    context_engine=context_engine,
    skill_creator=skill_creator,
    online_learner=online_learner,
    openclaw_rl=openclaw_rl,
    guardrail=guardrail,
    tracer=thought_chain_tracer,
    style_adapter=style_adapter,
    user_model=user_model,
    skill_versioning=skill_versioning,
)

cron_scheduler = CronScheduler()

task_store = TaskStore(path=os.getenv("AUTONOMY_TASKS_PATH", "./autonomy_tasks.json"))
autonomous_executor = AutonomousExecutor(
    store=task_store,
    llm=llm_client,
    skills=skill_registry,
    step_timeout=float(os.getenv("AUTONOMY_STEP_TIMEOUT", "60")),
)

canvas_server = CanvasServer(consciousness=consciousness)

subconsciousness = Subconsciousness(
    llm=llm_client,
    emotions=emotional_engine,
    memory=memory_system,
    procedural_memory=procedural_memory,
    semantic_memory=semantic_memory,
    interval=300
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    asyncio.create_task(subconsciousness.start())
    asyncio.create_task(cron_scheduler.start())
    await autonomous_executor.start()
    asyncio.create_task(canvas_server.start_streaming())
    yield
    # Shutdown
    await subconsciousness.stop()
    await cron_scheduler.stop()
    await autonomous_executor.stop()
    await canvas_server.stop_streaming()
    memory_system.close()

app = FastAPI(title="Magda Consciousness API", lifespan=lifespan)

class ProcessInputRequest(BaseModel):
    text: str
    user_id: Optional[int] = None

class ProcessInputResponse(BaseModel):
    response: str

@app.post("/process", response_model=ProcessInputResponse)
async def process_input(req: ProcessInputRequest):
    resp = await consciousness.process_input(req.text, req.user_id)
    return ProcessInputResponse(response=resp)

class StateResponse(BaseModel):
    state: str

@app.get("/state", response_model=StateResponse)
async def get_state():
    return StateResponse(state=consciousness.get_internal_state())

class HealthResponse(BaseModel):
    status: str

@app.get("/health", response_model=HealthResponse)
async def healthcheck():
    return HealthResponse(status="ok")

class TraceResponse(BaseModel):
    trace: list

@app.get("/trace", response_model=TraceResponse)
async def get_trace():
    return TraceResponse(trace=thought_chain_tracer.get_trace())


# --- Autonomous long-running tasks ---------------------------------------

class CreateTaskRequest(BaseModel):
    goal: str
    user_id: Optional[int] = None
    max_iterations: int = 20


class TaskSummaryResponse(BaseModel):
    task: Dict[str, Any]


class TaskListResponse(BaseModel):
    tasks: List[Dict[str, Any]]


@app.post("/tasks", response_model=TaskSummaryResponse)
async def create_task(req: CreateTaskRequest):
    max_iters = max(1, min(req.max_iterations, 200))
    task = await task_store.add_task(req.goal, user_id=req.user_id, max_iterations=max_iters)
    return TaskSummaryResponse(task=task.summary())


@app.get("/tasks", response_model=TaskListResponse)
async def list_tasks(user_id: Optional[int] = None):
    tasks = await task_store.list(user_id=user_id)
    return TaskListResponse(tasks=[t.summary() for t in tasks])


@app.get("/tasks/{task_id}")
async def get_task(task_id: str, since: int = 0):
    task = await task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    data = task.to_dict()
    if since > 0:
        data["progress"] = task.progress[since:]
    return data


@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    ok = await task_store.request_cancel(task_id)
    if not ok:
        raise HTTPException(status_code=409, detail="Task cannot be cancelled")
    return {"status": "ok"}


@app.post("/tasks/{task_id}/pause")
async def pause_task(task_id: str):
    ok = await task_store.request_pause(task_id)
    if not ok:
        raise HTTPException(status_code=409, detail="Task cannot be paused")
    return {"status": "ok"}


@app.post("/tasks/{task_id}/resume")
async def resume_task(task_id: str):
    ok = await task_store.resume(task_id)
    if not ok:
        raise HTTPException(status_code=409, detail="Task cannot be resumed")
    return {"status": "ok"}


@app.websocket("/ws/canvas")
async def websocket_canvas(websocket: WebSocket):
    await canvas_server.connect(websocket)
    try:
        while True:
            # We just keep the connection open, client doesn't need to send anything
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        canvas_server.disconnect(websocket)
