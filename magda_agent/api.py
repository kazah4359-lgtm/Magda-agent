import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from magda_agent.visualization.server import CanvasServer
from magda_agent.architecture.gateway import LocalFirstGateway
from magda_agent.visualization.canvas_api_v2 import get_canvas_v2_router
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
from magda_agent.integration.a2a_server import A2AServer
from magda_agent.integration.a2a_discovery import A2ADiscovery, AgentCard
from magda_agent.integration.a2a_delegation import A2ADelegator
from magda_agent.integration.a2a_security import A2ASecurityContext
from magda_agent.consciousness.core import Consciousness
from magda_agent.subconsciousness.reflection import Subconsciousness
from magda_agent.evaluation.agentbench import daily_agentbench_eval
from magda_agent.scheduler.cron import CronScheduler
from magda_agent.scheduler.cron_reports import DailyReportScheduler

from magda_agent.operations.cron_v3 import HermesCronSchedulerV3
from magda_agent.scheduler.autonomous_tasks import run_health_check, report_quality_metrics
from magda_agent.autonomy.task_store import TaskStore, TaskStatus
from magda_agent.autonomy.executor import AutonomousExecutor
from magda_agent.memory.long_term import LongTermMemory
from magda_agent.metacognition.evaluator import Evaluator
from magda_agent.metacognition.assert_evaluator import AssertEvaluator
from magda_agent.metacognition.confidence import ConfidenceCalibrator
from magda_agent.metacognition.tracker import QualityTracker
from magda_agent.learning.habits import HabitTracker
from magda_agent.learning.online import OnlineLearner
from magda_agent.learning.dialogue_v3 import DialogueOnlineLearnerV3
from magda_agent.learning.online_rl import OnlineRLIntegrator
from magda_agent.learning.openclaw_rl_v5 import OnlineRLIntegrator as OpenClawRLV5Integrator
from magda_agent.learning.online_rl_v6 import OnlineRLFeedbackLoopV6
from magda_agent.learning.openclaw_rl import OpenClawInteractiveLearner
from magda_agent.learning.lessons import TaskRecoveryLessons
from magda_agent.user_model.model import UserModel
from magda_agent.reflexes.brainstem import Brainstem
from magda_agent.thalamus.router import Thalamus
from magda_agent.drives.hypothalamus import Hypothalamus
from magda_agent.emotions.insula import Insula
from magda_agent.rhythms.pineal_gland import PinealGland
from magda_agent.emotions.mirror_neurons import MirrorNeurons
from magda_agent.attention.salience import SalienceNetwork
from magda_agent.gateway.router import GatewayRouter
from magda_agent.attention.workspace import GlobalWorkspace
from magda_agent.safety.guardrails import RealtimeGuardrail
from magda_agent.memory.context_engine import ContextEngine
from magda_agent.memory.default_context_plugin import DefaultContextPlugin
from magda_agent.skills.compression_v2 import OpenClawContextCompressorV2
from magda_agent.tracing.tracer import ThoughtChainTracer
from magda_agent.architecture.sub_agents import SubAgentRPCManager
from magda_agent.integration.cross_platform import CrossPlatformDispatcher
from magda_agent.integration.discord_bridge import DiscordBridge

logging.basicConfig(level=logging.INFO)

llm_client = LLMClient()
emotional_engine = EmotionalEngine()

# Initialize ContextEngine with default plugin and OpenClaw compressor
context_engine = ContextEngine(plugins=[
    DefaultContextPlugin(llm=llm_client),
    OpenClawContextCompressorV2(llm=llm_client)
])

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

online_rl_integrator = OnlineRLIntegrator(
    habit_tracker=habit_tracker,
    mirror_neurons=mirror_neurons
)

openclaw_rl_v5 = OpenClawRLV5Integrator()

dialogue_online_learner_v3 = DialogueOnlineLearnerV3()

online_rl_v6 = OnlineRLFeedbackLoopV6(
    habit_tracker=habit_tracker,
    mirror_neurons=mirror_neurons
)
style_adapter = StyleAdapter()
user_model = UserModel(llm=llm_client)

from magda_agent.learning.feedback_loop import FeedbackLoop

feedback_loop = FeedbackLoop(mirror_neurons=mirror_neurons, user_model=user_model)

recovery_lessons = TaskRecoveryLessons(procedural_memory=procedural_memory, llm=llm_client)

openclaw_rl = OpenClawInteractiveLearner(
    habit_tracker=habit_tracker,
    mirror_neurons=mirror_neurons,
    user_model=user_model,
    recovery_lessons=recovery_lessons
)

# A2A Integration Setup
a2a_security = A2ASecurityContext()
local_card = AgentCard(
    agent_id=os.getenv("AGENT_ID", "magda-agent-1"),
    name=os.getenv("AGENT_NAME", "Magda"),
    description="Experimental cognitive AGI agent",
    capabilities=["coding", "analysis", "reflection"],
    endpoints={"mcp": f"{os.getenv('BASE_URL', 'http://localhost:8000')}/a2a/rpc"}
)
a2a_discovery = A2ADiscovery(local_card=local_card, security_context=a2a_security)
a2a_delegator = A2ADelegator(discovery=a2a_discovery)

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
    online_rl_integrator=online_rl_integrator,
    openclaw_rl_v5=openclaw_rl_v5,
    online_rl_v6=online_rl_v6,
    dialogue_online_learner_v3=dialogue_online_learner_v3,
    openclaw_rl=openclaw_rl,
    feedback_loop=feedback_loop,
    guardrail=guardrail,
    tracer=thought_chain_tracer,
    style_adapter=style_adapter,
    user_model=user_model,
    skill_versioning=skill_versioning,
    a2a_delegator=a2a_delegator,
)

subconsciousness = Subconsciousness(
    llm=llm_client,
    emotions=emotional_engine,
    memory=memory_system,
    procedural_memory=procedural_memory,
    semantic_memory=semantic_memory,
    interval=300
)

cron_scheduler = CronScheduler()
daily_report_scheduler = DailyReportScheduler(scheduler=cron_scheduler)
operations_scheduler = HermesCronSchedulerV3(db_path="operations.sqlite3")

# Schedule Subconsciousness reflection
# Default interval was 300 seconds, which is every 5 minutes
cron_scheduler.schedule("*/5 * * * *", subconsciousness.reflect, name="reflection")

# Schedule autonomous health check every hour
cron_scheduler.schedule("0 * * * *", run_health_check, name="health_check")

# Schedule quality metrics report every day at midnight
cron_scheduler.schedule("0 0 * * *", report_quality_metrics, name="quality_report", tracker=quality_tracker)

# Schedule daily AgentBench evaluation
cron_scheduler.schedule("0 0 * * *", daily_agentbench_eval, name="agentbench_eval")

task_store = TaskStore(path=os.getenv("AUTONOMY_TASKS_PATH", "./autonomy_tasks.json"))
autonomous_executor = AutonomousExecutor(
    store=task_store,
    llm=llm_client,
    skills=skill_registry,
    step_timeout=float(os.getenv("AUTONOMY_STEP_TIMEOUT", "60")),
)


canvas_server = CanvasServer(consciousness=consciousness)
a2a_server = A2AServer(planner=planner, security_context=a2a_security)
rpc_manager = SubAgentRPCManager(llm=llm_client)
cross_platform_dispatcher = CrossPlatformDispatcher()
local_first_gateway = LocalFirstGateway()
local_first_gateway.set_message_handler(consciousness.process_input)

discord_bridge = DiscordBridge(token=os.getenv("DISCORD_BOT_TOKEN", "dummy"), agent_callback=consciousness.process_input)
cross_platform_dispatcher.register_platform("discord", discord_bridge)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await context_engine.bootstrap_all({})
    asyncio.create_task(cron_scheduler.start())
    asyncio.create_task(daily_report_scheduler.start())
    asyncio.create_task(operations_scheduler.start())
    await autonomous_executor.start()
    asyncio.create_task(canvas_server.start_streaming())
    asyncio.create_task(discord_bridge.start())
    yield
    # Shutdown
    await cron_scheduler.stop()
    await daily_report_scheduler.stop()
    await operations_scheduler.stop()
    await autonomous_executor.stop()
    await canvas_server.stop_streaming()
    await discord_bridge.stop()
    memory_system.close()

app = FastAPI(title="Magda Consciousness API", lifespan=lifespan)
app.mount("/a2a", a2a_server.app)
app.include_router(get_canvas_v2_router(canvas_server, token=os.getenv('MAGDA_API_TOKEN')))

_PUBLIC_HTTP_PATHS = {"/health"}


def _configured_api_token() -> Optional[str]:
    return os.getenv("MAGDA_API_TOKEN")


def _is_authorized_header(authorization: Optional[str]) -> bool:
    token = _configured_api_token()
    if not token or not authorization:
        return False
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value:
        return False
    return hmac.compare_digest(value, token)


@app.middleware("http")
async def require_api_authentication(request: Request, call_next):
    thought_chain_tracer.start_trace()
    if request.url.path in _PUBLIC_HTTP_PATHS:
        return await call_next(request)
    if not _configured_api_token():
        return JSONResponse(status_code=503, content={"detail": "MAGDA_API_TOKEN must be configured before using the API"})
    if not _is_authorized_header(request.headers.get("Authorization")):
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing bearer token"})
    return await call_next(request)

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
    token = websocket.query_params.get("token")
    if not _configured_api_token() or not hmac.compare_digest(token or "", _configured_api_token() or ""):
        await websocket.close(code=1008)
        return
    await canvas_server.connect(websocket)
    try:
        while True:
            # We just keep the connection open, client doesn't need to send anything
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        canvas_server.disconnect(websocket)
