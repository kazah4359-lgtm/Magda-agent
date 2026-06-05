with open("magda_agent/api.py", "r") as f:
    content = f.read()

# Add import
content = content.replace(
    "from magda_agent.context.default_plugin import DefaultContextPlugin",
    "from magda_agent.context.default_plugin import DefaultContextPlugin\nfrom magda_agent.tracing.tracer import ThoughtChainTracer"
)

# Instantiate tracer
content = content.replace(
    "global_workspace = GlobalWorkspace(salience_network=salience_network)",
    "global_workspace = GlobalWorkspace(salience_network=salience_network)\nthought_chain_tracer = ThoughtChainTracer()"
)

# Pass tracer to Consciousness
content = content.replace(
    "    skill_creator=skill_creator\n)",
    "    skill_creator=skill_creator,\n    tracer=thought_chain_tracer\n)"
)

# Add route
route_str = """
class TraceResponse(BaseModel):
    trace: list

@app.get("/trace", response_model=TraceResponse)
async def get_trace():
    return TraceResponse(trace=thought_chain_tracer.get_trace())
"""
content = content + route_str

with open("magda_agent/api.py", "w") as f:
    f.write(content)
