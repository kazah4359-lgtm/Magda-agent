from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Magda Consciousness Microservice")

class ConsciousnessRequest(BaseModel):
    input_text: str

class ConsciousnessResponse(BaseModel):
    thoughts: str

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/process", response_model=ConsciousnessResponse)
async def process_consciousness(request: ConsciousnessRequest):
    # Dummy processing for now
    thoughts = f"Thinking about: {request.input_text}"
    return ConsciousnessResponse(thoughts=thoughts)
