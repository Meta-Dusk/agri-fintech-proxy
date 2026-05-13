import os
from typing import Any, cast
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cerebras.cloud.sdk import Cerebras

app = FastAPI(title="Agri-FinTech Proxy")

client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

# --- DATA MODELS ---
class ColorData(BaseModel):
    purple: float
    brown: float
    yellow: float
    green_or_other: float

class FastAnalysisPayload(BaseModel):
    status: str
    local_diagnosis: str
    confidence: int
    color_data: ColorData

class VisionAnalysisPayload(BaseModel):
    image_base64: str

# --- HELPER FUNCTION ---
def extract_message_content(response: Any) -> str:
    """
    Safely extracts the message content from the SDK response 
    to satisfy strict type checkers like Pylance.
    """
    try:
        response_obj = cast(Any, response)
        content = response_obj.choices[0].message.content
        
        if content is None:
            raise ValueError("Received null content from the model.")
            
        return str(content)
    except (IndexError, AttributeError) as e:
        raise ValueError(f"Unexpected response structure: {str(e)}")


# --- ENDPOINTS ---
@app.get("/")
async def health_check():
    """
    Root endpoint for UptimeRobot to ping and keep the Render instance awake.
    """
    return {"status": "awake", "service": "Agri-FinTech Proxy"}

@app.post("/analyze-fast")
async def analyze_fast(payload: FastAnalysisPayload):
    """
    Handles High-Confidence local scans and Fast Triage choices.
    Converts raw color data into a Financial Risk & Crop Health Score.
    """
    system_prompt = (
        "You are an expert agricultural-fintech risk assessor in the Philippines. "
        "Analyze the provided local diagnosis and crop color distribution data. "
        "Output a concise evaluation containing: 1) A 'Crop Health Score' (0-100), "
        "2) A financial yield risk prediction, and 3) A micro-loan/fertilizer recommendation "
        "to satisfy SDG 1 & 8 goals. Keep it under 4 sentences."
    )
    
    user_prompt = (
        f"Diagnosis: {payload.local_diagnosis}. "
        f"Confidence: {payload.confidence}%. Data: "
        f"{payload.color_data.model_dump_json()}"
    )

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3.1-8b",
            stream=False, # Explicitly disable streaming for the type checker
            max_completion_tokens=150,
            temperature=0.2
        )
        
        content = extract_message_content(response)
        
        return {
            "success": True,
            "financial_assessment": content,
            "sdg_alignment": ["SDG 1", "SDG 8", "SDG 10"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-vision")
async def analyze_vision(payload: VisionAnalysisPayload):
    """
    Handles the Deep Vision Verification choice.
    Chains a 3rd-party vision API with Cerebras financial synthesis.
    """
    # TODO: Implement httpx call to your chosen 3rd-party vision API here
    # vision_result = await fetch_vision_data(payload.image_base64)
    
    # Mocking the vision API return for now
    mock_vision_data = "Vision API confirms leaf presence. High marginal necrosis detected."
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial risk assessor evaluating 3rd-party vision data for crop loans..."
                },
                {"role": "user", "content": mock_vision_data}
            ],
            model="llama3.3-70b",
            stream=False,
        )
        
        content = extract_message_content(response)
        
        return {
            "success": True,
            "financial_assessment": content,
            "sdg_alignment": ["SDG 1", "SDG 8", "SDG 10"],
            "vision_verified": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))