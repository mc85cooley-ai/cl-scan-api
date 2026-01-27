"""
Collectors League Scan API - Full AI Integration
Uses OpenAI Vision API for card identification and grading
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import base64
import os
import json
import secrets
import httpx

app = FastAPI(title="Collectors League Scan API")

# CORS for WordPress
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
APP_VERSION = "2026-01-28-ai-vision"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
POKEMONTCG_API_KEY = os.getenv("POKEMONTCG_API_KEY", "").strip()

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set!")


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "cl-scan-api",
        "version": APP_VERSION,
        "message": "Collectors League Card Grading API"
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "cl-scan-api",
        "version": APP_VERSION,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_pokemontcg_key": bool(POKEMONTCG_API_KEY),
        "model": "gpt-4o-mini"
    }


def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode('utf-8')


async def call_openai_vision(image_base64: str, prompt: str, max_tokens: int = 800) -> dict:
    """Call OpenAI Vision API"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                print(f"OpenAI API Error: {response.status_code}")
                print(f"Response: {response.text[:500]}")
                return {
                    "error": True,
                    "status": response.status_code,
                    "message": response.text[:200]
                }
            
            data = response.json()
            
            if "choices" not in data or len(data["choices"]) == 0:
                return {"error": True, "message": "No response from OpenAI"}
            
            content = data["choices"][0]["message"]["content"]
            return {"error": False, "content": content}
            
    except Exception as e:
        print(f"OpenAI API Exception: {str(e)}")
        return {"error": True, "message": str(e)}


@app.post("/api/identify")
async def identify(front: UploadFile = File(...)):
    """
    Identify a trading card from its front image using AI vision
    Returns: name, series, year, card_number, type, confidence
    """
    try:
        # Read image
        image_bytes = await front.read()
        
        if not image_bytes or len(image_bytes) < 1000:
            raise HTTPException(status_code=400, detail="Image is too small or empty")
        
        # Convert to base64
        image_base64 = image_to_base64(image_bytes)
        
        # Prepare prompt for identification
        prompt = """Identify this trading card. Analyze the image and provide ONLY a JSON response with these exact fields:

{
  "name": "exact card name including variants (ex, V, VMAX, holo, first edition, etc.)",
  "series": "set or series name",
  "year": "release year (4 digits)",
  "card_number": "card number if visible",
  "type": "Pokemon/Magic/YuGiOh/Sports/Other",
  "confidence": 0.0-1.0
}

Be specific with the card name. Include any variants like "ex", "V", "VMAX", "GX", "holo", "reverse holo", "first edition", etc.
If you cannot identify with confidence, set confidence to 0.0 and name to "Unknown".
Respond ONLY with valid JSON, no other text."""
        
        # Call OpenAI Vision
        result = await call_openai_vision(image_base64, prompt, max_tokens=500)
        
        if result.get("error"):
            print(f"Vision API error: {result.get('message')}")
            # Return fallback response
            return JSONResponse(content={
                "name": "Could not identify",
                "series": "Unknown",
                "year": "Unknown",
                "card_number": "",
                "type": "Other",
                "confidence": 0.0,
                "identify_token": f"idt_{secrets.token_urlsafe(12)}",
                "error": "AI identification failed"
            })
        
        # Parse JSON from response
        content = result["content"].strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            card_data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Content: {content}")
            # Return fallback
            return JSONResponse(content={
                "name": "Parse error",
                "series": "Unknown",
                "year": "Unknown",
                "card_number": "",
                "type": "Other",
                "confidence": 0.0,
                "identify_token": f"idt_{secrets.token_urlsafe(12)}",
                "error": "Could not parse AI response"
            })
        
        # Generate identify token
        identify_token = f"idt_{secrets.token_urlsafe(12)}"
        
        # Return identified card data
        return JSONResponse(content={
            "name": card_data.get("name", "Unknown"),
            "series": card_data.get("series", "Unknown"),
            "year": str(card_data.get("year", "Unknown")),
            "card_number": str(card_data.get("card_number", "")),
            "type": card_data.get("type", "Other"),
            "confidence": float(card_data.get("confidence", 0.0)),
            "identify_token": identify_token
        })
        
    except Exception as e:
        print(f"Identify endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/api/verify")
async def verify(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    angle: Optional[UploadFile] = File(None),
    identify_token: Optional[str] = Form(None)
):
    """
    Assess card condition from front and back images using AI vision
    Returns: grade, corners, edges, surface, centering, defects
    """
    try:
        # Read front image (primary for grading)
        front_bytes = await front.read()
        
        if not front_bytes or len(front_bytes) < 1000:
            raise HTTPException(status_code=400, detail="Front image is too small or empty")
        
        # Convert to base64
        front_base64 = image_to_base64(front_bytes)
        
        # Prepare prompt for condition assessment
        prompt = """You are a professional card grader. Analyze this trading card's condition and provide ONLY a JSON response with these exact fields:

{
  "pregrade": "estimated PSA-style grade 1-10",
  "grade_corners": {
    "grade": "Mint/Near Mint/Excellent/Good/Poor",
    "notes": "specific observations about corner condition"
  },
  "grade_edges": {
    "grade": "Mint/Near Mint/Excellent/Good/Poor",
    "notes": "specific observations about edge condition"
  },
  "grade_surface": {
    "grade": "Mint/Near Mint/Excellent/Good/Poor",
    "notes": "specific observations about surface condition"
  },
  "grade_centering": {
    "grade": "60/40 or better / 70/30 / 80/20 / Off-center",
    "notes": "specific observations about centering"
  },
  "confidence": 0.0-1.0,
  "defects": ["list specific defects found, or empty array if none"]
}

prompt = f"""You are a professional card grader...

CRITICAL INSTRUCTION FOR MODERN CARDS:
Many modern trading cards have INTENTIONAL textured surfaces as part of their design:
- Holographic patterns and effects
- Raised/textured printing
- Sparkle or glitter effects
- Designed texture finishes

DO NOT mark these intentional design features as "surface scratches" or defects.

ONLY mark actual damage:
- Scratches that break through the surface
- Dents, creases, or bends
- Wear marks from handling
- Actual manufacturing defects

If you see texture, first determine: Is this part of the card's design? If yes, DO NOT list as defect.

Be conservative in grading. Look for:
- Corner wear or whitening
- Edge chipping or roughness  
- Surface scratches, print lines, or stains
- Centering issues (uneven borders)

Respond ONLY with valid JSON, no other text."""
        
        # Call OpenAI Vision
        result = await call_openai_vision(front_base64, prompt, max_tokens=1000)
        
        if result.get("error"):
            print(f"Vision API error: {result.get('message')}")
            # Return fallback response
            return JSONResponse(content={
                "pregrade": "Unable to assess",
                "grade_corners": {"grade": "N/A", "notes": "Assessment failed"},
                "grade_edges": {"grade": "N/A", "notes": "Assessment failed"},
                "grade_surface": {"grade": "N/A", "notes": "Assessment failed"},
                "grade_centering": {"grade": "N/A", "notes": "Assessment failed"},
                "confidence": 0.0,
                "defects": [],
                "error": "AI grading failed"
            })
        
        # Parse JSON from response
        content = result["content"].strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            grade_data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Content: {content}")
            # Return fallback
            return JSONResponse(content={
                "pregrade": "Parse error",
                "grade_corners": {"grade": "N/A", "notes": "Could not parse response"},
                "grade_edges": {"grade": "N/A", "notes": "Could not parse response"},
                "grade_surface": {"grade": "N/A", "notes": "Could not parse response"},
                "grade_centering": {"grade": "N/A", "notes": "Could not parse response"},
                "confidence": 0.0,
                "defects": []
            })
        
        # Return grading data
        return JSONResponse(content={
            "pregrade": grade_data.get("pregrade", "N/A"),
            "grade_corners": grade_data.get("grade_corners", {"grade": "N/A", "notes": ""}),
            "grade_edges": grade_data.get("grade_edges", {"grade": "N/A", "notes": ""}),
            "grade_surface": grade_data.get("grade_surface", {"grade": "N/A", "notes": ""}),
            "grade_centering": grade_data.get("grade_centering", {"grade": "N/A", "notes": ""}),
            "confidence": float(grade_data.get("confidence", 0.0)),
            "defects": grade_data.get("defects", [])
        })
        
    except Exception as e:
        print(f"Verify endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
