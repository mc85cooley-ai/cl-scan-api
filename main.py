"""
Collectors League Scan API - Full AI Integration
Uses OpenAI Vision API for card identification and grading
VERSION 3.2 - Enhanced AI Detail + Texture Awareness
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
APP_VERSION = "2026-01-28-enhanced-detail"
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
        "message": "Collectors League Card Grading API - Enhanced Detail"
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
  "type": "Pokemon/Magic/YuGiOh/Sports/OnePiece/Other",
  "confidence": 0.0-1.0
}

Be specific with the card name. Include any variants like "ex", "V", "VMAX", "GX", "holo", "reverse holo", "first edition", "special illustration rare (SIR)", etc.
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
    
    ENHANCED VERSION: More detailed analysis + texture awareness
    """
    try:
        # Read front image (primary for grading)
        front_bytes = await front.read()
        
        if not front_bytes or len(front_bytes) < 1000:
            raise HTTPException(status_code=400, detail="Front image is too small or empty")
        
        # Convert to base64
        front_base64 = image_to_base64(front_bytes)
        
        # ENHANCED PROMPT - More detail + texture awareness
        prompt = """You are a professional card grader with expertise in modern trading cards. Analyze this card's condition in detail.

🚨 CRITICAL - MODERN CARD TEXTURES:
Many modern cards have INTENTIONAL textured finishes as design features:
- Holographic/prismatic rainbow patterns (normal on holos)
- Raised/embossed text and borders (normal on textured cards)
- Sparkle/glitter effects (normal on special editions)
- Metallic foil finishes (normal on ultra rares)
- Intentional textured backgrounds (normal on modern prints)

DO NOT mark these as damage or defects. These are FEATURES, not flaws.

ONLY mark ACTUAL DAMAGE:
- Scratches that break through the surface coating
- Physical dents, creases, or bends in the card
- Edge whitening or chipping from wear
- Corner fraying or whitening
- Print lines (factory defects)
- Stains, discoloration, or dirt
- Actual surface scuffs from handling

Be VERY specific in your analysis. For each category, describe exactly what you observe.

Provide ONLY a JSON response with these exact fields:

{
  "pregrade": "estimated PSA-style grade 1-10",
  "grade_corners": {
    "grade": "Mint/Near Mint/Excellent/Good/Poor",
    "notes": "Describe each corner specifically: Top-left: [condition]. Top-right: [condition]. Bottom-left: [condition]. Bottom-right: [condition]. Look for whitening, fraying, soft corners, or damage."
  },
  "grade_edges": {
    "grade": "Mint/Near Mint/Excellent/Good/Poor",
    "notes": "Examine all 4 edges. Describe: Are edges sharp and clean? Any whitening? Chipping? Roughness? Wear from handling? Be specific about which edges have issues."
  },
  "grade_surface": {
    "grade": "Mint/Near Mint/Excellent/Good/Poor",
    "notes": "IGNORE intentional textures/holo patterns. Look for: Actual scratches in coating? Scuffs from handling? Print lines? Stains or dirt? Creases? Surface wear? Be detailed about location and severity."
  },
  "grade_centering": {
    "grade": "60/40 or better / 70/30 / 80/20 / Off-center",
    "notes": "Measure border widths. Left vs Right borders: [X%]/[Y%]. Top vs Bottom borders: [X%]/[Y%]. Is text/image centered in frame? Specify which direction it's off if not centered."
  },
  "confidence": 0.0-1.0,
  "defects": ["List each specific defect with location, e.g., 'Top-right corner whitening', 'Horizontal scratch on center-left surface', 'Left edge minor whitening'. If no defects, empty array."]
}

GRADING SCALE:
- Mint (9-10): Perfect or near-perfect condition, sharp corners, clean edges, flawless surface
- Near Mint (7-8): Minor wear, slight corner softness, very minor edge wear
- Excellent (5-6): Noticeable wear, corner whitening, edge whitening, minor surface issues
- Good (3-4): Significant wear, damaged corners, edge damage, surface scratches
- Poor (1-2): Heavy damage, creases, major corner/edge damage

Be CONSERVATIVE but FAIR. Don't penalize intentional card features. Be DETAILED in your notes.

Respond ONLY with valid JSON, no other text."""
        
        # Call OpenAI Vision with MORE tokens for detailed response
        result = await call_openai_vision(front_base64, prompt, max_tokens=1500)
        
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
