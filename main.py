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
    
    ENHANCED VERSION: Analyzes BOTH front and back + texture awareness
    """
    try:
        # Read BOTH images
        front_bytes = await front.read()
        back_bytes = await back.read()
        
        if not front_bytes or len(front_bytes) < 1000:
            raise HTTPException(status_code=400, detail="Front image is too small or empty")
        
        if not back_bytes or len(back_bytes) < 1000:
            raise HTTPException(status_code=400, detail="Back image is too small or empty")
        
        # Convert both to base64
        front_base64 = image_to_base64(front_bytes)
        back_base64 = image_to_base64(back_bytes)
        
        # ENHANCED PROMPT - Analyzes BOTH sides
        prompt = """You are a professional card grader with expertise in modern trading cards. Analyze BOTH images (front and back) of this card to provide a comprehensive condition assessment.

🚨 CRITICAL - MODERN CARD TEXTURES:
Many modern cards have INTENTIONAL textured finishes as design features:
- Holographic/prismatic rainbow patterns (normal on holos)
- Raised/embossed text and borders (normal on textured cards)
- Sparkle/glitter effects (normal on special editions)
- Metallic foil finishes (normal on ultra rares)
- Intentional textured backgrounds (normal on modern prints)
- Textured energy symbols or card backs (normal)

DO NOT mark these as damage or defects. These are FEATURES, not flaws.

ONLY mark ACTUAL DAMAGE:
- Scratches that break through the surface coating
- Physical dents, creases, or bends in the card
- Edge whitening or chipping from wear
- Corner fraying or whitening
- Print lines (factory defects)
- Stains, discoloration, or dirt
- Actual surface scuffs from handling
- Silvering (edge wear showing white core)

📋 ASSESSMENT PROCESS:
Examine BOTH front and back images carefully:

**CORNERS** - Check all 4 corners on BOTH sides:
- Are corners sharp and well-defined?
- Any whitening or fraying visible?
- Any soft or rounded corners?
- Compare front vs back corner condition

**EDGES** - Examine all 4 edges on BOTH sides:
- Are edges clean and sharp?
- Any whitening (especially visible on back)?
- Any chipping or roughness?
- Any silvering (white card core showing)?

**SURFACE** - Analyze front AND back surfaces:
- Front: Ignore holo patterns. Look for scratches, scuffs, print lines
- Back: Check for scratches, scuffs, wear marks
- Any creases visible on either side?
- Any stains or discoloration?

**CENTERING** - Primarily check the front:
- Measure border widths around the card image
- Is the artwork/text centered properly?

Be VERY specific in your analysis. Mention which side (front/back) each issue appears on.

Provide ONLY a JSON response with these exact fields:

{
  "pregrade": "estimated PSA-style grade 1-10",
  "grade_corners": {
    "grade": "Mint/Near Mint/Excellent/Good/Poor",
    "notes": "Describe each corner on BOTH sides. Example: 'Front: Top-left sharp, top-right minor softness. Back: All corners sharp with slight whitening on bottom-right.' Be specific about which side and which corner."
  },
  "grade_edges": {
    "grade": "Mint/Near Mint/Excellent/Good/Poor",
    "notes": "Examine all 4 edges on BOTH front and back. Example: 'Front edges clean. Back shows minor whitening on left and top edges, typical of handling.' Specify front vs back."
  },
  "grade_surface": {
    "grade": "Mint/Near Mint/Excellent/Good/Poor",
    "notes": "IGNORE intentional textures/holo patterns. Analyze BOTH surfaces. Example: 'Front surface: Clean with no visible scratches. Holo pattern is intentional. Back surface: Minor scuff mark in center-left area, otherwise clean.' Be detailed about front vs back."
  },
  "grade_centering": {
    "grade": "60/40 or better / 70/30 / 80/20 / Off-center",
    "notes": "Measure front borders. Left vs Right: [X%]/[Y%]. Top vs Bottom: [X%]/[Y%]. Example: 'Front centering is 65/35 left-to-right, 60/40 top-to-bottom. Slightly off-center but acceptable.'"
  },
  "confidence": 0.0-1.0,
  "defects": ["List each specific defect with SIDE and location. Examples: 'Front: Top-right corner whitening', 'Back: Horizontal scratch center-left', 'Back: Left edge minor silvering', 'Both sides: Bottom-left corner soft'. If no defects, empty array."]
}

GRADING SCALE (PSA-style):
- 10 (Gem Mint): Flawless card, perfect centering, sharp corners, pristine surfaces
- 9 (Mint): Near-perfect, may have tiny imperfections, excellent centering
- 8 (Near Mint-Mint): Very minor wear, slight corner touches, minor centering issues
- 7 (Near Mint): Minor wear visible, slight corner whitening, good overall appearance
- 6 (Excellent-Mint): Noticeable wear, minor edge/corner issues, slightly off-center
- 5 (Excellent): Moderate wear, edge whitening, corner rounding, centering issues
- 4 (Very Good-Excellent): Significant wear, damaged corners/edges, surface issues
- 3 (Very Good): Heavy wear, major corner/edge damage, creases
- 2 (Good): Severe damage, creases, major surface issues
- 1 (Poor): Extremely damaged, multiple creases, barely collectible

IMPORTANT:
- Be CONSERVATIVE but FAIR
- Don't penalize intentional card features
- Consider BOTH front and back in your assessment
- Modern cards often have more texture - distinguish texture from damage
- Be DETAILED in your notes - specify which side each issue appears on
- Defects should always indicate FRONT or BACK

Respond ONLY with valid JSON, no other text."""
        
        # Call OpenAI Vision with BOTH images
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
                                "url": f"data:image/jpeg;base64,{front_base64}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": "FRONT IMAGE ABOVE ☝️ | BACK IMAGE BELOW 👇"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{back_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000,  # Increased for detailed dual-image analysis
            "temperature": 0.1
        }
        
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:  # Longer timeout for 2 images
                response = await client.post(url, headers=headers, json=payload)
                
                if response.status_code != 200:
                    print(f"OpenAI API Error: {response.status_code}")
                    print(f"Response: {response.text[:500]}")
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
                
                data = response.json()
                
                if "choices" not in data or len(data["choices"]) == 0:
                    return JSONResponse(content={
                        "pregrade": "No response",
                        "grade_corners": {"grade": "N/A", "notes": "No AI response"},
                        "grade_edges": {"grade": "N/A", "notes": "No AI response"},
                        "grade_surface": {"grade": "N/A", "notes": "No AI response"},
                        "grade_centering": {"grade": "N/A", "notes": "No AI response"},
                        "confidence": 0.0,
                        "defects": []
                    })
                
                content = data["choices"][0]["message"]["content"]
                
        except Exception as e:
            print(f"OpenAI API Exception: {str(e)}")
            return JSONResponse(content={
                "pregrade": "Error",
                "grade_corners": {"grade": "N/A", "notes": f"Error: {str(e)[:100]}"},
                "grade_edges": {"grade": "N/A", "notes": "Error"},
                "grade_surface": {"grade": "N/A", "notes": "Error"},
                "grade_centering": {"grade": "N/A", "notes": "Error"},
                "confidence": 0.0,
                "defects": []
            })
        
        # Parse JSON from response
        content = content.strip()
        
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
            print(f"Content: {content[:500]}")
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
