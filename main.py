"""
Collectors League Scan API - Full AI Integration
Uses OpenAI Vision API for card identification and grading
VERSION 4.0 - Added Memorabilia & Sealed Product Support
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
APP_VERSION = "2026-01-29-memorabilia-support"
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
        "message": "Collectors League Multi-Item Assessment API"
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "cl-scan-api",
        "version": APP_VERSION,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_pokemontcg_key": bool(POKEMONTCG_API_KEY),
        "model": "gpt-4o-mini",
        "supports": ["cards", "memorabilia", "sealed_products"]
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


# ============================================================================
# COLLECTOR CARDS - EXISTING ENDPOINTS
# ============================================================================

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
        
        # Return identification data
        return JSONResponse(content={
            "name": card_data.get("name", "Unknown"),
            "series": card_data.get("series", ""),
            "year": str(card_data.get("year", "")),
            "card_number": str(card_data.get("card_number", "")),
            "type": card_data.get("type", "Other"),
            "confidence": float(card_data.get("confidence", 0.0)),
            "identify_token": identify_token
        })
        
    except Exception as e:
        print(f"Identify endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/api/verify")
async def verify(front: UploadFile = File(...), back: UploadFile = File(...)):
    """
    AI-powered card grading with both front and back images
    """
    try:
        # Read both images
        front_bytes = await front.read()
        back_bytes = await back.read()
        
        if not front_bytes or not back_bytes or len(front_bytes) < 1000 or len(back_bytes) < 1000:
            raise HTTPException(status_code=400, detail="Images are too small or empty")
        
        # Convert to base64
        front_base64 = image_to_base64(front_bytes)
        back_base64 = image_to_base64(back_bytes)
        
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        # Prepare dual-image grading prompt
        prompt = """You are a professional trading card grader. Analyze BOTH the front and back images of this card and provide a comprehensive grade assessment.

**CORNERS** - Check all 4 corners on BOTH sides:
- Are corners sharp, soft, or rounded?
- Any whitening visible (especially on back)?
- Specify WHICH corner and WHICH side (front/back)

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
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
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


# ============================================================================
# MEMORABILIA & SEALED PRODUCTS - NEW ENDPOINTS
# ============================================================================

@app.post("/api/identify-memorabilia")
async def identify_memorabilia(
    image1: UploadFile = File(...),
    image2: UploadFile = File(None),
    image3: UploadFile = File(None),
    image4: UploadFile = File(None)
):
    """
    Identify memorabilia or sealed product from 1-4 images
    Returns: item_type, description, authenticity_notes, confidence
    """
    try:
        # Read primary image (required)
        image1_bytes = await image1.read()
        if not image1_bytes or len(image1_bytes) < 1000:
            raise HTTPException(status_code=400, detail="Primary image is too small or empty")
        
        # Convert images to base64
        images_base64 = [image_to_base64(image1_bytes)]
        
        # Process optional images
        if image2:
            img2_bytes = await image2.read()
            if img2_bytes and len(img2_bytes) >= 1000:
                images_base64.append(image_to_base64(img2_bytes))
        
        if image3:
            img3_bytes = await image3.read()
            if img3_bytes and len(img3_bytes) >= 1000:
                images_base64.append(image_to_base64(img3_bytes))
        
        if image4:
            img4_bytes = await image4.read()
            if img4_bytes and len(img4_bytes) >= 1000:
                images_base64.append(image_to_base64(img4_bytes))
        
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        # Build multi-image prompt
        prompt = """Analyze these images of collectible memorabilia or sealed product. Identify what this item is and assess its key characteristics.

Provide ONLY a JSON response with these exact fields:

{
  "item_type": "Sealed Booster Box/Elite Trainer Box/Blister Pack/Graded Card/Signed Memorabilia/Display Case/Other",
  "description": "Detailed description of the item including brand, set/series, year if visible",
  "signatures": "Description of any visible signatures or autographs, or 'None visible'",
  "seal_condition": "Factory sealed/Opened/Resealed/Not applicable",
  "authenticity_notes": "Any observations about authenticity markers, holograms, serial numbers, packaging quality, or red flags",
  "notable_features": "Any special features, variants, errors, or unique aspects",
  "confidence": 0.0-1.0
}

Look for:
- Product type and brand
- Set/series name and logos
- Seal integrity (shrink wrap condition, factory seals)
- Authentication markers (holograms, serial numbers, official stamps)
- Signs of tampering or resealing
- Signatures or autographs (check if present)
- Overall condition and packaging quality
- Any red flags or concerning aspects

If you cannot identify with confidence, set confidence to 0.0 and provide your best assessment in description.
Respond ONLY with valid JSON, no other text."""
        
        # Build content array for API call
        content = [{"type": "text", "text": prompt}]
        
        # Add all images with labels
        image_labels = ["Primary Image", "Additional View", "Detail/Close-up", "Alternative Angle"]
        for idx, img_b64 in enumerate(images_base64):
            if idx > 0:
                content.append({
                    "type": "text",
                    "text": f"--- {image_labels[idx]} ---"
                })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "high"
                }
            })
        
        # Call OpenAI Vision API
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                print(f"OpenAI API Error: {response.status_code}")
                return JSONResponse(content={
                    "item_type": "Unknown",
                    "description": "Could not identify",
                    "signatures": "Unable to assess",
                    "seal_condition": "Unable to assess",
                    "authenticity_notes": "AI identification failed",
                    "notable_features": "",
                    "confidence": 0.0,
                    "identify_token": f"idt_{secrets.token_urlsafe(12)}",
                    "error": "AI identification failed"
                })
            
            data = response.json()
            
            if "choices" not in data or len(data["choices"]) == 0:
                return JSONResponse(content={
                    "item_type": "Unknown",
                    "description": "No response from AI",
                    "signatures": "Unable to assess",
                    "seal_condition": "Unable to assess",
                    "authenticity_notes": "",
                    "notable_features": "",
                    "confidence": 0.0,
                    "identify_token": f"idt_{secrets.token_urlsafe(12)}"
                })
            
            content_text = data["choices"][0]["message"]["content"].strip()
            
            # Parse JSON
            if content_text.startswith("```json"):
                content_text = content_text[7:]
            if content_text.startswith("```"):
                content_text = content_text[3:]
            if content_text.endswith("```"):
                content_text = content_text[:-3]
            content_text = content_text.strip()
            
            try:
                item_data = json.loads(content_text)
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                return JSONResponse(content={
                    "item_type": "Parse error",
                    "description": "Could not parse AI response",
                    "signatures": "Unable to assess",
                    "seal_condition": "Unable to assess",
                    "authenticity_notes": "",
                    "notable_features": "",
                    "confidence": 0.0,
                    "identify_token": f"idt_{secrets.token_urlsafe(12)}"
                })
            
            # Generate identify token
            identify_token = f"idt_{secrets.token_urlsafe(12)}"
            
            return JSONResponse(content={
                "item_type": item_data.get("item_type", "Unknown"),
                "description": item_data.get("description", ""),
                "signatures": item_data.get("signatures", "None visible"),
                "seal_condition": item_data.get("seal_condition", "Not applicable"),
                "authenticity_notes": item_data.get("authenticity_notes", ""),
                "notable_features": item_data.get("notable_features", ""),
                "confidence": float(item_data.get("confidence", 0.0)),
                "identify_token": identify_token
            })
        
    except Exception as e:
        print(f"Identify memorabilia endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/api/assess-memorabilia")
async def assess_memorabilia(
    image1: UploadFile = File(...),
    image2: UploadFile = File(None),
    image3: UploadFile = File(None),
    image4: UploadFile = File(None)
):
    """
    Assess condition and authenticity of memorabilia/sealed products from 1-4 images
    """
    try:
        # Read primary image (required)
        image1_bytes = await image1.read()
        if not image1_bytes or len(image1_bytes) < 1000:
            raise HTTPException(status_code=400, detail="Primary image is too small or empty")
        
        # Convert images to base64
        images_base64 = [image_to_base64(image1_bytes)]
        
        # Process optional images
        if image2:
            img2_bytes = await image2.read()
            if img2_bytes and len(img2_bytes) >= 1000:
                images_base64.append(image_to_base64(img2_bytes))
        
        if image3:
            img3_bytes = await image3.read()
            if img3_bytes and len(img3_bytes) >= 1000:
                images_base64.append(image_to_base64(img3_bytes))
        
        if image4:
            img4_bytes = await image4.read()
            if img4_bytes and len(img4_bytes) >= 1000:
                images_base64.append(image_to_base64(img4_bytes))
        
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        # Build comprehensive assessment prompt
        prompt = """You are a professional memorabilia and sealed product authenticator. Analyze these images comprehensively.

For SEALED PRODUCTS, assess:
- Seal integrity (factory seal vs resealed, shrink wrap condition)
- Packaging condition (box corners, crushing, dents, shelf wear)
- Authentication markers (holograms, serial numbers, factory stamps)
- Signs of tampering or opening
- Overall preservation

For SIGNED MEMORABILIA, assess:
- Signature authenticity indicators
- Signature placement and quality
- Item condition
- Certificate of Authenticity if visible
- Any authentication markers

For ALL ITEMS, evaluate:
- Overall condition grade
- Notable defects or issues
- Authenticity confidence
- Value-impacting factors

Provide ONLY a JSON response:

{
  "overall_assessment": "Brief 2-3 sentence summary of item condition and authenticity",
  "condition_grade": "Mint/Near Mint/Excellent/Very Good/Good/Fair/Poor",
  "seal_integrity": {
    "grade": "Factory Sealed/Intact/Compromised/Opened/Not Applicable",
    "notes": "Detailed assessment of seals, shrink wrap, factory seals. Note any signs of tampering or resealing."
  },
  "packaging_condition": {
    "grade": "Mint/Near Mint/Excellent/Very Good/Good/Fair/Poor",
    "notes": "Assess box/packaging: corners, edges, crushing, dents, shelf wear, printing quality"
  },
  "authenticity_assessment": {
    "grade": "Highly Confident/Likely Authentic/Uncertain/Concerns Present/Likely Counterfeit",
    "notes": "Check authentication markers, holograms, serial numbers, packaging quality, known counterfeit indicators"
  },
  "signature_assessment": {
    "present": true/false,
    "grade": "Authentic/Likely Authentic/Uncertain/Concerns/Not Applicable",
    "notes": "If signatures present: assess authenticity markers, placement, ink quality, any COA visible"
  },
  "defects": ["List specific issues: 'Box corner crushing top-right', 'Shrink wrap tear left side', 'Signature smudged', etc. Empty array if none."],
  "value_factors": ["List factors affecting value: positive or negative"],
  "confidence": 0.0-1.0
}

GRADING SCALE:
- Mint: Perfect condition, factory fresh
- Near Mint: Minimal wear, excellent overall
- Excellent: Minor wear, still very presentable
- Very Good: Noticeable wear but intact
- Good: Significant wear, structural integrity maintained
- Fair: Heavy wear, may have damage
- Poor: Severe damage, barely collectible

Be thorough, specific, and honest. Note both positives and concerns.
Respond ONLY with valid JSON, no other text."""
        
        # Build content array
        content = [{"type": "text", "text": prompt}]
        
        image_labels = ["Primary Image", "Additional View", "Detail/Close-up", "Alternative Angle"]
        for idx, img_b64 in enumerate(images_base64):
            if idx > 0:
                content.append({
                    "type": "text",
                    "text": f"--- {image_labels[idx]} ---"
                })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "high"
                }
            })
        
        # Call OpenAI
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                print(f"OpenAI API Error: {response.status_code}")
                return JSONResponse(content={
                    "overall_assessment": "Assessment failed",
                    "condition_grade": "Unable to assess",
                    "seal_integrity": {"grade": "N/A", "notes": "Assessment failed"},
                    "packaging_condition": {"grade": "N/A", "notes": "Assessment failed"},
                    "authenticity_assessment": {"grade": "N/A", "notes": "Assessment failed"},
                    "signature_assessment": {"present": False, "grade": "N/A", "notes": "Assessment failed"},
                    "defects": [],
                    "value_factors": [],
                    "confidence": 0.0,
                    "error": "AI assessment failed"
                })
            
            data = response.json()
            
            if "choices" not in data or len(data["choices"]) == 0:
                return JSONResponse(content={
                    "overall_assessment": "No response",
                    "condition_grade": "N/A",
                    "seal_integrity": {"grade": "N/A", "notes": "No AI response"},
                    "packaging_condition": {"grade": "N/A", "notes": "No AI response"},
                    "authenticity_assessment": {"grade": "N/A", "notes": "No AI response"},
                    "signature_assessment": {"present": False, "grade": "N/A", "notes": "No AI response"},
                    "defects": [],
                    "value_factors": [],
                    "confidence": 0.0
                })
            
            content_text = data["choices"][0]["message"]["content"].strip()
            
            # Parse JSON
            if content_text.startswith("```json"):
                content_text = content_text[7:]
            if content_text.startswith("```"):
                content_text = content_text[3:]
            if content_text.endswith("```"):
                content_text = content_text[:-3]
            content_text = content_text.strip()
            
            try:
                assessment_data = json.loads(content_text)
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                print(f"Content: {content_text[:500]}")
                return JSONResponse(content={
                    "overall_assessment": "Parse error",
                    "condition_grade": "N/A",
                    "seal_integrity": {"grade": "N/A", "notes": "Could not parse response"},
                    "packaging_condition": {"grade": "N/A", "notes": "Could not parse response"},
                    "authenticity_assessment": {"grade": "N/A", "notes": "Could not parse response"},
                    "signature_assessment": {"present": False, "grade": "N/A", "notes": "Could not parse response"},
                    "defects": [],
                    "value_factors": [],
                    "confidence": 0.0
                })
            
            # Return assessment
            return JSONResponse(content={
                "overall_assessment": assessment_data.get("overall_assessment", ""),
                "condition_grade": assessment_data.get("condition_grade", "N/A"),
                "seal_integrity": assessment_data.get("seal_integrity", {"grade": "N/A", "notes": ""}),
                "packaging_condition": assessment_data.get("packaging_condition", {"grade": "N/A", "notes": ""}),
                "authenticity_assessment": assessment_data.get("authenticity_assessment", {"grade": "N/A", "notes": ""}),
                "signature_assessment": assessment_data.get("signature_assessment", {"present": False, "grade": "N/A", "notes": ""}),
                "defects": assessment_data.get("defects", []),
                "value_factors": assessment_data.get("value_factors", []),
                "confidence": float(assessment_data.get("confidence", 0.0))
            })
        
    except Exception as e:
        print(f"Assess memorabilia endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
