"""
The Collectors League Australia — Scan API
Full AI Integration + Click-only Market Context

VERSION 5.0.0 (2026-01-30)
- Hardened JSON parsing + unified OpenAI call helper
- Added click-only /api/market-context (informational market context; no ROI language)
- Kept legacy /api/market-intelligence as an alias (deprecated)
- Improved eBay sold scraping: recency weighting, outlier control, auction/BIN neutrality
- Optional TCGPlayer anchor (raw only) if POKEMONTCG_API_KEY provided
- Grade probability curve + expected value (condition/grade-confidence aware) when market samples exist

NOTES
- This service provides professional inspection outputs and informational market context only.
- Do NOT present outputs as financial advice, guaranteed outcomes, or future price predictions.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List, Tuple
import base64
import os
import json
import secrets
import re
from datetime import datetime
from statistics import mean, median

import httpx
from bs4 import BeautifulSoup

# ==============================
# App & Config
# ==============================
app = FastAPI(title="Collectors League Scan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # WordPress
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_VERSION = os.getenv("CL_SCAN_VERSION", "2026-01-30-v5.0.0")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
POKEMONTCG_API_KEY = os.getenv("POKEMONTCG_API_KEY", "").strip()

# eBay region: use AU by default for AUD-ish results
EBAY_DOMAIN = os.getenv("EBAY_DOMAIN", "www.ebay.com.au").strip() or "www.ebay.com.au"
ENABLE_130POINT = os.getenv("ENABLE_130POINT", "0").strip() == "1"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set!")


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "cl-scan-api",
        "version": APP_VERSION,
        "message": "The Collectors League Australia — Multi-Item Assessment API",
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "cl-scan-api",
        "version": APP_VERSION,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_pokemontcg_key": bool(POKEMONTCG_API_KEY),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "supports": ["cards", "memorabilia", "sealed_products", "market_context_click_only"],
        "ebay_domain": EBAY_DOMAIN,
    }


# ==============================
# Helpers
# ==============================
def _b64(img: bytes) -> str:
    return base64.b64encode(img).decode("utf-8")


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return (s or "").strip()


def _parse_json_or_none(s: str) -> Optional[Dict[str, Any]]:
    s = _strip_code_fences(s)
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Attempt to extract the first JSON object if extra text leaked in
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


async def _openai_chat(messages: List[Dict[str, Any]], max_tokens: int = 1200, temperature: float = 0.1) -> Dict[str, Any]:
    """
    Unified OpenAI Chat Completions call for text+image content payloads.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    url = "https://api.openai.com/v1/chat/completions"
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code != 200:
                return {"error": True, "status": r.status_code, "message": r.text[:400]}
            data = r.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return {"error": False, "content": content}
    except Exception as e:
        return {"error": True, "status": 0, "message": str(e)}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _grade_bucket(predicted_grade: str) -> Optional[int]:
    m = re.search(r"(\d{1,2})", (predicted_grade or "").strip())
    if not m:
        return None
    g = int(m.group(1))
    if g < 1 or g > 10:
        return None
    return g


def _grade_distribution(predicted_grade: int, confidence: float) -> Dict[str, float]:
    """
    Returns probability distribution over grades {10,9,8} (and optionally 7) around predicted_grade.
    Keeps output stable and conservative.
    """
    c = _clamp(confidence, 0.05, 0.95)
    # Base weight for predicted grade increases with confidence
    p_pred = 0.45 + 0.50 * c  # 0.475..0.925
    remainder = 1.0 - p_pred

    # Spread remainder to adjacent grades, skewing downward (conservative)
    if predicted_grade >= 10:
        dist = {"10": p_pred, "9": remainder * 0.75, "8": remainder * 0.25}
    elif predicted_grade == 9:
        dist = {"10": remainder * 0.25, "9": p_pred, "8": remainder * 0.75}
    else:  # predicted 8 or lower: we only model 10/9/8 for pricing context
        dist = {"10": remainder * 0.10, "9": remainder * 0.30, "8": p_pred + remainder * 0.60}

    # Normalize (guard rounding)
    total = sum(dist.values())
    if total <= 0:
        return {"10": 0.0, "9": 0.0, "8": 0.0}
    for k in dist:
        dist[k] = round(dist[k] / total, 4)
    return dist


# ==============================
# Card Endpoints
# ==============================
@app.post("/api/identify")
async def identify(front: UploadFile = File(...)):
    """
    Identify a trading card from its front image using AI vision
    Returns: name, series, year, card_number, type, confidence
    """
    image_bytes = await front.read()
    if not image_bytes or len(image_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Image is too small or empty")

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

    msg = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(image_bytes)}", "detail": "high"}},
        ],
    }]

    result = await _openai_chat(msg, max_tokens=700, temperature=0.1)
    if result.get("error"):
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

    data = _parse_json_or_none(result.get("content", "")) or {}

    return JSONResponse(content={
        "name": data.get("name", "Unknown"),
        "series": data.get("series", ""),
        "year": str(data.get("year", "")),
        "card_number": str(data.get("card_number", "")),
        "type": data.get("type", "Other"),
        "confidence": _safe_float(data.get("confidence", 0.0)),
        "identify_token": f"idt_{secrets.token_urlsafe(12)}",
    })


@app.post("/api/verify")
async def verify(
    front: UploadFile = File(...), 
    back: UploadFile = File(...),
    card_name: Optional[str] = Form(None),
    card_set: Optional[str] = Form(None),
    card_number: Optional[str] = Form(None),
    card_year: Optional[str] = Form(None),
    card_type: Optional[str] = Form(None)
):
    """
    AI-powered card pre-assessment using front + back images.
    Optionally accepts card identification context for more accurate market estimates.
    Returns a structured condition report and preliminary grade context.
    """
    front_bytes = await front.read()
    back_bytes = await back.read()

    if not front_bytes or not back_bytes or len(front_bytes) < 1000 or len(back_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Images are too small or empty")

    # Build context string if identification provided
    context = ""
    if card_name or card_set:
        context = "\n\nIDENTIFIED CARD DETAILS:\n"
        if card_name:
            context += f"- Card Name: {card_name}\n"
        if card_set:
            context += f"- Set: {card_set}\n"
        if card_number:
            context += f"- Card Number: {card_number}\n"
        if card_year:
            context += f"- Year: {card_year}\n"
        if card_type:
            context += f"- Type: {card_type}\n"
        context += "\nUse this identification to provide accurate market estimates for THIS SPECIFIC card variant."

    prompt = f"""You are a professional trading card grader with extensive market knowledge. Analyze BOTH the front and back images and provide a comprehensive condition assessment WITH market value estimates.
{context}

Provide ONLY a JSON response with these exact fields:

{{
  "pregrade": "estimated PSA-style grade 1-10",
  "grade_corners": {{ "grade": "Mint/Near Mint/Excellent/Good/Poor", "notes": "..." }},
  "grade_edges": {{ "grade": "Mint/Near Mint/Excellent/Good/Poor", "notes": "..." }},
  "grade_surface": {{ "grade": "Mint/Near Mint/Excellent/Good/Poor", "notes": "..." }},
  "grade_centering": {{ "grade": "60/40 or better / 70/30 / 80/20 / Off-center", "notes": "..." }},
  "confidence": 0.0-1.0,
  "defects": ["List each defect with SIDE and location. If none, empty array."],
  "market_estimate": {{
    "raw_value": "estimated current raw card value in USD",
    "psa_8_value": "estimated PSA 8 graded value in USD (or null if not applicable)",
    "psa_9_value": "estimated PSA 9 graded value in USD (or null if not applicable)", 
    "psa_10_value": "estimated PSA 10 graded value in USD (or null if not applicable)",
    "grading_recommendation": "strong/recommended/consider/not_recommended based on value vs cost",
    "notes": "Brief market context for THIS SPECIFIC card variant"
  }}
}}

Base market estimates on:
- The SPECIFIC card variant identified above (if provided)
- Card rarity and desirability
- Set and year significance
- Character/player popularity
- Current condition from your assessment
- Historical market trends for THIS EXACT card

Be conservative with estimates. Use null for grades that don't apply to this card's value tier.
If card identification was provided, ensure market estimates are for that EXACT variant (e.g., 1st Edition vs Unlimited, Holo vs Non-Holo).
Respond ONLY with valid JSON, no other text."""

    msg = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(front_bytes)}", "detail": "high"}},
            {"type": "text", "text": "FRONT IMAGE ABOVE ☝️ | BACK IMAGE BELOW 👇"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(back_bytes)}", "detail": "high"}},
        ],
    }]

    result = await _openai_chat(msg, max_tokens=1800, temperature=0.1)
    if result.get("error"):
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

    data = _parse_json_or_none(result.get("content", "")) or {}

    # Parse market estimate if available
    market_estimate = data.get("market_estimate", {})
    if market_estimate:
        # Convert string values to floats
        for key in ["raw_value", "psa_8_value", "psa_9_value", "psa_10_value"]:
            if key in market_estimate and market_estimate[key]:
                try:
                    market_estimate[key] = float(str(market_estimate[key]).replace("$", "").replace(",", "").strip())
                except:
                    market_estimate[key] = None

    return JSONResponse(content={
        "pregrade": data.get("pregrade", "N/A"),
        "grade_corners": data.get("grade_corners", {"grade": "N/A", "notes": ""}),
        "grade_edges": data.get("grade_edges", {"grade": "N/A", "notes": ""}),
        "grade_surface": data.get("grade_surface", {"grade": "N/A", "notes": ""}),
        "grade_centering": data.get("grade_centering", {"grade": "N/A", "notes": ""}),
        "confidence": _safe_float(data.get("confidence", 0.0)),
        "defects": data.get("defects", []) if isinstance(data.get("defects", []), list) else [],
        "market_estimate": market_estimate,
        "verify_token": f"vfy_{secrets.token_urlsafe(12)}",
    })


# ==============================
# Memorabilia / Sealed Endpoints
# ==============================
@app.post("/api/identify-memorabilia")
async def identify_memorabilia(
    image1: UploadFile = File(...),
    image2: UploadFile = File(None),
    image3: UploadFile = File(None),
    image4: UploadFile = File(None),
):
    """
    Identify memorabilia or sealed product from 1-4 images
    Returns: item_type, description, authenticity_notes, confidence
    """
    images: List[bytes] = []
    b1 = await image1.read()
    if not b1 or len(b1) < 1000:
        raise HTTPException(status_code=400, detail="Primary image is too small or empty")
    images.append(b1)

    for f in [image2, image3, image4]:
        if f:
            bb = await f.read()
            if bb and len(bb) >= 1000:
                images.append(bb)

    prompt = """Analyze these images of collectible memorabilia or sealed product. Identify what this item is and assess key characteristics.
Provide ONLY a JSON response with these exact fields:

{
  "item_type": "Sealed Booster Box/Elite Trainer Box/Blister Pack/Graded Card/Signed Memorabilia/Display Case/Other",
  "description": "Detailed description of the item including brand, set/series, year if visible",
  "signatures": "Description of any visible signatures or autographs, or 'None visible'",
  "seal_condition": "Factory sealed/Opened/Resealed/Not applicable",
  "authenticity_notes": "Observations about authenticity markers, holograms, serial numbers, packaging quality, or red flags",
  "notable_features": "Special features, variants, errors, or unique aspects",
  "confidence": 0.0-1.0
}

Respond ONLY with valid JSON, no other text."""

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    labels = ["Primary Image", "Additional View", "Detail / Close-up", "Alternative Angle"]
    for idx, bb in enumerate(images):
        if idx > 0:
            content.append({"type": "text", "text": f"--- {labels[idx]} ---"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(bb)}", "detail": "high"}})

    msg = [{"role": "user", "content": content}]

    result = await _openai_chat(msg, max_tokens=1100, temperature=0.1)
    if result.get("error"):
        return JSONResponse(content={
            "item_type": "Unknown",
            "description": "Could not identify",
            "signatures": "Unable to assess",
            "seal_condition": "Unable to assess",
            "authenticity_notes": "AI identification failed",
            "notable_features": "",
            "confidence": 0.0,
            "identify_token": f"idt_{secrets.token_urlsafe(12)}",
            "error": "AI identification failed",
        })

    data = _parse_json_or_none(result.get("content", "")) or {}
    return JSONResponse(content={
        "item_type": data.get("item_type", "Unknown"),
        "description": data.get("description", ""),
        "signatures": data.get("signatures", "None visible"),
        "seal_condition": data.get("seal_condition", "Not applicable"),
        "authenticity_notes": data.get("authenticity_notes", ""),
        "notable_features": data.get("notable_features", ""),
        "confidence": _safe_float(data.get("confidence", 0.0)),
        "identify_token": f"idt_{secrets.token_urlsafe(12)}",
    })


@app.post("/api/assess-memorabilia")
async def assess_memorabilia(
    image1: UploadFile = File(...),
    image2: UploadFile = File(None),
    image3: UploadFile = File(None),
    image4: UploadFile = File(None),
):
    """
    Assess condition and authenticity of memorabilia/sealed products from 1-4 images
    """
    images: List[bytes] = []
    b1 = await image1.read()
    if not b1 or len(b1) < 1000:
        raise HTTPException(status_code=400, detail="Primary image is too small or empty")
    images.append(b1)

    for f in [image2, image3, image4]:
        if f:
            bb = await f.read()
            if bb and len(bb) >= 1000:
                images.append(bb)

    prompt = """You are a professional memorabilia and sealed product authenticator with extensive market knowledge. Analyze these images comprehensively.

Provide ONLY a JSON response:

{
  "overall_assessment": "Brief 2-3 sentence summary",
  "condition_grade": "Mint/Near Mint/Excellent/Very Good/Good/Fair/Poor",
  "seal_integrity": {"grade":"Factory Sealed/Intact/Compromised/Opened/Not Applicable","notes":"..."},
  "packaging_condition": {"grade":"Mint/Near Mint/Excellent/Very Good/Good/Fair/Poor","notes":"..."},
  "authenticity_assessment": {"grade":"Highly Confident/Likely Authentic/Uncertain/Concerns Present/Likely Counterfeit","notes":"..."},
  "signature_assessment": {"present": true/false, "grade":"Authentic/Likely Authentic/Uncertain/Concerns/Not Applicable","notes":"..."},
  "defects": ["..."],
  "value_factors": ["..."],
  "confidence": 0.0-1.0,
  "market_estimate": {
    "current_value_low": "conservative market value estimate in USD",
    "current_value_high": "optimistic market value estimate in USD",
    "current_value_typical": "most likely market value in USD",
    "graded_value_estimate": "potential value if professionally graded/authenticated (or null if not applicable)",
    "market_trend": "rising/stable/declining/insufficient_data",
    "demand_level": "very_high/high/moderate/low/niche",
    "notes": "Brief market context including factors affecting value"
  }
}

Base market estimates on:
- Item rarity and production numbers
- Age and historical significance
- Condition from your assessment
- Signature authenticity (if applicable)
- Seal integrity (if applicable)
- Current collector demand
- Recent comparable sales

For sealed products: Consider set popularity, print run size, and nostalgia factor.
For autographs: Consider signer prominence, signature quality, and authentication status.
Be conservative with estimates and provide ranges when uncertain.

Respond ONLY with valid JSON, no other text."""

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    labels = ["Primary Image", "Additional View", "Detail / Close-up", "Alternative Angle"]
    for idx, bb in enumerate(images):
        if idx > 0:
            content.append({"type": "text", "text": f"--- {labels[idx]} ---"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(bb)}", "detail": "high"}})

    msg = [{"role": "user", "content": content}]

    result = await _openai_chat(msg, max_tokens=1800, temperature=0.1)
    if result.get("error"):
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
            "error": "AI assessment failed",
        })

    data = _parse_json_or_none(result.get("content", "")) or {}
    
    # Parse market estimate if available
    market_estimate = data.get("market_estimate", {})
    if market_estimate:
        # Convert string values to floats
        for key in ["current_value_low", "current_value_high", "current_value_typical", "graded_value_estimate"]:
            if key in market_estimate and market_estimate[key]:
                try:
                    # Handle various formats: "$1,234", "1234", "$1234.56"
                    value_str = str(market_estimate[key]).replace("$", "").replace(",", "").strip()
                    market_estimate[key] = float(value_str) if value_str and value_str.lower() != "null" else None
                except:
                    market_estimate[key] = None
    
    return JSONResponse(content={
        "overall_assessment": data.get("overall_assessment", ""),
        "condition_grade": data.get("condition_grade", "N/A"),
        "seal_integrity": data.get("seal_integrity", {"grade": "N/A", "notes": ""}),
        "packaging_condition": data.get("packaging_condition", {"grade": "N/A", "notes": ""}),
        "authenticity_assessment": data.get("authenticity_assessment", {"grade": "N/A", "notes": ""}),
        "signature_assessment": data.get("signature_assessment", {"present": False, "grade": "N/A", "notes": ""}),
        "defects": data.get("defects", []) if isinstance(data.get("defects", []), list) else [],
        "value_factors": data.get("value_factors", []) if isinstance(data.get("value_factors", []), list) else [],
        "confidence": _safe_float(data.get("confidence", 0.0)),
        "market_estimate": market_estimate,  # ← ADD THIS
    })

# ==============================
# Market Context (Click-only)
# ==============================
_PRICE_RE = re.compile(r"(?P<cur>AU\\$|A\\$|US\\$|\\$)?\\s*(?P<num>[0-9]{1,3}(?:,[0-9]{3})*(?:\\.[0-9]{1,2})?)")


def _extract_prices(text: str) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    for m in _PRICE_RE.finditer(text or ""):
        cur = (m.group("cur") or "").strip()
        num = m.group("num")
        try:
            val = float(num.replace(",", ""))
        except Exception:
            continue
        if 0.5 <= val <= 500000:
            out.append((val, cur or ""))
    return out


def _trim_outliers(values: List[float]) -> List[float]:
    if len(values) < 6:
        return values
    v = sorted(values)
    # robust trimming: drop top/bottom 10% (min 1)
    k = max(1, len(v) // 10)
    trimmed = v[k:-k] if len(v) > 2 * k else v
    return trimmed if trimmed else v


async def _fetch_html(url: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, headers={"User-Agent": UA}, follow_redirects=True)
        if r.status_code != 200:
            return ""
        return r.text or ""


async def _search_ebay_sold_prices(query: str, negate_terms: Optional[List[str]] = None, limit: int = 25) -> Dict[str, Any]:
    """
    Scrape eBay completed/sold results and return price samples.
    Uses EBAY_DOMAIN. Attempts to prefer most recent (sort by End Date: _sop=13).
    """
    negate_terms = negate_terms or []
    q = query.strip()
    for t in negate_terms:
        if t:
            q += f" -{t}"
    encoded = q.replace(" ", "+")
    url = f"https://{EBAY_DOMAIN}/sch/i.html?_nkw={encoded}&_sacat=0&LH_Sold=1&LH_Complete=1&_sop=13"

    html = await _fetch_html(url)
    if not html:
        return {"prices": [], "currency": "", "sample_size": 0, "url": url}

    soup = BeautifulSoup(html, "html.parser")
    price_spans = soup.find_all("span", class_=re.compile(r"s-item__price"))
    prices: List[float] = []
    currencies: List[str] = []

    for span in price_spans[: max(10, limit)]:
        txt = span.get_text(" ", strip=True)
        for val, cur in _extract_prices(txt):
            prices.append(val)
            currencies.append(cur)

    # Determine currency (best-effort)
    currency = ""
    if currencies:
        # If majority contains AU$ or A$
        au_count = sum(1 for c in currencies if c in ("AU$", "A$"))
        us_count = sum(1 for c in currencies if c == "US$")
        if au_count >= max(1, len(currencies) // 3):
            currency = "AUD"
        elif us_count >= max(1, len(currencies) // 3):
            currency = "USD"

    # Clean
    prices = [p for p in prices if 1 < p < 500000][:limit]

    return {"prices": prices, "currency": currency, "sample_size": len(prices), "url": url}


async def _tcgplayer_raw_anchor(card_name: str, card_set: Optional[str]) -> Dict[str, Any]:
    """
    Optional raw anchor using PokemonTCG API (not TCGPlayer proper).
    Returns a small list of market prices if available.
    """
    if not POKEMONTCG_API_KEY:
        return {"prices": [], "currency": "USD", "sample_size": 0}

    try:
        headers = {"X-Api-Key": POKEMONTCG_API_KEY}
        q = f"name:{card_name}"
        if card_set:
            q += f" set.name:{card_set}"
        url = f"https://api.pokemontcg.io/v2/cards?q={q}"
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, headers=headers)
            if r.status_code != 200:
                return {"prices": [], "currency": "USD", "sample_size": 0}
            data = r.json()
            cards = data.get("data") or []
            if not cards:
                return {"prices": [], "currency": "USD", "sample_size": 0}
            card = cards[0]
            prices = card.get("tcgplayer", {}).get("prices", {}) or {}
            out: List[float] = []
            for k in ("normal", "holofoil", "reverseHolofoil"):
                if k in prices:
                    v = prices[k].get("market")
                    if v:
                        out.append(float(v))
            return {"prices": out, "currency": "USD", "sample_size": len(out)}
    except Exception:
        return {"prices": [], "currency": "USD", "sample_size": 0}


def _market_trend_from_recent(prices: List[float]) -> str:
    """
    Very lightweight trend: compare average of most-recent 5 vs older 5 (based on list order from _sop=13).
    """
    if len(prices) < 10:
        return "insufficient_data"
    recent = prices[:5]
    older = prices[-5:]
    r = mean(recent) if recent else 0
    o = mean(older) if older else 0
    if o <= 0:
        return "insufficient_data"
    change = (r - o) / o
    if change > 0.10:
        return "increasing"
    if change < -0.10:
        return "decreasing"
    return "stable"


def _liquidity_label(n: int) -> str:
    if n >= 25:
        return "high"
    if n >= 10:
        return "moderate"
    if n >= 5:
        return "thin"
    return "insufficient"


def _stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"avg": 0, "median": 0, "low": 0, "high": 0, "sample_size": 0}
    trimmed = _trim_outliers(values)
    return {
        "avg": round(mean(trimmed), 2) if trimmed else 0,
        "median": round(median(trimmed), 2) if trimmed else 0,
        "low": round(min(values), 2),
        "high": round(max(values), 2),
        "sample_size": len(values),
    }
async def _scrape_pricecharting_direct(card_name: str, card_set: str, card_number: str) -> Dict[str, Any]:
    """
    Direct scrape of PriceCharting.com for graded prices
    """
    try:
        print(f"🔍 Searching PriceCharting: {card_name} {card_set} {card_number}")
        
        set_slug = card_set.lower().replace(" ", "-").replace("'", "")
        card_slug = card_name.lower().replace(" ", "-").replace("'", "")
        number_clean = card_number.lstrip("0") if card_number else ""
        
        urls = [
            f"https://www.pricecharting.com/game/pokemon-{set_slug}/{card_slug}-{number_clean}",
            f"https://www.pricecharting.com/game/pokemon-{set_slug}/{card_slug}",
        ]
        
        graded_data = {}
        
        for url in urls:
            print(f"   Trying: {url}")
            html = await _fetch_html(url)
            
            if not html or len(html) < 2000:
                continue
            
            soup = BeautifulSoup(html, "html.parser")
            
            for row in soup.find_all("tr"):
                text = row.get_text(" ", strip=True)
                
                grade = None
                if "PSA 10" in text or "Grade 10" in text:
                    grade = "10"
                elif "PSA 9" in text or "Grade 9" in text:
                    grade = "9"
                elif "PSA 8" in text or "Grade 8" in text:
                    grade = "8"
                
                if grade:
                    prices = _extract_prices(text)
                    for price, _ in prices:
                        if 10 < price < 100000:
                            if grade not in graded_data:
                                graded_data[grade] = []
                            graded_data[grade].append(price)
                            print(f"   ✅ Found PSA {grade}: ${price}")
            
            if graded_data:
                print(f"   ✅ PriceCharting: {sum(len(v) for v in graded_data.values())} prices")
                break
        
        return {"graded_prices": graded_data}
        
    except Exception as e:
        print(f"   ❌ PriceCharting error: {str(e)}")
        return {"graded_prices": {}}


async def _scrape_tcgplayer_direct(card_name: str, card_set: str) -> Dict[str, Any]:
    """
    Direct scrape of TCGPlayer.com for market prices
    """
    try:
        print(f"🔍 Searching TCGPlayer: {card_name} {card_set}")
        
        query = f"{card_name} {card_set}".strip()
        encoded = query.replace(" ", "%20")
        url = f"https://www.tcgplayer.com/search/pokemon/product?q={encoded}"
        
        print(f"   Trying: {url}")
        html = await _fetch_html(url)
        
        if not html or len(html) < 2000:
            return {"raw_prices": [], "graded_prices": {}}
        
        soup = BeautifulSoup(html, "html.parser")
        raw_prices = []
        graded_prices = {}
        
        for elem in soup.find_all(["span", "div"], class_=re.compile(r"price|market", re.I)):
            text = elem.get_text(" ", strip=True)
            context = elem.parent.get_text(" ", strip=True) if elem.parent else ""
            
            prices = _extract_prices(text)
            for price, _ in prices:
                if 1 < price < 50000:
                    if "PSA 10" in context.upper():
                        if "10" not in graded_prices:
                            graded_prices["10"] = []
                        graded_prices["10"].append(price)
                        print(f"   ✅ Found PSA 10: ${price}")
                    elif "PSA 9" in context.upper():
                        if "9" not in graded_prices:
                            graded_prices["9"] = []
                        graded_prices["9"].append(price)
                        print(f"   ✅ Found PSA 9: ${price}")
                    elif "GRADED" not in context.upper():
                        raw_prices.append(price)
                        print(f"   ✅ Found raw: ${price}")
        
        return {"raw_prices": raw_prices, "graded_prices": graded_prices}
        
    except Exception as e:
        print(f"   ❌ TCGPlayer error: {str(e)}")
        return {"raw_prices": [], "graded_prices": {}}

@app.post("/api/market-context")
async def market_context(
    card_name: str = Form(...),
    predicted_grade: str = Form(...),
    confidence: float = Form(0.0),
    card_number: Optional[str] = Form(None),
    card_set: Optional[str] = Form(None),
    grading_cost: float = Form(55.0),
):
    """
    Click-only market context endpoint.

    Returns observed sold pricing context and grade sensitivity.
    This is informational only and should not be framed as financial advice.
    """
    # Clean up inputs
    clean_name = card_name.strip()
    clean_set = (card_set or "").strip()
    clean_number = (card_number or "").replace("#", "").lstrip("0").strip()  # Remove # and leading zeros
    
    # Build identifier (avoid duplicates)
    parts = [clean_name]
    if clean_set and clean_set.lower() not in clean_name.lower():
        parts.append(clean_set)
    if clean_number:
        parts.append(clean_number)
    
    ident = " ".join(parts)
    conf = _clamp(_safe_float(confidence, 0.0), 0.0, 1.0)
    g = _grade_bucket(predicted_grade) or 9

    # --------------------
    # Gather sold samples
    # --------------------
    sources: List[str] = []
    meta_urls: Dict[str, str] = {}

    # Raw sold (negate graded terms)
    raw_res = await _search_ebay_sold_prices(ident, negate_terms=["graded", "psa", "bgs", "cgc", "slab"], limit=30)
    raw_prices = raw_res["prices"]
    raw_currency = raw_res.get("currency", "")
    meta_urls["ebay_raw"] = raw_res.get("url", "")
    if raw_prices:
        sources.append("eBay (sold/completed)")

    graded_prices: Dict[str, List[float]] = {}
    for grade in ("10", "9", "8"):
        gr_res = await _search_ebay_sold_prices(f"{ident} PSA {grade}", negate_terms=[], limit=20)
        meta_urls[f"ebay_psa_{grade}"] = gr_res.get("url", "")
        if gr_res["prices"]:
            graded_prices[grade] = gr_res["prices"]
            if "eBay (sold/completed)" not in sources:
                sources.append("eBay (sold/completed)")

    # Try multiple sources for raw prices
    tcg_anchor = await _tcgplayer_raw_anchor(card_name, card_set)
    if tcg_anchor.get("prices"):
        sources.append("PokemonTCG API")
    anchor_prices = tcg_anchor.get("prices", [])
    
    # Direct TCGPlayer scrape
    tcg_direct = await _scrape_tcgplayer_direct(clean_name, clean_set)
    if tcg_direct.get("raw_prices"):
        raw_prices.extend(tcg_direct["raw_prices"])
        sources.append("TCGPlayer.com")
    if tcg_direct.get("graded_prices"):
        for grade, prices in tcg_direct["graded_prices"].items():
            if grade not in graded_prices:
                graded_prices[grade] = []
            graded_prices[grade].extend(prices)
    
    # PriceCharting scrape
    pc_data = await _scrape_pricecharting_direct(clean_name, clean_set, clean_number)
    if pc_data.get("graded_prices"):
        sources.append("PriceCharting.com")
        for grade, prices in pc_data["graded_prices"].items():
            if grade not in graded_prices:
                graded_prices[grade] = []
            graded_prices[grade].extend(prices)

    # Decide if we have enough data to present meaningful context
    has_raw = len(raw_prices) >= 3 or len(anchor_prices) >= 2
    has_graded = any(len(v) >= 3 for v in graded_prices.values())

    if not has_raw and not has_graded:
        return JSONResponse(content={
            "available": False,
            "card": {"name": card_name, "set": card_set, "number": card_number, "identifier": ident},
            "message": "No sufficient market samples found for this item.",
            "sources": sources,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "disclaimer": "Informational market context only. Not financial advice."
        })

    # --------------------
    # Compute statistics
    # --------------------
    raw_stats = _stats(raw_prices)
    anchor_stats = _stats(anchor_prices)

    # Choose a raw baseline: prefer eBay raw median/avg if sample ok, else anchor median/avg
    raw_baseline = 0.0
    if raw_stats["sample_size"] >= 3:
        raw_baseline = raw_stats["median"] or raw_stats["avg"]
    elif anchor_stats["sample_size"] >= 2:
        raw_baseline = anchor_stats["median"] or anchor_stats["avg"]

    graded_stats: Dict[str, Any] = {k: _stats(v) for k, v in graded_prices.items()}

    # Determine trend & liquidity based on raw sold sample (if available)
    trend = _market_trend_from_recent(raw_prices) if raw_prices else "insufficient_data"
    liquidity = _liquidity_label(raw_stats["sample_size"] if raw_prices else anchor_stats["sample_size"])

    # Grade probability + expected value (only if we have at least PSA 8/9 samples)
    dist = _grade_distribution(g, conf)
    expected_value = None
    required = 0
    for need in ("8", "9"):
        if need in graded_stats and graded_stats[need]["sample_size"] >= 3:
            required += 1

    if required >= 2:
        expected_value = 0.0
        for grade, p in dist.items():
            st = graded_stats.get(grade, {"avg": 0, "median": 0, "sample_size": 0})
            # Prefer median if sample smallish; else avg of trimmed
            val = (st["median"] if st["sample_size"] < 10 else st["avg"]) or st["avg"] or 0.0
            expected_value += p * val
        expected_value = round(expected_value, 2)

    # Sensitivity: how much PSA 10 differs from PSA 9 if we have both
    sensitivity = "unknown"
    if "10" in graded_stats and "9" in graded_stats and graded_stats["10"]["sample_size"] >= 3 and graded_stats["9"]["sample_size"] >= 3:
        p10 = graded_stats["10"]["median"] or graded_stats["10"]["avg"]
        p9 = graded_stats["9"]["median"] or graded_stats["9"]["avg"]
        if p9 and p10:
            ratio = p10 / p9
            if ratio >= 1.8:
                sensitivity = "very_high"
            elif ratio >= 1.3:
                sensitivity = "high"
            elif ratio >= 1.1:
                sensitivity = "moderate"
            else:
                sensitivity = "low"

    # Provide an "estimated value difference" only if expected_value exists and raw baseline exists
    value_difference = None
    if expected_value is not None and raw_baseline:
        value_difference = round(expected_value - raw_baseline - float(grading_cost or 0.0), 2)

    # Currency best-effort
    currency = raw_currency or ("USD" if tcg_anchor.get("currency") else "")

    return JSONResponse(content={
        "available": True,
        "card": {"name": card_name, "set": card_set, "number": card_number, "identifier": ident},
        "observed": {
            "currency": currency or "unknown",
            "raw": raw_stats,
            "raw_anchor": anchor_stats if anchor_stats["sample_size"] else None,
            "graded_psa": graded_stats,
            "trend": trend,
            "liquidity": liquidity,
        },
        "grade_impact": {
            "predicted_grade": str(g),
            "confidence": conf,
            "grade_distribution": dist,
            "expected_graded_value": expected_value,
            "raw_baseline_value": round(raw_baseline, 2) if raw_baseline else None,
            "grading_cost": round(float(grading_cost or 0.0), 2),
            "estimated_value_difference": value_difference,
            "sensitivity": sensitivity,
        },
        "sources": sources,
        "meta": {"urls": meta_urls},
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "disclaimer": "Informational market context only. Figures are based on observed recent sales and/or third‑party price anchors and may vary. This is not financial advice or a guarantee of outcome."
    })


# Legacy alias (keep existing frontend working if it calls the old route)
@app.post("/api/market-intelligence")
async def market_intelligence_alias(
    card_name: str = Form(...),
    predicted_grade: str = Form(...),
    confidence: float = Form(0.0),
    card_number: Optional[str] = Form(None),
    card_set: Optional[str] = Form(None),
    grading_cost: float = Form(55.0),
):
    res = await market_context(
        card_name=card_name,
        predicted_grade=predicted_grade,
        confidence=confidence,
        card_number=card_number,
        card_set=card_set,
        grading_cost=grading_cost,
    )
    # attach a small deprecation hint
    if isinstance(res, JSONResponse):
        payload = json.loads(res.body.decode("utf-8"))
        payload["deprecated"] = True
        payload["deprecated_message"] = "Use /api/market-context (click-only) instead."
        return JSONResponse(content=payload, status_code=res.status_code)
    return res


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
