"""
The Collectors League Australia — Scan API
Futureproof v6 (2026-01-31)

Goals:
- Canonical identification object (stable IDs)
- Set-code aware identification (e.g., PFL -> Phantasmal Flames)
- Richer grading detail (front/back + per-corner breakdown + detailed flags)
- Click-only Market Context that is resilient (multi-pass query ladder + never "nothing" if any samples exist)
- No financial advice framing; informational context only.

NOTE: This file expects these dependencies in requirements.txt:
fastapi, uvicorn[standard], python-multipart, httpx, beautifulsoup4, lxml
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, Response
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

# CORS: allow your site(s) to call Render directly.
# (If you later proxy through WordPress via admin-ajax, you can tighten this further.)
ALLOWED_ORIGINS = [
    "https://collectors-league.com",
    "https://www.collectors-league.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # keep False unless you are using cookies/sessions across domains
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

APP_VERSION = os.getenv("CL_SCAN_VERSION", "2026-01-31-v6.2.1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
POKEMONTCG_API_KEY = os.getenv("POKEMONTCG_API_KEY", "").strip()

POKEMONTCG_BASE = "https://api.pokemontcg.io/v2"

# eBay region: AU by default
EBAY_DOMAIN = os.getenv("EBAY_DOMAIN", "www.ebay.com.au").strip() or "www.ebay.com.au"

# If you want to completely disable eBay scraping (recommended for stability), set:
#   EBAY_ENABLED=0
EBAY_ENABLED = os.getenv("EBAY_ENABLED", "1").strip() not in ("0", "false", "False", "no", "NO")

# By default we DO NOT use eBay for Pokemon (PokemonTCG provides authoritative prices + images).
# If you want it back for Pokemon, set:
#   USE_EBAY_FOR_POKEMON=1
USE_EBAY_FOR_POKEMON = os.getenv("USE_EBAY_FOR_POKEMON", "0").strip() in ("1", "true", "True", "yes", "YES")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set!")

# ==============================
# Set Code Mapping (expand over time)
# ==============================
# Keep this conservative: only include codes you are certain about.
# You can expand this safely as you see more items.
SET_CODE_MAP: Dict[str, str] = {
    # Examples (replace/expand with your real catalog)
    "PFL": "Phantasmal Flames",
    "OBF": "Obsidian Flames",
    "SVI": "Scarlet & Violet",
    "BS": "Base Set",
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
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _norm_printed_card_number(s: str) -> str:
    """
    IMPORTANT: preserve the printed card number exactly as it appears on the card,
    including leading zeros. Examples:
      - '004/120' stays '004/120'
      - '#004/120' -> '004/120'
      - '  004  ' -> '004'
    """
    s = (s or "").strip().replace("#", "").strip()
    # normalize whitespace but do NOT strip leading zeros
    s = re.sub(r"\s+", "", s)  # card numbers should not have spaces
    return s


def _num_for_pokemontcg_query(printed: str) -> str:
    """
    PokemonTCG API typically stores number as '4' not '004/120'.
    Convert printed number to best query number:
      - '004/120' -> '4'
      - '004' -> '4'
      - '4/120' -> '4'
    """
    s = _norm_printed_card_number(printed)
    if not s:
        return ""
    if "/" in s:
        s = s.split("/", 1)[0]
    s = s.lstrip("0") or "0"
    return s


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


def _grade_bucket(predicted_grade: str) -> Optional[int]:
    m = re.search(r"(\d{1,2})", (predicted_grade or "").strip())
    if not m:
        return None
    g = int(m.group(1))
    if 1 <= g <= 10:
        return g
    return None


def _grade_distribution(predicted_grade: int, confidence: float) -> Dict[str, float]:
    """
    Conservative distribution over grades {10,9,8} around predicted_grade.
    """
    c = _clamp(confidence, 0.05, 0.95)
    p_pred = 0.45 + 0.50 * c
    remainder = 1.0 - p_pred

    if predicted_grade >= 10:
        dist = {"10": p_pred, "9": remainder * 0.75, "8": remainder * 0.25}
    elif predicted_grade == 9:
        dist = {"10": remainder * 0.25, "9": p_pred, "8": remainder * 0.75}
    else:
        dist = {"10": remainder * 0.10, "9": remainder * 0.30, "8": p_pred + remainder * 0.60}

    total = sum(dist.values())
    if total <= 0:
        return {"10": 0.0, "9": 0.0, "8": 0.0}
    for k in dist:
        dist[k] = round(dist[k] / total, 4)
    return dist


def _canonicalize_set(set_code: Optional[str], set_name: Optional[str]) -> Dict[str, Any]:
    sc = (set_code or "").strip().upper()
    sn = _norm_ws(set_name or "")

    mapped = SET_CODE_MAP.get(sc) if sc else None
    if mapped:
        return {"set_code": sc, "set_name": mapped, "set_source": "code_map"}
    if sc and sn:
        return {"set_code": sc, "set_name": sn, "set_source": "ai+code"}
    if sn:
        rev = {v.lower(): k for k, v in SET_CODE_MAP.items()}
        maybe = rev.get(sn.lower())
        if maybe:
            return {"set_code": maybe, "set_name": sn, "set_source": "reverse_map"}
        return {"set_code": "", "set_name": sn, "set_source": "ai_name_only"}
    if sc:
        return {"set_code": sc, "set_name": "", "set_source": "code_only"}
    return {"set_code": "", "set_name": "", "set_source": "unknown"}


# ==============================
# PokemonTCG.io helpers (authoritative card data + prices)
# ==============================
async def _pokemontcg_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """GET helper for PokemonTCG API. `path` may be full URL or relative like '/cards'."""
    if not POKEMONTCG_API_KEY:
        return {}
    url = path if path.startswith("http") else f"{POKEMONTCG_BASE}{path}"
    headers = {"X-Api-Key": POKEMONTCG_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, headers=headers, params=params)
            if r.status_code != 200:
                return {}
            return r.json() if r.content else {}
    except Exception:
        return {}


async def _pokemontcg_set_by_ptcgo_code(ptcgo_code: str) -> Dict[str, Any]:
    """Resolve set by ptcgoCode so we can query by set.id when available (more precise)."""
    code = (ptcgo_code or "").strip().upper()
    if not code:
        return {}
    data = await _pokemontcg_get("/sets", params={"q": f"ptcgoCode:{code}", "pageSize": 5})
    sets = data.get("data") or []
    if sets:
        return sets[0]
    return {}


async def _pokemontcg_resolve_card_id(card_name: str, set_code: str, card_number_printed: str, set_name: str = "") -> str:
    """
    Resolve most likely PokemonTCG card id using a query ladder + set lookup.
    IMPORTANT: Uses a derived number for PokemonTCG query (e.g. '004/120' -> '4')
    but does NOT mutate the printed number stored in your system.
    """
    name = _norm_ws(card_name)
    sc = (set_code or "").strip().upper()
    sn = _norm_ws(set_name or "")

    printed = _norm_printed_card_number(card_number_printed or "")
    qnum = _num_for_pokemontcg_query(printed)

    # If we have a set code, try to get set.id first (most precise)
    set_obj = await _pokemontcg_set_by_ptcgo_code(sc) if sc else {}
    set_id = str(set_obj.get("id", "")) if isinstance(set_obj, dict) else ""

    queries: List[str] = []
    if set_id and qnum:
        queries.append(f"set.id:{set_id} number:{qnum}")
    if sc and qnum:
        queries.append(f"set.ptcgoCode:{sc} number:{qnum}")
    if name and qnum and set_id:
        queries.append(f'name:"{name}" set.id:{set_id} number:{qnum}')
    if name and qnum and sn:
        queries.append(f'name:"{name}" number:{qnum} set.name:"{sn}"')
    if name and qnum:
        queries.append(f'name:"{name}" number:{qnum}')
    if name and set_id:
        queries.append(f'name:"{name}" set.id:{set_id}')
    if name and sn:
        queries.append(f'name:"{name}" set.name:"{sn}"')
    if name:
        queries.append(f'name:"{name}"')

    for q in queries:
        data = await _pokemontcg_get("/cards", params={"q": q, "pageSize": 10, "orderBy": "number"})
        cards = (data.get("data") or [])
        if not cards:
            continue

        # Prefer exact number match when we have it
        if qnum:
            for c in cards:
                if str(c.get("number", "")).strip() == qnum:
                    return str(c.get("id", "")) or ""
        return str(cards[0].get("id", "")) or ""

    return ""


async def _pokemontcg_card_by_id(card_id: str) -> Dict[str, Any]:
    """Authoritative card record: GET /cards/<id>"""
    if not card_id:
        return {}
    data = await _pokemontcg_get(f"/cards/{card_id}")
    card = data.get("data") if isinstance(data, dict) else None
    return card if isinstance(card, dict) else {}


def _pokemontcg_extract_prices(card: Dict[str, Any]) -> Dict[str, Any]:
    """Return full price objects + a simple numeric anchor list for calculations."""
    tcg = (card or {}).get("tcgplayer", {}) or {}
    cm = (card or {}).get("cardmarket", {}) or {}

    tcg_prices = tcg.get("prices") if isinstance(tcg, dict) else None
    cm_prices = cm.get("prices") if isinstance(cm, dict) else None

    anchor: List[float] = []
    currencies: List[str] = []

    # TCGplayer markets (USD)
    if isinstance(tcg_prices, dict) and tcg_prices:
        currencies.append("USD")
        for _, obj in tcg_prices.items():
            if isinstance(obj, dict) and obj.get("market") is not None:
                try:
                    anchor.append(float(obj["market"]))
                except Exception:
                    pass

    # Cardmarket trend/avg (EUR)
    if isinstance(cm_prices, dict) and cm_prices:
        currencies.append("EUR")
        for k in ("trendPrice", "averageSellPrice"):
            if cm_prices.get(k) is not None:
                try:
                    anchor.append(float(cm_prices[k]))
                except Exception:
                    pass

    return {
        "anchor_values": anchor,
        "currencies": currencies,
        "tcgplayer": {"url": tcg.get("url", ""), "prices": tcg_prices} if isinstance(tcg_prices, dict) and tcg_prices else None,
        "cardmarket": {"url": cm.get("url", ""), "prices": cm_prices} if isinstance(cm_prices, dict) and cm_prices else None,
    }


# ==============================
# Root & Health
# ==============================
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "cl-scan-api",
        "version": APP_VERSION,
        "message": "The Collectors League Australia — Multi-Item Assessment API (Futureproof v6)",
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
        "supports": ["cards", "memorabilia", "sealed_products", "market_context_click_only", "canonical_id", "detailed_grading"],
        "ebay_domain": EBAY_DOMAIN,
        "ebay_enabled": EBAY_ENABLED,
        "use_ebay_for_pokemon": USE_EBAY_FOR_POKEMON,
        "cors_allowed_origins": ALLOWED_ORIGINS,
    }


@app.head("/")
def head_root():
    return Response(status_code=200)


@app.head("/health")
def head_health():
    return Response(status_code=200)


# ==============================
# Card: Identify
# ==============================
@app.post("/api/identify")
async def identify(front: UploadFile = File(...)):
    """
    Identify a trading card from its front image using AI vision.
    Returns canonical_id plus (for Pokemon) an optional authoritative PokemonTCG card resolution.
    """
    image_bytes = await front.read()
    if not image_bytes or len(image_bytes) < 200:
        raise HTTPException(status_code=400, detail="Image is too small or empty")

    prompt = """You are identifying a trading card from a front photo.

Return ONLY valid JSON with these exact fields:

{
  "card_name": "exact card name including variants (ex, V, VMAX, holo, first edition, etc.)",
  "card_type": "Pokemon/Magic/YuGiOh/Sports/OnePiece/Other",
  "year": "4 digit year if visible else empty string",
  "card_number": "card number if visible (e.g. 4/102 or 004/120) else empty string",
  "set_code": "2-4 letter/number set abbreviation printed on the card if visible (e.g. PFL, OBF, SVI, MEW). EMPTY if not visible",
  "set_name": "set or series name (full name) if visible/known else empty string",
  "confidence": 0.0-1.0,
  "notes": "one short sentence about how you identified it"
}

Rules:
- PRIORITIZE the set_code if you can see one.
- Do not guess a set_code you cannot see.
- If you cannot identify with confidence, set card_name to "Unknown" and confidence to 0.0.
Respond ONLY with JSON, no extra text."""

    msg = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(image_bytes)}", "detail": "high"}},
        ],
    }]

    result = await _openai_chat(msg, max_tokens=800, temperature=0.1)
    if result.get("error"):
        return JSONResponse(content={
            "card_name": "Unknown",
            "card_type": "Other",
            "year": "",
            "card_number": "",
            "set_code": "",
            "set_name": "",
            "confidence": 0.0,
            "notes": "AI identification failed",
            "identify_token": f"idt_{secrets.token_urlsafe(12)}",
            "canonical_id": {
                "card_name": "Unknown",
                "card_type": "Other",
                "year": "",
                "card_number": "",
                "set_code": "",
                "set_name": "",
                "confidence": 0.0,
                "set_source": "unknown"
            },
            "pokemontcg": None,
            "error": "AI identification failed"
        })

    data = _parse_json_or_none(result.get("content", "")) or {}

    card_name = _norm_ws(str(data.get("card_name", "Unknown")))
    card_type = _norm_ws(str(data.get("card_type", "Other")))
    year = _norm_ws(str(data.get("year", "")))

    # Preserve printed number (leading zeros!)
    card_number_printed = _norm_printed_card_number(str(data.get("card_number", "")))

    set_code = _norm_ws(str(data.get("set_code", ""))).upper()
    set_name = _norm_ws(str(data.get("set_name", "")))
    conf = _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0)
    notes = _norm_ws(str(data.get("notes", "")))

    set_info = _canonicalize_set(set_code, set_name)

    canonical_id: Dict[str, Any] = {
        "card_name": card_name,
        "card_type": card_type,
        "year": year,
        "card_number": card_number_printed,
        "set_code": set_info["set_code"],
        "set_name": set_info["set_name"],
        "set_source": set_info["set_source"],
        "confidence": conf,
    }

    # Optional: resolve PokemonTCG card id for Pokemon items (authoritative metadata)
    pokemontcg_payload = None
    if POKEMONTCG_API_KEY and card_type.strip().lower() == "pokemon":
        pid = await _pokemontcg_resolve_card_id(
            card_name=card_name,
            set_code=set_info["set_code"],
            card_number_printed=card_number_printed,
            set_name=set_info["set_name"],
        )
        if pid:
            card = await _pokemontcg_card_by_id(pid)
            canonical_id["external_ids"] = {"pokemontcg_id": pid}
            pokemontcg_payload = {
                "id": pid,
                "name": card.get("name", ""),
                "number": card.get("number", ""),
                "rarity": card.get("rarity", ""),
                "set": card.get("set", {}) or {},
                "images": card.get("images", {}) or {},
                "links": {
                    "tcgplayer": (card.get("tcgplayer", {}) or {}).get("url", ""),
                    "cardmarket": (card.get("cardmarket", {}) or {}).get("url", ""),
                }
            }

    return JSONResponse(content={
        "card_name": card_name,
        "card_type": card_type,
        "year": year,
        "card_number": card_number_printed,
        "set_code": set_info["set_code"],
        "set_name": set_info["set_name"],
        "confidence": conf,
        "notes": notes,
        "identify_token": f"idt_{secrets.token_urlsafe(12)}",
        "canonical_id": canonical_id,
        "pokemontcg": pokemontcg_payload,
    })


# ==============================
# Card: Verify (detailed grading)
# ==============================
@app.post("/api/verify")
async def verify(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    card_name: Optional[str] = Form(None),
    card_set: Optional[str] = Form(None),
    card_number: Optional[str] = Form(None),
    card_year: Optional[str] = Form(None),
    card_type: Optional[str] = Form(None),
    set_code: Optional[str] = Form(None),
):
    """
    AI-powered card pre-assessment using front + back images.
    Returns detailed grading: corners/edges/surface front+back + per-corner notes + flags.
    Market estimates are NOT computed here (click-only endpoint).
    """
    front_bytes = await front.read()
    back_bytes = await back.read()

    if (not front_bytes or not back_bytes or len(front_bytes) < 200 or len(back_bytes) < 200):
        raise HTTPException(status_code=400, detail="Images are too small or empty")

    # Canonicalize provided ID context (if any)
    provided_name = _norm_ws(card_name or "")
    provided_set = _norm_ws(card_set or "")
    provided_num = _norm_printed_card_number(card_number or "")
    provided_year = _norm_ws(card_year or "")
    provided_type = _norm_ws(card_type or "")
    provided_code = _norm_ws(set_code or "").upper()

    set_info = _canonicalize_set(provided_code, provided_set)

    context = ""
    if provided_name or set_info["set_name"] or provided_num or provided_year or provided_type or set_info["set_code"]:
        context = "\n\nKNOWN/PROVIDED CARD DETAILS (use as hints, do not force if images contradict):\n"
        if provided_name:
            context += f"- Card Name: {provided_name}\n"
        if set_info["set_name"]:
            context += f"- Set Name: {set_info['set_name']}\n"
        if set_info["set_code"]:
            context += f"- Set Code: {set_info['set_code']}\n"
        if provided_num:
            context += f"- Card Number (printed): {provided_num}\n"
        if provided_year:
            context += f"- Year: {provided_year}\n"
        if provided_type:
            context += f"- Type: {provided_type}\n"

    prompt = f"""You are a professional trading card grader.

Analyze BOTH images (front + back). Return ONLY valid JSON.

You MUST provide:
- Separate assessments for FRONT and BACK
- Per-corner notes (top_left, top_right, bottom_left, bottom_right)
- Separate edges and surface notes for front/back
- Centering ratios for front/back (e.g. "55/45" and "60/40")
- A clear, more detailed Assessment Summary (2-4 sentences) mentioning the specific issues you see.

{context}

Return JSON with this EXACT structure:

{{
  "pregrade": "estimated PSA-style grade 1-10 (e.g. 8, 9, 10)",
  "confidence": 0.0-1.0,
  "centering": {{
    "front": {{"grade":"55/45","notes":"..."}},
    "back":  {{"grade":"60/40","notes":"..."}}
  }},
  "corners": {{
    "front": {{
      "top_left": {{"condition":"sharp/minor_whitening/whitening/bend/ding","notes":"..."}},
      "top_right": {{"condition":"...","notes":"..."}},
      "bottom_left": {{"condition":"...","notes":"..."}},
      "bottom_right": {{"condition":"...","notes":"..."}}
    }},
    "back": {{
      "top_left": {{"condition":"...","notes":"..."}},
      "top_right": {{"condition":"...","notes":"..."}},
      "bottom_left": {{"condition":"...","notes":"..."}},
      "bottom_right": {{"condition":"...","notes":"..."}}
    }}
  }},
  "edges": {{
    "front": {{"grade":"Mint/Near Mint/Excellent/Good/Poor","notes":"detailed notes"}},
    "back":  {{"grade":"Mint/Near Mint/Excellent/Good/Poor","notes":"detailed notes"}}
  }},
  "surface": {{
    "front": {{"grade":"Mint/Near Mint/Excellent/Good/Poor","notes":"detailed notes"}},
    "back":  {{"grade":"Mint/Near Mint/Excellent/Good/Poor","notes":"detailed notes"}}
  }},
  "defects": [
    "Each defect as a clear sentence with SIDE and location (e.g. 'BACK bottom-left corner: whitening')"
  ],
  "flags": [
    "Short bullet flags for important issues (print line, holo scratches, whitening, edge chipping, dent, crease, stain, writing)"
  ],
  "assessment_summary": "2-4 sentence narrative. Mention front vs back differences and the biggest grade limiters.",
  "observed_id": {{
    "card_name": "best-effort from images",
    "set_code": "best-effort from images (only if visible)",
    "set_name": "best-effort from images",
    "card_number": "best-effort from images (keep leading zeros if visible)",
    "year": "best-effort from images",
    "card_type": "Pokemon/Magic/YuGiOh/Sports/OnePiece/Other"
  }}
}}

Important:
- If something cannot be determined, use empty string "" or empty arrays.
- Do not include market values in this endpoint.
Respond ONLY with JSON."""

    msg = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(front_bytes)}", "detail": "high"}},
            {"type": "text", "text": "FRONT IMAGE ABOVE ☝️ | BACK IMAGE BELOW 👇"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(back_bytes)}", "detail": "high"}},
        ],
    }]

    result = await _openai_chat(msg, max_tokens=2200, temperature=0.1)
    if result.get("error"):
        return JSONResponse(content={
            "pregrade": "Unable to assess",
            "confidence": 0.0,
            "centering": {"front": {"grade": "N/A", "notes": ""}, "back": {"grade": "N/A", "notes": ""}},
            "corners": {"front": {}, "back": {}},
            "edges": {"front": {"grade": "N/A", "notes": ""}, "back": {"grade": "N/A", "notes": ""}},
            "surface": {"front": {"grade": "N/A", "notes": ""}, "back": {"grade": "N/A", "notes": ""}},
            "defects": [],
            "flags": [],
            "assessment_summary": "Assessment failed.",
            "observed_id": {"card_name": "", "set_code": "", "set_name": "", "card_number": "", "year": "", "card_type": ""},
            "error": "AI grading failed"
        })

    data = _parse_json_or_none(result.get("content", "")) or {}

    # Normalize observed ID with code map (but preserve printed number)
    obs = data.get("observed_id", {}) if isinstance(data.get("observed_id", {}), dict) else {}
    obs_name = _norm_ws(str(obs.get("card_name", "")))
    obs_type = _norm_ws(str(obs.get("card_type", "")))
    obs_year = _norm_ws(str(obs.get("year", "")))
    obs_num = _norm_printed_card_number(str(obs.get("card_number", "")))
    obs_code = _norm_ws(str(obs.get("set_code", ""))).upper()
    obs_set = _norm_ws(str(obs.get("set_name", "")))
    obs_set_info = _canonicalize_set(obs_code, obs_set)

    canonical_id = {
        "card_name": provided_name or obs_name,
        "card_type": provided_type or obs_type,
        "year": provided_year or obs_year,
        "card_number": provided_num or obs_num,
        "set_code": set_info["set_code"] or obs_set_info["set_code"],
        "set_name": set_info["set_name"] or obs_set_info["set_name"],
        "set_source": (set_info["set_source"] if (set_info["set_code"] or set_info["set_name"]) else obs_set_info["set_source"]),
        "confidence": _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0),
    }

    # Normalize pregrade to a simple "1"-"10" string for downstream DB consistency
    raw_pregrade = str(data.get("pregrade", "")).strip()
    m_pg = re.search(r"(10|[1-9])", raw_pregrade)
    pregrade_norm = m_pg.group(1) if m_pg else (str(_grade_bucket(raw_pregrade) or "") if raw_pregrade else "")

    return JSONResponse(content={
        "pregrade": pregrade_norm or "N/A",
        "confidence": _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0),
        "centering": data.get("centering", {"front": {"grade": "", "notes": ""}, "back": {"grade": "", "notes": ""}}),
        "corners": data.get("corners", {"front": {}, "back": {}}),
        "edges": data.get("edges", {"front": {"grade": "", "notes": ""}, "back": {"grade": "", "notes": ""}}),
        "surface": data.get("surface", {"front": {"grade": "", "notes": ""}, "back": {"grade": "", "notes": ""}}),
        "defects": data.get("defects", []) if isinstance(data.get("defects", []), list) else [],
        "flags": data.get("flags", []) if isinstance(data.get("flags", []), list) else [],
        "assessment_summary": _norm_ws(str(data.get("assessment_summary", ""))),
        "observed_id": {
            "card_name": obs_name,
            "card_type": obs_type,
            "year": obs_year,
            "card_number": obs_num,
            "set_code": obs_set_info["set_code"],
            "set_name": obs_set_info["set_name"],
            "set_source": obs_set_info["set_source"],
        },
        "canonical_id": canonical_id,
        "verify_token": f"vfy_{secrets.token_urlsafe(12)}",
        "market_context_mode": "click_only",
    })


# ==============================
# Market Context (Click-only, resilient)
# ==============================
_PRICE_RE = re.compile(r"(?P<cur>AU\$|A\$|US\$|\$)?\s*(?P<num>[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)")


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
    negate_terms = negate_terms or []
    q = query.strip()
    for t in negate_terms:
        if t:
            q += f" -{t}"
    encoded = q.replace(" ", "+")
    url = f"https://{EBAY_DOMAIN}/sch/i.html?_nkw={encoded}&_sacat=0&LH_Sold=1&LH_Complete=1&_sop=13"

    html = await _fetch_html(url)
    if not html:
        return {"prices": [], "currency": "", "sample_size": 0, "url": url, "query": query}

    soup = BeautifulSoup(html, "html.parser")
    price_spans = soup.find_all("span", class_=re.compile(r"s-item__price"))
    prices: List[float] = []
    currencies: List[str] = []

    for span in price_spans[: max(10, limit)]:
        txt = span.get_text(" ", strip=True)
        for val, cur in _extract_prices(txt):
            prices.append(val)
            currencies.append(cur)

    currency = ""
    if currencies:
        au_count = sum(1 for c in currencies if c in ("AU$", "A$"))
        us_count = sum(1 for c in currencies if c == "US$")
        if au_count >= max(1, len(currencies) // 3):
            currency = "AUD"
        elif us_count >= max(1, len(currencies) // 3):
            currency = "USD"

    prices = [p for p in prices if 1 < p < 500000][:limit]

    return {"prices": prices, "currency": currency, "sample_size": len(prices), "url": url, "query": query}


def _market_trend_from_recent(prices: List[float]) -> str:
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
    if n >= 2:
        return "very_thin"
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


def _build_query_ladder(card_name: str, set_name: str, set_code: str, card_number_printed: str, card_type: str) -> List[str]:
    """
    Build multiple query variants from strict -> relaxed.
    IMPORTANT: keep printed number as-is (leading zeros), because it helps match humans.
    """
    name = _norm_ws(card_name)
    sn = _norm_ws(set_name)
    sc = (set_code or "").strip().upper()
    num_printed = _norm_printed_card_number(card_number_printed)
    t = _norm_ws(card_type)

    variants: List[str] = []
    base_parts = [name]
    if sn and sn.lower() not in name.lower():
        base_parts.append(sn)
    if num_printed:
        base_parts.append(num_printed)
    variants.append(" ".join([p for p in base_parts if p]))

    if sc:
        parts = [name, sc]
        if num_printed:
            parts.append(num_printed)
        variants.append(" ".join(parts))

    if sn:
        variants.append(" ".join([name, sn]))

    if sc:
        variants.append(" ".join([name, sc]))

    if t and t.lower() not in name.lower():
        variants.append(" ".join([name, t]))
    variants.append(name)

    out: List[str] = []
    seen = set()
    for q in variants:
        qn = _norm_ws(q)
        if qn and qn.lower() not in seen:
            out.append(qn)
            seen.add(qn.lower())
    return out


async def _gather_market_samples(query_ladder: List[str]) -> Dict[str, Any]:
    """
    Try multiple query strings until we have at least some usable samples.
    Returns raw + graded samples and meta urls for transparency.
    """
    meta_urls: Dict[str, str] = {}
    used_query = ""
    raw_prices: List[float] = []
    raw_currency = ""
    graded_prices: Dict[str, List[float]] = {"10": [], "9": [], "8": []}

    target_raw = 3
    target_graded_each = 2

    if not EBAY_ENABLED:
        return {
            "used_query": "",
            "raw_prices": [],
            "raw_currency": "",
            "graded_prices": graded_prices,
            "meta_urls": meta_urls,
        }

    for q in query_ladder:
        used_query = q

        raw_res = await _search_ebay_sold_prices(q, negate_terms=["graded", "psa", "bgs", "cgc", "slab"], limit=30)
        meta_urls[f"ebay_raw::{q}"] = raw_res.get("url", "")
        if raw_res["prices"]:
            raw_prices = raw_res["prices"]
            raw_currency = raw_res.get("currency", "") or raw_currency

        for grade in ("10", "9", "8"):
            gr_res = await _search_ebay_sold_prices(f"{q} PSA {grade}", negate_terms=[], limit=20)
            meta_urls[f"ebay_psa_{grade}::{q}"] = gr_res.get("url", "")
            if gr_res["prices"]:
                graded_prices[grade] = gr_res["prices"]

        ok_raw = len(raw_prices) >= target_raw
        ok_graded = sum(1 for g in ("10", "9", "8") if len(graded_prices[g]) >= target_graded_each) >= 2
        if ok_raw or ok_graded:
            break

    return {
        "used_query": used_query,
        "raw_prices": raw_prices,
        "raw_currency": raw_currency,
        "graded_prices": graded_prices,
        "meta_urls": meta_urls,
    }


@app.post("/api/market-context")
async def market_context(
    card_name: str = Form(...),
    predicted_grade: str = Form(...),
    confidence: float = Form(0.0),
    card_number: Optional[str] = Form(None),
    card_set: Optional[str] = Form(None),
    set_code: Optional[str] = Form(None),
    card_type: Optional[str] = Form(None),
    grading_cost: float = Form(55.0),
):
    """
    Click-only market context endpoint.

    For Pokemon cards, this endpoint uses PokemonTCG.io to fetch:
    - authoritative card metadata (hi-res images, set info)
    - TCGplayer + Cardmarket price objects (when available)

    eBay sold/completed is optional and disabled for Pokemon by default.
    """
    clean_name = _norm_ws(card_name)
    clean_set = _norm_ws(card_set or "")
    clean_code = _norm_ws(set_code or "").upper()
    clean_number_printed = _norm_printed_card_number(card_number or "")
    clean_type = _norm_ws(card_type or "")

    set_info = _canonicalize_set(clean_code, clean_set)

    conf = _clamp(_safe_float(confidence, 0.0), 0.0, 1.0)
    g = _grade_bucket(predicted_grade) or 9

    is_pokemon = "pokemon" in clean_type.lower()

    query_ladder = _build_query_ladder(
        card_name=clean_name,
        set_name=set_info["set_name"],
        set_code=set_info["set_code"],
        card_number_printed=clean_number_printed,
        card_type=clean_type,
    )

    # Gather eBay samples via ladder (optional)
    gathered = {"used_query": "", "raw_prices": [], "raw_currency": "", "graded_prices": {"10": [], "9": [], "8": []}, "meta_urls": {}}
    if EBAY_ENABLED and (not is_pokemon or USE_EBAY_FOR_POKEMON):
        gathered = await _gather_market_samples(query_ladder)

    used_query = gathered["used_query"]
    raw_prices = gathered["raw_prices"]
    raw_currency = gathered["raw_currency"]
    graded_prices = gathered["graded_prices"]
    meta_urls = gathered["meta_urls"]

    sources: List[str] = []
    if raw_prices or any(graded_prices.get(k) for k in ("10", "9", "8")):
        sources.append("eBay (sold/completed)")

    # PokemonTCG authoritative details + price anchors (Pokemon only)
    pokemontcg_card: Optional[Dict[str, Any]] = None
    pokemontcg_id: str = ""
    ptcg_prices: Optional[Dict[str, Any]] = None
    anchor_prices: List[float] = []
    anchor_currencies: List[str] = []
    anchor_currency: str = ""

    if POKEMONTCG_API_KEY and is_pokemon:
        pokemontcg_id = await _pokemontcg_resolve_card_id(
            card_name=clean_name,
            set_code=set_info["set_code"],
            card_number_printed=clean_number_printed,
            set_name=set_info["set_name"],
        )
        if pokemontcg_id:
            pokemontcg_card = await _pokemontcg_card_by_id(pokemontcg_id)
            ptcg_prices = _pokemontcg_extract_prices(pokemontcg_card or {})
            anchor_prices = (ptcg_prices.get("anchor_values") or [])
            anchor_currencies = (ptcg_prices.get("currencies") or [])
            anchor_currency = anchor_currencies[0] if anchor_currencies else "USD"
            sources.append("PokemonTCG API (details + prices)")

    # Decide availability: if ANY samples exist, return available with liquidity label
    has_any = bool(raw_prices) or any(len(v) > 0 for v in graded_prices.values()) or bool(anchor_prices) or bool(pokemontcg_card)
    if not has_any:
        return JSONResponse(content={
            "available": False,
            "mode": "click_only",
            "card": {
                "name": clean_name,
                "set": set_info["set_name"],
                "set_code": set_info["set_code"],
                "number": clean_number_printed,
                "type": clean_type,
            },
            "query_ladder": query_ladder,
            "used_query": used_query,
            "message": "No market samples found from the available sources for this item. Try adjusting the set name/code or card number.",
            "sources": sources,
            "pokemontcg": {"id": pokemontcg_id} if pokemontcg_id else None,
            "meta": {"urls": meta_urls},
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "disclaimer": "Informational market context only. Not financial advice."
        })

    raw_stats = _stats(raw_prices)
    anchor_stats = _stats(anchor_prices)
    graded_stats: Dict[str, Any] = {k: _stats(v) for k, v in graded_prices.items() if v}

    raw_baseline = 0.0
    if raw_stats["sample_size"] >= 2:
        raw_baseline = raw_stats["median"] or raw_stats["avg"]
    elif anchor_stats["sample_size"] >= 1:
        raw_baseline = anchor_stats["median"] or anchor_stats["avg"]

    trend = _market_trend_from_recent(raw_prices) if raw_prices else "insufficient_data"
    liquidity = _liquidity_label(max(raw_stats["sample_size"], anchor_stats["sample_size"]))

    dist = _grade_distribution(g, conf)
    expected_value = None

    if graded_stats:
        expected_value = 0.0
        for grade, p in dist.items():
            st = graded_stats.get(grade)
            if not st:
                continue
            val = (st["median"] if st["sample_size"] < 10 else st["avg"]) or st["avg"] or 0.0
            expected_value += p * val
        expected_value = round(expected_value, 2) if expected_value > 0 else None

    value_difference = None
    if expected_value is not None and raw_baseline:
        value_difference = round(expected_value - raw_baseline - float(grading_cost or 0.0), 2)

    sensitivity = "unknown"
    if "10" in graded_stats and "9" in graded_stats and graded_stats["10"]["sample_size"] >= 2 and graded_stats["9"]["sample_size"] >= 2:
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

    currency = raw_currency or (anchor_currency if anchor_prices else "") or "unknown"

    sample_total = raw_stats["sample_size"] + sum(st["sample_size"] for st in graded_stats.values()) + anchor_stats["sample_size"]
    confidence_label = "high" if sample_total >= 20 else "moderate" if sample_total >= 8 else "low"

    pokemontcg_out = None
    if pokemontcg_id and pokemontcg_card:
        pokemontcg_out = {
            "id": pokemontcg_id,
            "name": pokemontcg_card.get("name", ""),
            "number": pokemontcg_card.get("number", ""),
            "rarity": pokemontcg_card.get("rarity", ""),
            "set": pokemontcg_card.get("set", {}) or {},
            "images": pokemontcg_card.get("images", {}) or {},
            "prices": ptcg_prices,
        }

    return JSONResponse(content={
        "available": True,
        "mode": "click_only",
        "confidence": confidence_label,
        "card": {
            "name": clean_name,
            "set": set_info["set_name"],
            "set_code": set_info["set_code"],
            "number": clean_number_printed,
            "type": clean_type,
        },
        "query_ladder": query_ladder,
        "used_query": used_query,
        "observed": {
            "currency": currency,
            "raw": raw_stats,
            "pokemon_prices_anchor": anchor_stats if anchor_stats["sample_size"] else None,
            "graded_psa": graded_stats,
            "trend": trend,
            "liquidity": liquidity,
        },
        "grade_impact": {
            "predicted_grade": str(g),
            "confidence_input": conf,
            "grade_distribution": dist,
            "expected_graded_value": expected_value,
            "raw_baseline_value": round(raw_baseline, 2) if raw_baseline else None,
            "grading_cost": round(float(grading_cost or 0.0), 2),
            "estimated_value_difference": value_difference,
            "sensitivity": sensitivity,
        },
        "pokemontcg": pokemontcg_out,
        "sources": sources,
        "meta": {"urls": meta_urls},
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "disclaimer": "Informational market context only. Figures are based on observed recent sales and/or third‑party price anchors and may vary. This is not financial advice or a guarantee of outcome."
    })


# Legacy alias
@app.post("/api/market-intelligence")
async def market_intelligence_alias(
    card_name: str = Form(...),
    predicted_grade: str = Form(...),
    confidence: float = Form(0.0),
    card_number: Optional[str] = Form(None),
    card_set: Optional[str] = Form(None),
    set_code: Optional[str] = Form(None),
    card_type: Optional[str] = Form(None),
    grading_cost: float = Form(55.0),
):
    res = await market_context(
        card_name=card_name,
        predicted_grade=predicted_grade,
        confidence=confidence,
        card_number=card_number,
        card_set=card_set,
        set_code=set_code,
        card_type=card_type,
        grading_cost=grading_cost,
    )
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
