"""
The Collectors League Australia — Scan API
Futureproof v6.2.2 (2026-01-31)

Key fixes in this build:
- ✅ Correct CORS for collectors-league.com (no wildcard+credentials mismatch)
- ✅ Preserve leading zeros in card numbers for UI (e.g., 006/165)
- ✅ Resolve set name from PokemonTCG using set.ptcgoCode when AI leaves set_name blank
- ✅ Use PokemonTCG (TCGplayer + Cardmarket) as primary market source; eBay scraping optional via env USE_EBAY=1
- ✅ Stronger PokemonTCG card resolver using all known inputs (name, number, set code/name)
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

# IMPORTANT:
# - Browsers reject allow_credentials=True with allow_origins=["*"].
# - We do NOT need cookies for this API, so keep credentials OFF and use explicit origins.
ALLOWED_ORIGINS = [
    "https://collectors-league.com",
    "https://www.collectors-league.com",
    # optional local dev:
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

APP_VERSION = os.getenv("CL_SCAN_VERSION", "2026-01-31-v6.2.2")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
POKEMONTCG_API_KEY = os.getenv("POKEMONTCG_API_KEY", "").strip()

POKEMONTCG_BASE = "https://api.pokemontcg.io/v2"

# eBay scraping is OPTIONAL (often rate limited). Default OFF.
USE_EBAY = os.getenv("USE_EBAY", "0").strip() in ("1", "true", "TRUE", "yes", "YES")

# If enabled, choose AU by default
EBAY_DOMAIN = os.getenv("EBAY_DOMAIN", "www.ebay.com.au").strip() or "www.ebay.com.au"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set!")
if not POKEMONTCG_API_KEY:
    print("WARNING: POKEMONTCG_API_KEY not set! PokemonTCG enrichment will be disabled.")

# ==============================
# Set Code Mapping (expand over time)
# ==============================
# Only include codes you are sure of. PokemonTCG uses set.ptcgoCode for many modern sets (e.g., MEW for SV 151).
SET_CODE_MAP: Dict[str, str] = {
    # examples
    "PFL": "Phantasmal Flames",
    "OBF": "Obsidian Flames",
    "SVI": "Scarlet & Violet",
    "MEW": "Scarlet & Violet—151",  # common
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


def _clean_card_number_display(s: str) -> str:
    """
    Preserve leading zeros for UI.
    Examples:
      "006/165" -> "006/165"
      "#006/165" -> "006/165"
      "006" -> "006"
    """
    s = (s or "").strip()
    s = s.replace("#", "").strip()
    if not s:
        return ""
    # normalize whitespace only; do NOT strip leading zeros
    s = re.sub(r"\s+", "", s)
    return s


def _card_number_for_query(s: str) -> str:
    """
    Normalize for PokemonTCG searches/comparisons:
    - Strip leading zeros from the FIRST part only.
    Examples:
      "006/165" -> "6/165"
      "006" -> "6"
      "0/165" -> "0/165"
    """
    s = _clean_card_number_display(s)
    if not s:
        return ""
    if "/" in s:
        a, b = s.split("/", 1)
        a2 = a.lstrip("0") or "0"
        return f"{a2}/{b}"
    return s.lstrip("0") or "0"


def _same_number(a: str, b: str) -> bool:
    """Compare card numbers ignoring leading zeros on the first segment."""
    return _card_number_for_query(a) == _card_number_for_query(b)


async def _openai_chat(messages: List[Dict[str, Any]], max_tokens: int = 1200, temperature: float = 0.1) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    url = "https://api.openai.com/v1/chat/completions"
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code != 200:
                return {"error": True, "status": r.status_code, "message": r.text[:600]}
            data = r.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return {"error": False, "content": content}
    except Exception as e:
        return {"error": True, "status": 0, "message": str(e)}


def _grade_bucket(predicted_grade: str) -> Optional[int]:
    m = re.search(r"(10|[1-9])", (predicted_grade or "").strip())
    if not m:
        return None
    g = int(m.group(1))
    return g if 1 <= g <= 10 else None


def _grade_distribution(predicted_grade: int, confidence: float) -> Dict[str, float]:
    c = _clamp(confidence, 0.05, 0.95)
    p_pred = 0.45 + 0.50 * c
    remainder = 1.0 - p_pred

    if predicted_grade >= 10:
        dist = {"10": p_pred, "9": remainder * 0.75, "8": remainder * 0.25}
    elif predicted_grade == 9:
        dist = {"10": remainder * 0.25, "9": p_pred, "8": remainder * 0.75}
    else:
        dist = {"10": remainder * 0.10, "9": remainder * 0.30, "8": p_pred + remainder * 0.60}

    total = sum(dist.values()) or 1.0
    return {k: round(v / total, 4) for k, v in dist.items()}


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
        return {"set_code": sc or "", "set_name": sn, "set_source": "ai_name_only"}
    if sc:
        return {"set_code": sc, "set_name": "", "set_source": "code_only"}
    return {"set_code": "", "set_name": "", "set_source": "unknown"}

# ==============================
# PokemonTCG helpers (authoritative metadata + prices)
# ==============================
async def _pokemontcg_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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


async def _pokemontcg_resolve_set_by_ptcgo(ptcgo: str) -> Dict[str, Any]:
    """
    Resolve set object from set.ptcgoCode (e.g., MEW).
    Uses /sets endpoint: GET /v2/sets?q=ptcgoCode:MEW
    """
    ptcgo = (ptcgo or "").strip().upper()
    if not ptcgo or not POKEMONTCG_API_KEY:
        return {}
    data = await _pokemontcg_get("/sets", params={"q": f"ptcgoCode:{ptcgo}", "pageSize": 5})
    sets = data.get("data") or []
    return sets[0] if sets else {}


async def _pokemontcg_resolve_card_id(card_name: str, set_code: str, card_number_display: str, set_name: str = "", set_id: str = "") -> str:
    """
    Resolve most likely PokemonTCG card id using a strict->relaxed ladder.
    Uses ALL known inputs:
      - exact name (quoted)
      - number (normalized for query)
      - set.ptcgoCode, set.name, set.id when available
    """
    name = _norm_ws(card_name)
    sc = (set_code or "").strip().upper()
    num_q = _card_number_for_query(card_number_display or "")
    sn = _norm_ws(set_name or "")
    sid = (set_id or "").strip()

    queries: List[str] = []

    # strict
    if name and num_q and sid:
        queries.append(f'name:"{name}" number:{num_q.split("/")[0]} set.id:{sid}')
    if num_q and sid:
        queries.append(f'number:{num_q.split("/")[0]} set.id:{sid}')
    if name and num_q and sc:
        queries.append(f'name:"{name}" number:{num_q.split("/")[0]} set.ptcgoCode:{sc}')
    if num_q and sc:
        queries.append(f'number:{num_q.split("/")[0]} set.ptcgoCode:{sc}')
    if name and num_q and sn:
        queries.append(f'name:"{name}" number:{num_q.split("/")[0]} set.name:"{sn}"')

    # relaxed
    if name and sn:
        queries.append(f'name:"{name}" set.name:"{sn}"')
    if name and sc:
        queries.append(f'name:"{name}" set.ptcgoCode:{sc}')
    if name:
        queries.append(f'name:"{name}"')
    if sc:
        queries.append(f"set.ptcgoCode:{sc}")

    for q in queries:
        data = await _pokemontcg_get("/cards", params={"q": q, "pageSize": 25})
        cards = (data.get("data") or [])
        if not cards:
            continue

        # If number given, prefer exact number match ignoring leading zeros
        if card_number_display:
            for c in cards:
                if _same_number(str(c.get("number", "")), num_q.split("/")[0]):
                    return str(c.get("id", "")) or ""
        return str(cards[0].get("id", "")) or ""

    return ""


async def _pokemontcg_card_by_id(card_id: str) -> Dict[str, Any]:
    if not card_id:
        return {}
    data = await _pokemontcg_get(f"/cards/{card_id}")
    card = data.get("data") if isinstance(data, dict) else None
    return card if isinstance(card, dict) else {}


def _pokemontcg_extract_prices(card: Dict[str, Any]) -> Dict[str, Any]:
    tcg = (card or {}).get("tcgplayer", {}) or {}
    cm = (card or {}).get("cardmarket", {}) or {}

    tcg_prices = tcg.get("prices") if isinstance(tcg, dict) else None
    cm_prices = cm.get("prices") if isinstance(cm, dict) else None

    anchor: List[float] = []
    currencies: List[str] = []

    if isinstance(tcg_prices, dict) and tcg_prices:
        currencies.append("USD")
        for _, obj in tcg_prices.items():
            if isinstance(obj, dict) and obj.get("market") is not None:
                try:
                    anchor.append(float(obj["market"]))
                except Exception:
                    pass

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
    return {"status": "ok", "service": "cl-scan-api", "version": APP_VERSION}

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "cl-scan-api",
        "version": APP_VERSION,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_pokemontcg_key": bool(POKEMONTCG_API_KEY),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "use_ebay": USE_EBAY,
        "allowed_origins": ALLOWED_ORIGINS,
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
    image_bytes = await front.read()
    if not image_bytes or len(image_bytes) < 200:
        raise HTTPException(status_code=400, detail="Image is too small or empty")

    prompt = """You are identifying a trading card from a front photo.

Return ONLY valid JSON with these exact fields:

{
  "card_name": "exact card name including variants (ex, V, VMAX, holo, first edition, etc.)",
  "card_type": "Pokemon/Magic/YuGiOh/Sports/OnePiece/Other",
  "year": "4 digit year if visible else empty string",
  "card_number": "card number if visible (e.g. 006/165 or 004 or 4/102) else empty string",
  "set_code": "set abbreviation printed on the card if visible (often 2-4 chars like MEW, OBF, SVI). EMPTY if not visible",
  "set_name": "set or series name (full name) if visible/known else empty string",
  "confidence": 0.0-1.0,
  "notes": "one short sentence about how you identified it"
}

Rules:
- PRIORITIZE set_code if you can see it.
- Do NOT guess a set_code you cannot see.
- Preserve leading zeros in card_number if shown (e.g., 006/165).
- If you cannot identify with confidence, set card_name to "Unknown" and confidence to 0.0.
Respond ONLY with JSON, no extra text.
"""

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
    card_number_display = _clean_card_number_display(str(data.get("card_number", "")))
    set_code = _norm_ws(str(data.get("set_code", ""))).upper()
    set_name = _norm_ws(str(data.get("set_name", "")))
    conf = _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0)
    notes = _norm_ws(str(data.get("notes", "")))

    # If AI didn't provide set_name but did provide set_code, fetch set name from PokemonTCG sets endpoint
    ptcg_set = {}
    if POKEMONTCG_API_KEY and set_code and not set_name:
        ptcg_set = await _pokemontcg_resolve_set_by_ptcgo(set_code)
        if ptcg_set.get("name"):
            set_name = _norm_ws(str(ptcg_set.get("name", "")))

    set_info = _canonicalize_set(set_code, set_name)

    canonical_id: Dict[str, Any] = {
        "card_name": card_name,
        "card_type": card_type,
        "year": year,
        "card_number": card_number_display,              # display-safe
        "card_number_norm": _card_number_for_query(card_number_display),  # query-safe
        "set_code": set_info["set_code"],
        "set_name": set_info["set_name"],
        "set_source": set_info["set_source"],
        "confidence": conf,
    }

    pokemontcg_payload = None
    if POKEMONTCG_API_KEY and card_type.strip().lower() == "pokemon":
        pid = await _pokemontcg_resolve_card_id(
            card_name=card_name,
            set_code=set_info["set_code"],
            card_number_display=card_number_display,
            set_name=set_info["set_name"],
            set_id=str(ptcg_set.get("id", "")) if ptcg_set else "",
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
                "prices": _pokemontcg_extract_prices(card or {}),
                "links": {
                    "tcgplayer": (card.get("tcgplayer", {}) or {}).get("url", ""),
                    "cardmarket": (card.get("cardmarket", {}) or {}).get("url", ""),
                }
            }
            # If set_name still empty, backfill from card.set.name
            if not canonical_id.get("set_name") and (card.get("set") or {}).get("name"):
                canonical_id["set_name"] = _norm_ws(str((card.get("set") or {}).get("name", "")))
                canonical_id["set_source"] = "pokemontcg_card"

    return JSONResponse(content={
        "card_name": card_name,
        "card_type": card_type,
        "year": year,
        "card_number": card_number_display,
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
    front_bytes = await front.read()
    back_bytes = await back.read()

    if (not front_bytes or not back_bytes or len(front_bytes) < 200 or len(back_bytes) < 200):
        raise HTTPException(status_code=400, detail="Images are too small or empty")

    provided_name = _norm_ws(card_name or "")
    provided_set = _norm_ws(card_set or "")
    provided_num_display = _clean_card_number_display(card_number or "")
    provided_year = _norm_ws(card_year or "")
    provided_type = _norm_ws(card_type or "")
    provided_code = _norm_ws(set_code or "").upper()

    # Backfill set_name from PokemonTCG sets endpoint if needed
    ptcg_set = {}
    if POKEMONTCG_API_KEY and provided_code and not provided_set:
        ptcg_set = await _pokemontcg_resolve_set_by_ptcgo(provided_code)
        if ptcg_set.get("name"):
            provided_set = _norm_ws(str(ptcg_set.get("name", "")))

    set_info = _canonicalize_set(provided_code, provided_set)

    context = ""
    if provided_name or set_info["set_name"] or provided_num_display or provided_year or provided_type or set_info["set_code"]:
        context = "\n\nKNOWN/PROVIDED CARD DETAILS (use as hints, do not force if images contradict):\n"
        if provided_name:
            context += f"- Card Name: {provided_name}\n"
        if set_info["set_name"]:
            context += f"- Set Name: {set_info['set_name']}\n"
        if set_info["set_code"]:
            context += f"- Set Code: {set_info['set_code']}\n"
        if provided_num_display:
            context += f"- Card Number: {provided_num_display}\n"
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
- A clear Assessment Summary (2-4 sentences) mentioning specific issues you see.

{context}

Return JSON with this EXACT structure:

{{
  "pregrade": "estimated PSA-style grade 1-10 (e.g. 8, 9, 10)",
  "confidence": 0.0-1.0,
  "centering": {{
    "front": {{"grade":"55/45","notes":"..."}} ,
    "back":  {{"grade":"60/40","notes":"..."}}
  }},
  "corners": {{
    "front": {{
      "top_left": {{"condition":"sharp/minor_whitening/whitening/bend/ding","notes":"..."}} ,
      "top_right": {{"condition":"...","notes":"..."}} ,
      "bottom_left": {{"condition":"...","notes":"..."}} ,
      "bottom_right": {{"condition":"...","notes":"..."}}
    }},
    "back": {{
      "top_left": {{"condition":"...","notes":"..."}} ,
      "top_right": {{"condition":"...","notes":"..."}} ,
      "bottom_left": {{"condition":"...","notes":"..."}} ,
      "bottom_right": {{"condition":"...","notes":"..."}}
    }}
  }},
  "edges": {{
    "front": {{"grade":"Mint/Near Mint/Excellent/Good/Poor","notes":"detailed notes"}} ,
    "back":  {{"grade":"Mint/Near Mint/Excellent/Good/Poor","notes":"detailed notes"}}
  }},
  "surface": {{
    "front": {{"grade":"Mint/Near Mint/Excellent/Good/Poor","notes":"detailed notes"}} ,
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
    "card_number": "best-effort from images (preserve leading zeros if shown)",
    "year": "best-effort from images",
    "card_type": "Pokemon/Magic/YuGiOh/Sports/OnePiece/Other"
  }}
}}

Important:
- If something cannot be determined, use empty string "" or empty arrays.
- Do not include market values in this endpoint.
Respond ONLY with JSON.
"""

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

    obs = data.get("observed_id", {}) if isinstance(data.get("observed_id", {}), dict) else {}
    obs_name = _norm_ws(str(obs.get("card_name", "")))
    obs_type = _norm_ws(str(obs.get("card_type", "")))
    obs_year = _norm_ws(str(obs.get("year", "")))
    obs_num_display = _clean_card_number_display(str(obs.get("card_number", "")))
    obs_code = _norm_ws(str(obs.get("set_code", ""))).upper()
    obs_set = _norm_ws(str(obs.get("set_name", "")))

    # If observed set is missing but code exists, backfill from PokemonTCG
    obs_ptcg_set = {}
    if POKEMONTCG_API_KEY and obs_code and not obs_set:
        obs_ptcg_set = await _pokemontcg_resolve_set_by_ptcgo(obs_code)
        if obs_ptcg_set.get("name"):
            obs_set = _norm_ws(str(obs_ptcg_set.get("name", "")))

    obs_set_info = _canonicalize_set(obs_code, obs_set)

    canonical_id = {
        "card_name": provided_name or obs_name,
        "card_type": provided_type or obs_type,
        "year": provided_year or obs_year,
        "card_number": provided_num_display or obs_num_display,
        "card_number_norm": _card_number_for_query(provided_num_display or obs_num_display),
        "set_code": set_info["set_code"] or obs_set_info["set_code"],
        "set_name": set_info["set_name"] or obs_set_info["set_name"],
        "set_source": (set_info["set_source"] if (set_info["set_code"] or set_info["set_name"]) else obs_set_info["set_source"]),
        "confidence": _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0),
    }

    raw_pregrade = str(data.get("pregrade", "")).strip()
    g = _grade_bucket(raw_pregrade)
    pregrade_norm = str(g) if g is not None else ""

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
            "card_number": obs_num_display,
            "set_code": obs_set_info["set_code"],
            "set_name": obs_set_info["set_name"],
            "set_source": obs_set_info["set_source"],
        },
        "canonical_id": canonical_id,
        "verify_token": f"vfy_{secrets.token_urlsafe(12)}",
        "market_context_mode": "click_only",
    })

# ==============================
# Market Context (Click-only)
# - PokemonTCG is PRIMARY for Pokemon cards
# - eBay scraping is OPTIONAL (USE_EBAY=1)
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

def _liquidity_label(n: int) -> str:
    if n >= 25: return "high"
    if n >= 10: return "moderate"
    if n >= 5:  return "thin"
    if n >= 2:  return "very_thin"
    return "insufficient"

async def _fetch_html(url: str) -> str:
    async with httpx.AsyncClient(timeout=20.0) as client:
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
        if au_count >= max(1, len(currencies) // 3): currency = "AUD"
        elif us_count >= max(1, len(currencies) // 3): currency = "USD"

    prices = [p for p in prices if 1 < p < 500000][:limit]
    return {"prices": prices, "currency": currency, "sample_size": len(prices), "url": url, "query": query}

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
    clean_name = _norm_ws(card_name)
    clean_set = _norm_ws(card_set or "")
    clean_code = _norm_ws(set_code or "").upper()
    clean_number_display = _clean_card_number_display(card_number or "")
    clean_type = _norm_ws(card_type or "")

    # Backfill set name using PokemonTCG sets endpoint if code exists
    ptcg_set = {}
    if POKEMONTCG_API_KEY and clean_code and not clean_set:
        ptcg_set = await _pokemontcg_resolve_set_by_ptcgo(clean_code)
        if ptcg_set.get("name"):
            clean_set = _norm_ws(str(ptcg_set.get("name", "")))

    set_info = _canonicalize_set(clean_code, clean_set)

    conf = _clamp(_safe_float(confidence, 0.0), 0.0, 1.0)
    g = _grade_bucket(predicted_grade) or 9
    dist = _grade_distribution(g, conf)

    sources: List[str] = []
    meta_urls: Dict[str, str] = {}

    # 1) PokemonTCG first (FAST, reliable)
    pokemontcg_out = None
    ptcg_prices = None
    anchor_prices: List[float] = []
    anchor_stats = {"avg": 0, "median": 0, "low": 0, "high": 0, "sample_size": 0}
    currency = "unknown"

    if POKEMONTCG_API_KEY and ("pokemon" in clean_type.lower()):
        pid = await _pokemontcg_resolve_card_id(
            card_name=clean_name,
            set_code=set_info["set_code"],
            card_number_display=clean_number_display,
            set_name=set_info["set_name"],
            set_id=str(ptcg_set.get("id", "")) if ptcg_set else "",
        )
        if pid:
            card = await _pokemontcg_card_by_id(pid)
            ptcg_prices = _pokemontcg_extract_prices(card or {})
            anchor_prices = (ptcg_prices.get("anchor_values") or [])
            anchor_stats = _stats(anchor_prices)
            currency = (ptcg_prices.get("currencies") or ["unknown"])[0]
            sources.append("PokemonTCG API (TCGplayer/Cardmarket)")

            pokemontcg_out = {
                "id": pid,
                "name": card.get("name", ""),
                "number": card.get("number", ""),
                "rarity": card.get("rarity", ""),
                "set": card.get("set", {}) or {},
                "images": card.get("images", {}) or {},
                "prices": ptcg_prices,
            }

    # 2) Optional eBay (slow / rate-limited) — only if enabled
    raw_prices: List[float] = []
    raw_currency = ""
    if USE_EBAY:
        q_bits = [clean_name]
        if set_info["set_name"] and set_info["set_name"].lower() not in clean_name.lower():
            q_bits.append(set_info["set_name"])
        if clean_number_display:
            q_bits.append(clean_number_display)
        query = " ".join([b for b in q_bits if b]).strip()

        raw_res = await _search_ebay_sold_prices(query, negate_terms=["graded", "psa", "bgs", "cgc", "slab"], limit=25)
        meta_urls["ebay_raw"] = raw_res.get("url", "")
        raw_prices = raw_res.get("prices", []) or []
        raw_currency = raw_res.get("currency", "") or ""
        if raw_prices:
            sources.append("eBay (sold/completed)")

    has_any = bool(anchor_prices) or bool(raw_prices)
    if not has_any:
        return JSONResponse(content={
            "available": False,
            "mode": "click_only",
            "card": {
                "name": clean_name,
                "set": set_info["set_name"],
                "set_code": set_info["set_code"],
                "number": clean_number_display,
                "type": clean_type,
            },
            "message": "No market data available from PokemonTCG (and eBay is disabled or returned no samples).",
            "sources": sources,
            "pokemontcg": pokemontcg_out,
            "meta": {"urls": meta_urls},
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "disclaimer": "Informational market context only. Not financial advice."
        })

    raw_stats = _stats(raw_prices)
    liquidity = _liquidity_label(max(raw_stats["sample_size"], anchor_stats["sample_size"]))
    currency = raw_currency or currency

    # baseline: prefer PokemonTCG median if available, else eBay median
    baseline = 0.0
    if anchor_stats["sample_size"] >= 1:
        baseline = anchor_stats["median"] or anchor_stats["avg"]
    elif raw_stats["sample_size"] >= 1:
        baseline = raw_stats["median"] or raw_stats["avg"]

    expected_value = None
    # we do not model graded premiums from PokemonTCG; this remains informational
    value_difference = None
    if baseline and grading_cost:
        value_difference = round(baseline - float(grading_cost or 0.0), 2)

    sample_total = raw_stats["sample_size"] + anchor_stats["sample_size"]
    confidence_label = "high" if sample_total >= 20 else "moderate" if sample_total >= 8 else "low"

    return JSONResponse(content={
        "available": True,
        "mode": "click_only",
        "confidence": confidence_label,
        "card": {
            "name": clean_name,
            "set": set_info["set_name"],
            "set_code": set_info["set_code"],
            "number": clean_number_display,
            "type": clean_type,
        },
        "observed": {
            "currency": currency,
            "pokemon_prices_anchor": anchor_stats if anchor_stats["sample_size"] else None,
            "ebay_raw": raw_stats if raw_stats["sample_size"] else None,
            "liquidity": liquidity,
        },
        "grade_impact": {
            "predicted_grade": str(g),
            "confidence_input": conf,
            "grade_distribution": dist,
            "baseline_value": round(baseline, 2) if baseline else None,
            "grading_cost": round(float(grading_cost or 0.0), 2),
            "baseline_minus_grading_cost": value_difference,
        },
        "pokemontcg": pokemontcg_out,
        "sources": sources,
        "meta": {"urls": meta_urls, "use_ebay": USE_EBAY},
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "disclaimer": "Informational market context only. Figures are third‑party anchors and/or observed sales and may vary. Not financial advice."
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
    return await market_context(
        card_name=card_name,
        predicted_grade=predicted_grade,
        confidence=confidence,
        card_number=card_number,
        card_set=card_set,
        set_code=set_code,
        card_type=card_type,
        grading_cost=grading_cost,
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
