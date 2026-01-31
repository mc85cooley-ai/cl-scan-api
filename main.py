"""
The Collectors League Australia — Scan API
Futureproof v6.2.2 (2026-01-31)

Key fixes in this build:
- CORS fixed (single middleware, configurable origins; works with WordPress fetch)
- PokémonTCG.io integration strengthened:
  - Resolve SET from set code via /v2/sets?q=ptcgoCode:XXX
  - Resolve CARD via /v2/cards using set.id + number (handles leading zeros) + name ladder
  - If set_name is missing, it is filled from PokémonTCG data (authoritative)
- Card number handling:
  - Keeps leading zeros for display ("006/165")
  - Uses BOTH "006" and "6" variants when querying PokémonTCG
- Market context:
  - PokémonTCG price anchors (TCGplayer/Cardmarket via PokémonTCG API) are the primary source
  - eBay scraping is OPTIONAL and OFF by default (ENABLE_EBAY=1 to turn on)

Dependencies (requirements.txt):
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

APP_VERSION = os.getenv("CL_SCAN_VERSION", "2026-01-31-v6.2.2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
POKEMONTCG_API_KEY = os.getenv("POKEMONTCG_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

POKEMONTCG_BASE = "https://api.pokemontcg.io/v2"

# Market sources toggles
ENABLE_EBAY = os.getenv("ENABLE_EBAY", "0").strip() in ("1", "true", "TRUE", "yes", "YES")

# eBay region: AU by default (only if ENABLE_EBAY=1)
EBAY_DOMAIN = os.getenv("EBAY_DOMAIN", "www.ebay.com.au").strip() or "www.ebay.com.au"

UA = os.getenv(
    "UA",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# ---- CORS ----
# WordPress fetch does NOT need credentials here, so keep allow_credentials=False.
# Provide allowed origins via env CL_ALLOWED_ORIGINS (comma-separated).
# Default to your domains + localhost for testing; fallback to "*" if you prefer.
_default_origins = [
    "https://collectors-league.com",
    "https://www.collectors-league.com",
    "http://localhost",
    "http://localhost:3000",
]
origins_env = os.getenv("CL_ALLOWED_ORIGINS", "").strip()
if origins_env:
    allowed_origins = [o.strip() for o in origins_env.split(",") if o.strip()]
else:
    allowed_origins = _default_origins

# If you really want wildcard CORS (not recommended generally), set CL_CORS_WILDCARD=1
if os.getenv("CL_CORS_WILDCARD", "0").strip() in ("1", "true", "TRUE", "yes", "YES"):
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set!")
if not POKEMONTCG_API_KEY:
    print("WARNING: POKEMONTCG_API_KEY not set! (Pokémon lookups will be skipped)")

# ==============================
# Set Code Mapping (optional overrides)
# ==============================
# You can keep this small. PokémonTCG lookup should now fill set_name anyway.
SET_CODE_MAP: Dict[str, str] = {
    # Examples only (safe to delete if you like)
    "OBF": "Obsidian Flames",
    "SVI": "Scarlet & Violet",
    "BS": "Base Set",
    # NOTE: MEW is a PTCGO code for Pokémon 151; API will now resolve it via /sets.
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


def _num_parts(raw: str) -> Tuple[str, str]:
    """
    Split number like '006/165' into ('006', '165') and keep formatting.
    If no '/', returns (raw, '').
    """
    s = (raw or "").strip().replace("#", "").strip()
    if "/" in s:
        a, b = s.split("/", 1)
        return (a.strip(), b.strip())
    return (s, "")


def _num_display(raw: str) -> str:
    """Keep leading zeros for display. '006/165' stays '006/165'."""
    a, b = _num_parts(raw)
    if b:
        return f"{a}/{b}"
    return a


def _num_strip_leading_zeros(raw: str) -> str:
    """
    Strip leading zeros only from the FIRST part for matching on some data sources.
    '006/165' -> '6/165', '004' -> '4'
    """
    a, b = _num_parts(raw)
    a2 = a.lstrip("0") or "0"
    if b:
        return f"{a2}/{b}"
    return a2


def _num_variants(raw: str) -> List[str]:
    """
    Provide both display and normalized variants to maximize match probability.
    Example: '006/165' -> ['006/165','6/165','006','6']
    """
    disp = _num_display(raw)
    norm = _num_strip_leading_zeros(raw)
    a_disp, b_disp = _num_parts(disp)
    a_norm, b_norm = _num_parts(norm)
    variants = []
    for v in (disp, norm):
        if v and v not in variants:
            variants.append(v)
    # also include just the first part variants
    for v in (a_disp, a_norm):
        if v and v not in variants:
            variants.append(v)
    return variants


async def _openai_chat(messages: List[Dict[str, Any]], max_tokens: int = 1200, temperature: float = 0.1) -> Dict[str, Any]:
    """
    OpenAI Chat Completions call for text+image.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}

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
    m = re.search(r"(10|[1-9])", (predicted_grade or "").strip())
    if not m:
        return None
    g = int(m.group(1))
    if 1 <= g <= 10:
        return g
    return None


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


async def _pokemontcg_resolve_set_by_code(ptcgo_code: str) -> Dict[str, Any]:
    """
    Resolve set object using /sets endpoint from ptcgoCode.
    Returns dict: {"id","name","series","ptcgoCode","printedTotal","total"} (best-effort).
    """
    code = (ptcgo_code or "").strip().upper()
    if not code:
        return {}
    data = await _pokemontcg_get("/sets", params={"q": f"ptcgoCode:{code}", "pageSize": 5})
    sets = data.get("data") or []
    if not sets:
        return {}
    # Prefer exact ptcgoCode match if multiple
    for s in sets:
        if str(s.get("ptcgoCode", "")).strip().upper() == code:
            return s
    return sets[0]


async def _pokemontcg_resolve_set_by_name(set_name: str) -> Dict[str, Any]:
    name = _norm_ws(set_name)
    if not name:
        return {}
    # Exact phrase match first, then relaxed
    for q in (f'set.name:"{name}"', f'set.name:{name}'):
        data = await _pokemontcg_get("/sets", params={"q": q, "pageSize": 5})
        sets = data.get("data") or []
        if sets:
            return sets[0]
    return {}


def _ptcg_escape_phrase(s: str) -> str:
    # very small sanitizer to avoid breaking the q string; PokémonTCG is fairly forgiving
    return (s or "").replace('"', '\\"')


async def _pokemontcg_resolve_card_id(
    card_name: str,
    set_code: str,
    card_number: str,
    set_name: str = ""
) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve most likely PokemonTCG card id using a query ladder.
    Returns (card_id, best_set_obj).
    Strategy:
    1) Resolve set via ptcgoCode (set_code) -> set.id
    2) Query cards with set.id + number variants (handles leading zeros)
    3) If that fails, query with set.name + number/name
    4) Fallback to name-only
    """
    name = _norm_ws(card_name)
    sc = (set_code or "").strip().upper()
    sn = _norm_ws(set_name or "")
    num_disp = _num_display(card_number or "")
    num_variants = _num_variants(card_number or "")

    set_obj = {}
    set_id = ""

    if sc:
        set_obj = await _pokemontcg_resolve_set_by_code(sc)
        set_id = str(set_obj.get("id", "") or "").strip()

    # If we don't have set from code but do have set name, try resolve it too
    if not set_id and sn:
        set_obj = await _pokemontcg_resolve_set_by_name(sn)
        set_id = str(set_obj.get("id", "") or "").strip()

    queries: List[str] = []

    # ---- Strict: set.id + number (try both "006" and "6") ----
    if set_id and num_variants:
        # Build OR clause for number variants. Use quoted values to preserve leading zeros.
        # Example: (number:"006" OR number:"6" OR number:"006/165" OR number:"6/165")
        ors = " OR ".join([f'number:"{_ptcg_escape_phrase(v)}"' for v in num_variants])
        queries.append(f"set.id:{set_id} ({ors})")

    # Strict + name
    if set_id and name and num_variants:
        ors = " OR ".join([f'number:"{_ptcg_escape_phrase(v)}"' for v in num_variants])
        queries.append(f'set.id:{set_id} name:"{_ptcg_escape_phrase(name)}" ({ors})')

    # ---- Next: set.name + number ----
    if sn and num_variants:
        ors = " OR ".join([f'number:"{_ptcg_escape_phrase(v)}"' for v in num_variants])
        queries.append(f'set.name:"{_ptcg_escape_phrase(sn)}" ({ors})')

    if sn and name and num_variants:
        ors = " OR ".join([f'number:"{_ptcg_escape_phrase(v)}"' for v in num_variants])
        queries.append(f'set.name:"{_ptcg_escape_phrase(sn)}" name:"{_ptcg_escape_phrase(name)}" ({ors})')

    # ---- Next: name + number only ----
    if name and num_variants:
        ors = " OR ".join([f'number:"{_ptcg_escape_phrase(v)}"' for v in num_variants])
        queries.append(f'name:"{_ptcg_escape_phrase(name)}" ({ors})')

    # ---- Fallback: name only ----
    if name:
        queries.append(f'name:"{_ptcg_escape_phrase(name)}"')
        queries.append(f"name:{name}")

    # Execute ladder
    for q in queries:
        data = await _pokemontcg_get("/cards", params={"q": q, "pageSize": 10, "orderBy": "number"})
        cards = data.get("data") or []
        if not cards:
            continue

        # Prefer exact number match (using our variants) if possible
        if num_variants:
            num_set = set(v.lower() for v in num_variants if v)
            for c in cards:
                cnum = str(c.get("number", "") or "").strip()
                if cnum and cnum.lower() in num_set:
                    return (str(c.get("id", "")) or "", set_obj or (c.get("set", {}) or {}))

        # Otherwise take first result
        best = cards[0]
        # If we didn't have a set object, capture it from the card
        if not set_obj and isinstance(best.get("set", {}), dict):
            set_obj = best.get("set", {}) or {}
        return (str(best.get("id", "")) or "", set_obj)

    return ("", set_obj)


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
        "model": OPENAI_MODEL,
        "supports": ["cards", "memorabilia", "sealed_products", "market_context_click_only", "canonical_id", "detailed_grading"],
        "market_sources": {
            "pokemontcg_prices": bool(POKEMONTCG_API_KEY),
            "ebay_scrape_enabled": ENABLE_EBAY,
            "ebay_domain": EBAY_DOMAIN if ENABLE_EBAY else None,
        },
        "cors": {"allowed_origins": allowed_origins},
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
    For Pokemon:
      - resolve set via /sets (ptcgoCode)
      - resolve card via /cards (set.id + number variants + name)
      - return authoritative set + id + links
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
  "card_number": "card number if visible (e.g. 4/102 or 004 or 006/165) else empty string",
  "set_code": "2-4 letter/number set abbreviation printed on the card if visible (e.g. MEW, OBF, SVI). EMPTY if not visible",
  "set_name": "set or series name (full name) if visible/known else empty string",
  "confidence": 0.0-1.0,
  "notes": "one short sentence about how you identified it"
}

Rules:
- PRIORITIZE the set_code if you can see one.
- Do not guess a set_code you cannot see.
- KEEP LEADING ZEROS if they appear on the card number (e.g. 006/165).
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
                "set_source": "unknown",
            },
            "pokemontcg": None,
            "error": "AI identification failed",
        })

    data = _parse_json_or_none(result.get("content", "")) or {}

    card_name = _norm_ws(str(data.get("card_name", "Unknown")))
    card_type = _norm_ws(str(data.get("card_type", "Other")))
    year = _norm_ws(str(data.get("year", "")))
    card_number_raw = _norm_ws(str(data.get("card_number", "")))
    card_number = _num_display(card_number_raw)  # KEEP ZEROS
    set_code = _norm_ws(str(data.get("set_code", ""))).upper()
    set_name = _norm_ws(str(data.get("set_name", "")))
    conf = _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0)
    notes = _norm_ws(str(data.get("notes", "")))

    set_info = _canonicalize_set(set_code, set_name)

    canonical_id: Dict[str, Any] = {
        "card_name": card_name,
        "card_type": card_type,
        "year": year,
        "card_number": card_number,
        "set_code": set_info["set_code"],
        "set_name": set_info["set_name"],
        "set_source": set_info["set_source"],
        "confidence": conf,
    }

    # Optional: resolve PokemonTCG card id + authoritative set details
    pokemontcg_payload = None
    if POKEMONTCG_API_KEY and card_type.strip().lower() == "pokemon" and card_name and card_name.lower() != "unknown":
        pid, set_obj = await _pokemontcg_resolve_card_id(
            card_name=card_name,
            set_code=set_info["set_code"],
            card_number=card_number,
            set_name=set_info["set_name"],
        )

        # Fill missing set_name from authoritative set object if we can
        if set_obj and not set_info["set_name"]:
            set_info = {
                "set_code": set_info["set_code"],
                "set_name": _norm_ws(str(set_obj.get("name", ""))),
                "set_source": "pokemontcg_set",
            }
            canonical_id["set_name"] = set_info["set_name"]
            canonical_id["set_source"] = set_info["set_source"]

        if pid:
            card = await _pokemontcg_card_by_id(pid)
            canonical_id["external_ids"] = {"pokemontcg_id": pid}
            # If still missing, fill set name from card.set
            if card and not canonical_id.get("set_name"):
                cset = card.get("set", {}) or {}
                if isinstance(cset, dict) and cset.get("name"):
                    canonical_id["set_name"] = _norm_ws(str(cset.get("name", "")))
                    canonical_id["set_source"] = "pokemontcg_card"

            pokemontcg_payload = {
                "id": pid,
                "name": card.get("name", ""),
                "number": card.get("number", ""),
                "rarity": card.get("rarity", ""),
                "set": card.get("set", {}) or (set_obj or {}),
                "images": card.get("images", {}) or {},
                "links": {
                    "tcgplayer": (card.get("tcgplayer", {}) or {}).get("url", ""),
                    "cardmarket": (card.get("cardmarket", {}) or {}).get("url", ""),
                },
            }

    return JSONResponse(content={
        "card_name": card_name,
        "card_type": card_type,
        "year": year,
        "card_number": card_number,
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
    provided_num = _num_display(_norm_ws(card_number or ""))  # KEEP ZEROS
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
            context += f"- Card Number: {provided_num}\n"
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
- A clear Assessment Summary (2-4 sentences) mentioning the specific issues you see.

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
    "card_number": "best-effort from images (KEEP LEADING ZEROS if shown)",
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
            "error": "AI grading failed",
        })

    data = _parse_json_or_none(result.get("content", "")) or {}

    obs = data.get("observed_id", {}) if isinstance(data.get("observed_id", {}), dict) else {}
    obs_name = _norm_ws(str(obs.get("card_name", "")))
    obs_type = _norm_ws(str(obs.get("card_type", "")))
    obs_year = _norm_ws(str(obs.get("year", "")))
    obs_num = _num_display(_norm_ws(str(obs.get("card_number", ""))))  # KEEP ZEROS
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
# Market Context (Click-only)
# ==============================
_PRICE_RE = re.compile(r"(?P<cur>AU\$|A\$|US\$|\$|EUR|€)?\s*(?P<num>[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)")


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


# ---- OPTIONAL eBay scraping (OFF by default) ----
async def _fetch_html(url: str) -> str:
    async with httpx.AsyncClient(timeout=12.0) as client:
        r = await client.get(url, headers={"User-Agent": UA}, follow_redirects=True)
        if r.status_code != 200:
            return ""
        return r.text or ""


async def _search_ebay_sold_prices(query: str, negate_terms: Optional[List[str]] = None, limit: int = 25) -> Dict[str, Any]:
    if not ENABLE_EBAY:
        return {"prices": [], "currency": "", "sample_size": 0, "url": "", "query": query}

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


def _build_query_ladder(card_name: str, set_name: str, set_code: str, card_number: str, card_type: str) -> List[str]:
    name = _norm_ws(card_name)
    sn = _norm_ws(set_name)
    sc = (set_code or "").strip().upper()
    num = _num_display(card_number)
    t = _norm_ws(card_type)

    variants: List[str] = []
    base_parts = [name]
    if sn and sn.lower() not in name.lower():
        base_parts.append(sn)
    if num:
        base_parts.append(num)
    variants.append(" ".join([p for p in base_parts if p]))

    if sc:
        parts = [name, sc]
        if num:
            parts.append(num)
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
    meta_urls: Dict[str, str] = {}
    used_query = ""
    raw_prices: List[float] = []
    raw_currency = ""
    graded_prices: Dict[str, List[float]] = {"10": [], "9": [], "8": []}

    if not ENABLE_EBAY:
        return {
            "used_query": "",
            "raw_prices": [],
            "raw_currency": "",
            "graded_prices": graded_prices,
            "meta_urls": {},
        }

    target_raw = 3
    target_graded_each = 2

    for q in query_ladder:
        used_query = q
        raw_res = await _search_ebay_sold_prices(q, negate_terms=["graded", "psa", "bgs", "cgc", "slab"], limit=20)
        meta_urls[f"ebay_raw::{q}"] = raw_res.get("url", "")
        if raw_res["prices"]:
            raw_prices = raw_res["prices"]
            raw_currency = raw_res.get("currency", "") or raw_currency

        for grade in ("10", "9", "8"):
            gr_res = await _search_ebay_sold_prices(f"{q} PSA {grade}", negate_terms=[], limit=12)
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

    Primary source for Pokémon:
      - PokémonTCG API (card details + TCGplayer/Cardmarket price objects)
    Optional secondary source:
      - eBay sold/completed scraping (ENABLE_EBAY=1)  [not recommended as "authoritative"]
    """
    clean_name = _norm_ws(card_name)
    clean_set = _norm_ws(card_set or "")
    clean_code = _norm_ws(set_code or "").upper()
    clean_number = _num_display(_norm_ws(card_number or ""))  # KEEP ZEROS
    clean_type = _norm_ws(card_type or "")

    set_info = _canonicalize_set(clean_code, clean_set)
    conf = _clamp(_safe_float(confidence, 0.0), 0.0, 1.0)
    g = _grade_bucket(predicted_grade) or 9

    sources: List[str] = []
    meta_urls: Dict[str, str] = {}

    # ---- PokémonTCG authoritative details + price anchors (Pokémon cards only) ----
    pokemontcg_card: Optional[Dict[str, Any]] = None
    pokemontcg_id: str = ""
    ptcg_prices: Optional[Dict[str, Any]] = None
    anchor_prices: List[float] = []
    anchor_currencies: List[str] = []
    anchor_currency: str = ""

    if POKEMONTCG_API_KEY and ("pokemon" in clean_type.lower()):
        pokemontcg_id, set_obj = await _pokemontcg_resolve_card_id(
            card_name=clean_name,
            set_code=set_info["set_code"],
            card_number=clean_number,
            set_name=set_info["set_name"],
        )
        if set_obj and not set_info["set_name"]:
            set_info = {"set_code": set_info["set_code"], "set_name": _norm_ws(str(set_obj.get("name", ""))), "set_source": "pokemontcg_set"}

        if pokemontcg_id:
            pokemontcg_card = await _pokemontcg_card_by_id(pokemontcg_id)
            ptcg_prices = _pokemontcg_extract_prices(pokemontcg_card or {})
            anchor_prices = (ptcg_prices.get("anchor_values") or [])
            anchor_currencies = (ptcg_prices.get("currencies") or [])
            anchor_currency = anchor_currencies[0] if anchor_currencies else "USD"
            sources.append("PokemonTCG API (details + TCGplayer/Cardmarket)")

    # ---- Optional eBay observed sales ----
    query_ladder = _build_query_ladder(clean_name, set_info["set_name"], set_info["set_code"], clean_number, clean_type)
    gathered = await _gather_market_samples(query_ladder)
    used_query = gathered["used_query"]
    raw_prices = gathered["raw_prices"]
    raw_currency = gathered["raw_currency"]
    graded_prices = gathered["graded_prices"]
    meta_urls.update(gathered["meta_urls"])

    if ENABLE_EBAY and (raw_prices or any(graded_prices.get(k) for k in ("10", "9", "8"))):
        sources.append("eBay (sold/completed)")

    # Decide availability: if ANY samples exist (Pokemon anchors OR eBay)
    has_any = bool(anchor_prices) or bool(raw_prices) or any(len(v) > 0 for v in graded_prices.values())
    if not has_any:
        return JSONResponse(content={
            "available": False,
            "mode": "click_only",
            "card": {"name": clean_name, "set": set_info["set_name"], "set_code": set_info["set_code"], "number": clean_number, "type": clean_type},
            "pokemontcg": {"id": pokemontcg_id} if pokemontcg_id else None,
            "query_ladder": query_ladder,
            "used_query": used_query,
            "message": "No market samples found from the available sources. Confirm set code + number, or try again.",
            "sources": sources,
            "meta": {"urls": meta_urls},
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "disclaimer": "Informational market context only. Not financial advice.",
        })

    raw_stats = _stats(raw_prices)
    anchor_stats = _stats(anchor_prices)
    graded_stats: Dict[str, Any] = {k: _stats(v) for k, v in graded_prices.items() if v}

    # Baseline preference:
    # 1) PokemonTCG anchor (more stable) if present
    # 2) eBay raw if present
    raw_baseline = 0.0
    if anchor_stats["sample_size"] >= 1:
        raw_baseline = anchor_stats["median"] or anchor_stats["avg"]
    elif raw_stats["sample_size"] >= 2:
        raw_baseline = raw_stats["median"] or raw_stats["avg"]

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
            "set": pokemontcg_out_set if (pokemontcg_out_set := (pokemontcg_card.get("set", {}) or {})) else {},
            "images": pokemontcg_card.get("images", {}) or {},
            "prices": ptcg_prices,
            "links": {
                "tcgplayer": (pokemontcg_card.get("tcgplayer", {}) or {}).get("url", ""),
                "cardmarket": (pokemontcg_card.get("cardmarket", {}) or {}).get("url", ""),
            },
        }

    return JSONResponse(content={
        "available": True,
        "mode": "click_only",
        "confidence": confidence_label,
        "card": {"name": clean_name, "set": set_info["set_name"], "set_code": set_info["set_code"], "number": clean_number, "type": clean_type},
        "query_ladder": query_ladder,
        "used_query": used_query,
        "observed": {
            "currency": currency,
            "pokemon_prices_anchor": anchor_stats if anchor_stats["sample_size"] else None,
            "raw_ebay": raw_stats if raw_stats["sample_size"] else None,
            "graded_psa_ebay": graded_stats if graded_stats else None,
            "trend_ebay": trend if ENABLE_EBAY else "disabled",
            "liquidity": liquidity,
        },
        "grade_impact": {
            "predicted_grade": str(g),
            "confidence_input": conf,
            "grade_atomic_distribution": dist,
            "expected_graded_value": expected_value,
            "raw_baseline_value": round(raw_baseline, 2) if raw_baseline else None,
            "grading_cost": round(float(grading_cost or 0.0), 2),
            "estimated_value_difference": value_difference,
        },
        "pokemontcg": pokemontcg_out,
        "sources": sources,
        "meta": {"urls": meta_urls},
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "disclaimer": "Informational market context only. Figures are based on third‑party price anchors and/or observed sales and may vary. Not financial advice.",
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

@app.post("/api/identify-memorabilia")
async def identify_memorabilia(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    image3: Optional[UploadFile] = File(None),
    image4: Optional[UploadFile] = File(None),
):
    """
    Identify memorabilia / sealed products from up to 4 images.
    Returns fields expected by your WP front-end:
      item_type, description, signatures, seal_condition, confidence,
      authenticity_notes, notable_features, identify_token
    """
    imgs: List[bytes] = []
    for f in [image1, image2, image3, image4]:
        if f is None:
            continue
        b = await f.read()
        if b and len(b) >= 200:
            imgs.append(b)

    if len(imgs) < 2:
        raise HTTPException(status_code=400, detail="Please provide at least 2 clear images")

    prompt = """You are identifying a collectible item (memorabilia OR sealed product) from photos.

Return ONLY valid JSON with these exact fields:

{
  "item_type": "Memorabilia/Sealed Product/Other",
  "description": "Concise but specific description of the item (what it is, brand/team/player/set, era if visible)",
  "signatures": "List what is signed and by who if visible, otherwise 'None visible'",
  "seal_condition": "If sealed: describe seal/wrap condition (tight, loose, tears, re-sealed concerns). If not sealed: 'Not applicable'",
  "authenticity_notes": "Any authenticity observations or concerns based on what is visible (stickers, holograms, COA, inconsistencies, etc.)",
  "notable_features": "Short list of notable features (serial #, hologram, COA, tags, inscriptions, limited edition markings)",
  "confidence": 0.0-1.0
}

Rules:
- Do NOT invent signatures or serial numbers you cannot see.
- If unsure, be explicit in authenticity_notes and lower confidence.
- Keep it factual, based only on the images.
Respond ONLY with JSON, no extra text.
"""

    content = [{"type": "text", "text": prompt}]
    # Use existing helper _b64() from your file
    for i, b in enumerate(imgs, start=1):
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_b64(b)}", "detail": "high"}
        })
        if i < len(imgs):
            content.append({"type": "text", "text": f"Image {i} above | Image {i+1} below"})

    msg = [{"role": "user", "content": content}]

    result = await _openai_chat(msg, max_tokens=900, temperature=0.1)
    if result.get("error"):
        return JSONResponse(content={
            "item_type": "Unknown Item",
            "description": "",
            "signatures": "None visible",
            "seal_condition": "Not applicable",
            "authenticity_notes": "AI identification failed",
            "notable_features": "",
            "confidence": 0.0,
            "identify_token": f"idt_{secrets.token_urlsafe(12)}",
            "error": "AI identification failed"
        })

    data = _parse_json_or_none(result.get("content", "")) or {}

    return JSONResponse(content={
        "item_type": _norm_ws(str(data.get("item_type", "Unknown Item"))),
        "description": _norm_ws(str(data.get("description", ""))),
        "signatures": _norm_ws(str(data.get("signatures", "None visible"))) or "None visible",
        "seal_condition": _norm_ws(str(data.get("seal_condition", "Not applicable"))) or "Not applicable",
        "authenticity_notes": _norm_ws(str(data.get("authenticity_notes", ""))),
        "notable_features": _norm_ws(str(data.get("notable_features", ""))),
        "confidence": _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0),
        "identify_token": f"idt_{secrets.token_urlsafe(12)}",
    })

@app.post("/api/assess-memorabilia")
async def assess_memorabilia(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    image3: Optional[UploadFile] = File(None),
    image4: Optional[UploadFile] = File(None),

    # Optional context from the IDENTIFY step (send via FormData)
    item_type: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    signatures: Optional[str] = Form(None),
    seal_condition: Optional[str] = Form(None),
):
    """
    Assess memorabilia / sealed products from up to 4 images.
    Designed as the "step 2" after identify-memorabilia.
    """

    imgs: List[bytes] = []
    for f in [image1, image2, image3, image4]:
        if f is None:
            continue
        b = await f.read()
        if b and len(b) >= 200:
            imgs.append(b)

    if len(imgs) < 2:
        raise HTTPException(status_code=400, detail="Please provide at least 2 clear images")

    # Clean provided context
    ctx_item_type = _norm_ws(item_type or "")
    ctx_desc = _norm_ws(description or "")
    ctx_sigs = _norm_ws(signatures or "")
    ctx_seal = _norm_ws(seal_condition or "")

    context_block = ""
    if ctx_item_type or ctx_desc or ctx_sigs or ctx_seal:
        context_block = f"""
KNOWN/PROVIDED DETAILS FROM STEP 1 (use as hints; do not invent):
- item_type: {ctx_item_type}
- description: {ctx_desc}
- signatures: {ctx_sigs}
- seal_condition: {ctx_seal}
"""

    prompt = f"""You are a professional collectibles authenticator and condition assessor with current market value experience.

You are assessing a collectible item (memorabilia OR sealed product) from photos.
Return ONLY valid JSON with this EXACT structure:

{{
  "assessment": {{
    "overall_condition": "Mint/Near Mint/Excellent/Good/Fair/Poor",
    "seal_assessment": {{
      "status": "Sealed/Not sealed/Unclear",
      "notes": "If sealed: describe wrap/tape integrity, holes, tears, loose wrap, reseal signs. If not sealed: 'Not applicable'."
    }},
    "signature_assessment": {{
      "status": "No signature/Signature present/Unclear",
      "notes": "If signed: what appears signed and any observations (ink consistency, placement). Do not guess identity."
    }},
    "authenticity_observations": [
      "Bullet points: hologram/COA, sticker types, serial tags, stitching quality, print consistency, warning signs, etc."
    ],
    "defects": [
      "Clear defect sentences with location (e.g., 'Front top edge: crease', 'Wrap: tear near right seam')"
    ],
    "notable_features": [
      "Observable features: serial numbers (ONLY if visible), holograms, COA, inscriptions, limited edition markings"
    ],
    "risk_flags": [
      "Short flags: possible reseal, inconsistent hologram, suspicious sticker, missing COA, heavy wear, etc."
    ],
    "confidence": 0.0-1.0,
    "summary": "2-4 sentences describing condition + biggest risks/limiters based on what is visible."
  }},
  "recommended_action": {{
    "decision": "approve_for_submission/needs_manual_review/decline",
    "reasons": [
      "Short reasons tied to what you can see"
    ]
  }}
}}

Rules:
- Do NOT invent serial numbers, COA details, or signature identities you cannot clearly see.
- If uncertain, lower confidence and choose needs_manual_review.
- Focus on what is visible and verifiable from the images.

{context_block}

Respond ONLY with JSON, no extra text.
"""

    content = [{"type": "text", "text": prompt}]
    for i, b in enumerate(imgs, start=1):
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_b64(b)}", "detail": "high"}
        })
        if i < len(imgs):
            content.append({"type": "text", "text": f"Image {i} above | Image {i+1} below"})

    msg = [{"role": "user", "content": content}]

    result = await _openai_chat(msg, max_tokens=1200, temperature=0.1)
    if result.get("error"):
        return JSONResponse(content={
            "assessment": {
                "overall_condition": "Unclear",
                "seal_assessment": {"status": "Unclear", "notes": ""},
                "signature_assessment": {"status": "Unclear", "notes": ""},
                "authenticity_observations": [],
                "defects": [],
                "notable_features": [],
                "risk_flags": ["ai_failed"],
                "confidence": 0.0,
                "summary": "Assessment failed."
            },
            "recommended_action": {
                "decision": "needs_manual_review",
                "reasons": ["AI assessment failed"]
            },
            "assess_token": f"asm_{secrets.token_urlsafe(12)}",
            "error": "AI assessment failed"
        })

    data = _parse_json_or_none(result.get("content", "")) or {}
    assessment = data.get("assessment", {}) if isinstance(data.get("assessment", {}), dict) else {}
    rec = data.get("recommended_action", {}) if isinstance(data.get("recommended_action", {}), dict) else {}

    # Normalize a few fields
    conf = _clamp(_safe_float(assessment.get("confidence", 0.0)), 0.0, 1.0)
    decision = _norm_ws(str(rec.get("decision", "needs_manual_review"))) or "needs_manual_review"
    if decision not in ("approve_for_submission", "needs_manual_review", "decline"):
        decision = "needs_manual_review"

    return JSONResponse(content={
        "assessment": {
            "overall_condition": _norm_ws(str(assessment.get("overall_condition", "Unclear"))),
            "seal_assessment": assessment.get("seal_assessment", {"status": "Unclear", "notes": ""}),
            "signature_assessment": assessment.get("signature_assessment", {"status": "Unclear", "notes": ""}),
            "authenticity_observations": assessment.get("authenticity_observations", []) if isinstance(assessment.get("authenticity_observations", []), list) else [],
            "defects": assessment.get("defects", []) if isinstance(assessment.get("defects", []), list) else [],
            "notable_features": assessment.get("notable_features", []) if isinstance(assessment.get("notable_features", []), list) else [],
            "risk_flags": assessment.get("risk_flags", []) if isinstance(assessment.get("risk_flags", []), list) else [],
            "confidence": conf,
            "summary": _norm_ws(str(assessment.get("summary", ""))),
        },
        "recommended_action": {
            "decision": decision,
            "reasons": rec.get("reasons", []) if isinstance(rec.get("reasons", []), list) else [],
        },
        "assess_token": f"asm_{secrets.token_urlsafe(12)}",
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
