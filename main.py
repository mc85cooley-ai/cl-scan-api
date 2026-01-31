"""
The Collectors League Australia — Scan API
Futureproof v6.2.3 (2026-01-31)

Goals:
- Trading card identification + grading pre-assessment (front/back)
- Memorabilia / sealed product identification + assessment (multi-image)
- Click-only market context (informational anchors only)
  - Pokemon cards: PokemonTCG API (TCGplayer + Cardmarket) is PRIMARY
  - eBay scraping is OPTIONAL and OFF by default (USE_EBAY=1)
Notes:
- This service provides informational market context only. Not financial advice.
"""

from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List, Tuple
import base64
import os
import json
import secrets
import re
import traceback
from datetime import datetime
from statistics import mean, median
from functools import wraps

import httpx
from bs4 import BeautifulSoup

# ==============================
# App & Config
# ==============================
app = FastAPI(title="Collectors League Scan API")

# CORS:
# Browsers reject allow_credentials=True with allow_origins=["*"].
# This API does NOT need cookies, so keep credentials OFF.
ALLOWED_ORIGINS = [
    "https://collectors-league.com",
    "https://www.collectors-league.com",
    # Local dev
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

APP_VERSION = os.getenv("CL_SCAN_VERSION", "2026-01-31-v6.2.3")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
POKEMONTCG_API_KEY = os.getenv("POKEMONTCG_API_KEY", "").strip()
POKEMONTCG_BASE = "https://api.pokemontcg.io/v2"

# Optional scraping (OFF by default)
USE_EBAY = os.getenv("USE_EBAY", "0").strip().lower() in ("1", "true", "yes")
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
# Safety wrapper
# ==============================
def safe_endpoint(func):
    """Catch unhandled exceptions and return JSON, without crashing the worker."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Endpoint {func.__name__} crashed: {e}")
            traceback.print_exc()
            return JSONResponse(
                content={"error": True, "endpoint": func.__name__, "message": str(e)},
                status_code=500
            )
    return wrapper

# ==============================
# Set Code Mapping (expand over time)
# ==============================
SET_CODE_MAP: Dict[str, str] = {
    # Examples (keep small + correct)
    "OBF": "Obsidian Flames",
    "SVI": "Scarlet & Violet",
    "MEW": "Scarlet & Violet—151",
    "BS": "Base Set",
}

# ==============================
# Helpers (generic)
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
    """Preserve leading zeros for UI display."""
    s = (s or "").strip().replace("#", "").strip()
    if not s:
        return ""
    return re.sub(r"\s+", "", s)

def _card_number_for_query(s: str) -> str:
    """Normalize for PokemonTCG queries by stripping leading zeros of the first segment."""
    s = _clean_card_number_display(s)
    if not s:
        return ""
    if "/" in s:
        a, b = s.split("/", 1)
        a2 = a.lstrip("0") or "0"
        return f"{a2}/{b}"
    return s.lstrip("0") or "0"

def _same_number(a: str, b: str) -> bool:
    return _card_number_for_query(a) == _card_number_for_query(b)

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
        return {"set_code": sc or "", "set_name": sn, "set_source": "ai_name_only"}
    if sc:
        return {"set_code": sc, "set_name": "", "set_source": "code_only"}
    return {"set_code": "", "set_name": "", "set_source": "unknown"}

# ==============================
# OpenAI helper
# ==============================
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
                return {"error": True, "status": r.status_code, "message": r.text[:800]}
            data = r.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return {"error": False, "content": content}
    except Exception as e:
        return {"error": True, "status": 0, "message": str(e)}

# ==============================
# PokemonTCG helpers (metadata + price anchors)
# ==============================
async def _pokemontcg_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not POKEMONTCG_API_KEY:
        return {}
    url = path if path.startswith("http") else f"{POKEMONTCG_BASE}{path}"
    headers = {"X-Api-Key": POKEMONTCG_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.get(url, headers=headers, params=params)
            if r.status_code != 200:
                return {}
            return r.json() if r.content else {}
    except Exception:
        return {}

async def _pokemontcg_resolve_set_by_ptcgo(ptcgo: str) -> Dict[str, Any]:
    ptcgo = (ptcgo or "").strip().upper()
    if not ptcgo or not POKEMONTCG_API_KEY:
        return {}
    data = await _pokemontcg_get("/sets", params={"q": f"ptcgoCode:{ptcgo}", "pageSize": 5})
    sets = data.get("data") or []
    return sets[0] if sets else {}

async def _pokemontcg_resolve_card_id(
    card_name: str,
    set_code: str,
    card_number_display: str,
    set_name: str = "",
    set_id: str = "",
) -> str:
    name = _norm_ws(card_name)
    sc = (set_code or "").strip().upper()
    num_q = _card_number_for_query(card_number_display or "")
    sn = _norm_ws(set_name or "")
    sid = (set_id or "").strip()

    queries: List[str] = []
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
        if card_number_display:
            want = num_q.split("/")[0] if num_q else ""
            for c in cards:
                if want and _same_number(str(c.get("number", "")), want):
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
# Optional eBay sold scraping (OFF by default)
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
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.get(url, headers={"User-Agent": UA}, follow_redirects=True)
            if r.status_code != 200:
                return ""
            return r.text or ""
    except Exception:
        return ""

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
# Card: Identify (AI + PokemonTCG enrichment)
# ==============================
@app.post("/api/identify")
@safe_endpoint
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
                "card_name": "Unknown", "card_type": "Other", "year": "", "card_number": "",
                "set_code": "", "set_name": "", "set_id": "", "set_source": "unknown",
                "confidence": 0.0, "pokemontcg_verified": False
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

    # PokemonTCG enrichment (Pokemon only)
    enriched = False
    final_set_id = ""
    ptcg_card: Optional[Dict[str, Any]] = None

    if POKEMONTCG_API_KEY and card_type.strip().lower() == "pokemon" and card_name and card_name.lower() != "unknown":
        # If set_code present but set_name missing, resolve set name first
        ptcg_set = {}
        if set_code and not set_name:
            ptcg_set = await _pokemontcg_resolve_set_by_ptcgo(set_code)
            if ptcg_set.get("name"):
                set_name = _norm_ws(str(ptcg_set.get("name", "")))
                final_set_id = str(ptcg_set.get("id", "")) if ptcg_set.get("id") else ""

        pid = await _pokemontcg_resolve_card_id(
            card_name=card_name,
            set_code=set_code,
            card_number_display=card_number_display,
            set_name=set_name,
            set_id=final_set_id,
        )

        if pid:
            ptcg_card = await _pokemontcg_card_by_id(pid)
            if ptcg_card:
                enriched = True
                # Overwrite with authoritative metadata where available
                card_name = _norm_ws(str(ptcg_card.get("name", card_name)))
                card_number_display = _clean_card_number_display(str(ptcg_card.get("number", card_number_display)))
                set_obj = ptcg_card.get("set", {}) or {}
                set_name = _norm_ws(str(set_obj.get("name", set_name)))
                set_code = _norm_ws(str(set_obj.get("ptcgoCode", set_code))).upper()
                final_set_id = str(set_obj.get("id", final_set_id)) if set_obj.get("id") else final_set_id
                release_date = str(set_obj.get("releaseDate", ""))
                if not year and release_date[:4].isdigit():
                    year = release_date[:4]
                conf = min(1.0, conf + 0.2)
                notes = f"Verified via PokemonTCG API (id: {pid})"

    set_info = _canonicalize_set(set_code, set_name)

    canonical_id = {
        "card_name": card_name,
        "card_type": card_type,
        "year": year,
        "card_number": card_number_display,
        "card_number_norm": _card_number_for_query(card_number_display),
        "set_code": set_info["set_code"],
        "set_name": set_info["set_name"],
        "set_id": final_set_id,
        "set_source": ("pokemontcg_api" if enriched else set_info["set_source"]),
        "confidence": conf,
        "pokemontcg_verified": enriched,
    }

    pokemontcg_payload = None
    if isinstance(ptcg_card, dict) and ptcg_card:
        pokemontcg_payload = {
            "id": ptcg_card.get("id", ""),
            "name": ptcg_card.get("name", ""),
            "number": ptcg_card.get("number", ""),
            "rarity": ptcg_card.get("rarity", ""),
            "set": ptcg_card.get("set", {}) or {},
            "images": ptcg_card.get("images", {}) or {},
            "prices": _pokemontcg_extract_prices(ptcg_card),
            "links": {
                "tcgplayer": (ptcg_card.get("tcgplayer", {}) or {}).get("url", ""),
                "cardmarket": (ptcg_card.get("cardmarket", {}) or {}).get("url", ""),
            },
        }

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
        "pokemontcg_enriched": enriched,
    })

# ==============================
# Card: Verify (detailed grading)
# ==============================
@app.post("/api/verify")
@safe_endpoint
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

    # backfill set name by ptcgo code if missing
    ptcg_set = {}
    if POKEMONTCG_API_KEY and provided_code and not provided_set:
        ptcg_set = await _pokemontcg_resolve_set_by_ptcgo(provided_code)
        if ptcg_set.get("name"):
            provided_set = _norm_ws(str(ptcg_set.get("name", "")))

    set_info = _canonicalize_set(provided_code, provided_set)

    context = ""
    if provided_name or set_info["set_name"] or provided_num_display or provided_year or provided_type or set_info["set_code"]:
        context = "\n\nKNOWN/PROVIDED CARD DETAILS (use as hints, do not force if images contradict):\n"
        if provided_name: context += f"- Card Name: {provided_name}\n"
        if set_info["set_name"]: context += f"- Set Name: {set_info['set_name']}\n"
        if set_info["set_code"]: context += f"- Set Code: {set_info['set_code']}\n"
        if provided_num_display: context += f"- Card Number: {provided_num_display}\n"
        if provided_year: context += f"- Year: {provided_year}\n"
        if provided_type: context += f"- Type: {provided_type}\n"

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
# Memorabilia & Sealed Products: Identify
# ==============================
@app.post("/api/identify-memorabilia")
@safe_endpoint
async def identify_memorabilia(
    image1: UploadFile = File(...),
    image2: Optional[UploadFile] = File(None),
    image3: Optional[UploadFile] = File(None),
    image4: Optional[UploadFile] = File(None),
):
    b1 = await image1.read()
    if not b1 or len(b1) < 500:
        raise HTTPException(status_code=400, detail="Image 1 is too small or empty")

    images = [b1]
    for f in (image2, image3, image4):
        if f:
            bb = await f.read()
            if bb and len(bb) >= 500:
                images.append(bb)

    prompt = """You are identifying a collectible item (memorabilia or sealed product) from photos.

Return ONLY valid JSON with these exact fields:

{
  "item_type": "sealed booster box/sealed pack/elite trainer box/autographed memorabilia/game-used memorabilia/graded item/display case/other",
  "description": "detailed description of the item including brand, set/series, year if visible",
  "signatures": "names of any visible signatures or 'None visible'",
  "seal_condition": "Factory Sealed/Opened/Resealed/Damaged/Not applicable",
  "authenticity_notes": "authenticity indicators visible (holograms, stickers, certificates) and red flags",
  "notable_features": "unique features worth noting",
  "confidence": 0.0-1.0,
  "category": "memorabilia/sealed/other"
}

Rules:
- Be specific about condition and authenticity markers.
- Note any visible damage to seals/packaging.
- If you cannot identify with confidence, set confidence to 0.0 and item_type to "other".
Respond ONLY with JSON, no extra text.
"""

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for i, img in enumerate(images):
        if i > 0:
            content.append({"type": "text", "text": f"--- IMAGE {i+1} ---"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(img)}", "detail": "high"}})

    msg = [{"role": "user", "content": content}]
    result = await _openai_chat(msg, max_tokens=900, temperature=0.1)
    if result.get("error"):
        return JSONResponse(content={
            "item_type": "other",
            "description": "Identification failed",
            "signatures": "Unknown",
            "seal_condition": "Unknown",
            "authenticity_notes": "",
            "notable_features": "",
            "confidence": 0.0,
            "category": "other",
            "identify_token": f"idt_{secrets.token_urlsafe(12)}",
            "error": "AI identification failed",
        })

    data = _parse_json_or_none(result.get("content", "")) or {}
    return JSONResponse(content={
        "item_type": _norm_ws(str(data.get("item_type", "other"))),
        "description": _norm_ws(str(data.get("description", ""))),
        "signatures": _norm_ws(str(data.get("signatures", "None visible"))),
        "seal_condition": _norm_ws(str(data.get("seal_condition", "Not applicable"))),
        "authenticity_notes": _norm_ws(str(data.get("authenticity_notes", ""))),
        "notable_features": _norm_ws(str(data.get("notable_features", ""))),
        "confidence": _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0),
        "category": _norm_ws(str(data.get("category", "other"))),
        "identify_token": f"idt_{secrets.token_urlsafe(12)}",
    })

# ==============================
# Memorabilia & Sealed Products: Assess
# ==============================
@app.post("/api/assess-memorabilia")
@safe_endpoint
async def assess_memorabilia(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    image3: Optional[UploadFile] = File(None),
    image4: Optional[UploadFile] = File(None),
    item_type: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    b1 = await image1.read()
    b2 = await image2.read()
    if not b1 or not b2:
        raise HTTPException(status_code=400, detail="Both images required")

    images = [b1, b2]
    for f in (image3, image4):
        if f:
            bb = await f.read()
            if bb and len(bb) >= 500:
                images.append(bb)

    ctx = ""
    if item_type or description:
        ctx = "\n\nKNOWN ITEM DETAILS:\n"
        if item_type: ctx += f"- Item Type: {item_type}\n"
        if description: ctx += f"- Description: {description}\n"

    prompt = f"""You are a professional memorabilia/sealed product authenticator.

Analyze ALL images. Return ONLY valid JSON.

You MUST provide:
- Overall condition grade (Mint/Near Mint/Excellent/Good/Fair/Poor)
- Seal integrity assessment (if applicable)
- Packaging condition notes
- Signature assessment (if applicable)
- Value-affecting factors (not prices)
- Detailed assessment summary (2-4 sentences)

{ctx}

Return JSON with this EXACT structure:

{{
  "condition_grade": "Mint/Near Mint/Excellent/Good/Fair/Poor",
  "confidence": 0.0-1.0,
  "seal_integrity": {{
    "status": "Factory Sealed/Opened/Resealed/Compromised/Not Applicable",
    "notes": "detailed notes about seal condition"
  }},
  "packaging_condition": {{
    "grade": "Mint/Near Mint/Excellent/Good/Fair/Poor",
    "notes": "detailed notes about packaging"
  }},
  "signature_assessment": {{
    "present": true/false,
    "quality": "Clear/Faded/Smudged/Not Applicable",
    "notes": "notes about signature condition if present"
  }},
  "value_factors": [
    "List specific factors that affect value (positive or negative)"
  ],
  "defects": [
    "Each defect as a clear sentence (e.g., 'Packaging: corner crease top-right')"
  ],
  "overall_assessment": "2-4 sentence narrative about condition and key factors",
  "flags": [
    "Important issues (damage, authenticity concerns, condition notes)"
  ]
}}

Important:
- If something doesn't apply, use "Not Applicable" or empty arrays.
- Do not include market values in this endpoint.
Respond ONLY with JSON.
"""

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for i, img in enumerate(images):
        if i > 0:
            content.append({"type": "text", "text": f"--- IMAGE {i+1} ---"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(img)}", "detail": "high"}})

    msg = [{"role": "user", "content": content}]
    result = await _openai_chat(msg, max_tokens=2200, temperature=0.1)
    if result.get("error"):
        return JSONResponse(content={
            "condition_grade": "Unable to assess",
            "confidence": 0.0,
            "seal_integrity": {"status": "Unknown", "notes": ""},
            "packaging_condition": {"grade": "Unknown", "notes": ""},
            "signature_assessment": {"present": False, "quality": "Not Applicable", "notes": ""},
            "value_factors": [],
            "defects": [],
            "overall_assessment": "Assessment failed.",
            "flags": [],
            "error": "AI assessment failed"
        })

    data = _parse_json_or_none(result.get("content", "")) or {}
    return JSONResponse(content={
        "condition_grade": _norm_ws(str(data.get("condition_grade", "N/A"))),
        "confidence": _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0),
        "seal_integrity": data.get("seal_integrity", {"status": "Unknown", "notes": ""}),
        "packaging_condition": data.get("packaging_condition", {"grade": "Unknown", "notes": ""}),
        "signature_assessment": data.get("signature_assessment", {"present": False, "quality": "Not Applicable", "notes": ""}),
        "value_factors": data.get("value_factors", []) if isinstance(data.get("value_factors"), list) else [],
        "defects": data.get("defects", []) if isinstance(data.get("defects"), list) else [],
        "overall_assessment": _norm_ws(str(data.get("overall_assessment", ""))),
        "flags": data.get("flags", []) if isinstance(data.get("flags"), list) else [],
        "verify_token": f"vfy_{secrets.token_urlsafe(12)}",
    })

# ==============================
# Market Context (click-only)
# ==============================
@app.post("/api/market-context")
@safe_endpoint
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
    import asyncio
    try:
        return await asyncio.wait_for(
            _market_context_impl(
                card_name=card_name,
                predicted_grade=predicted_grade,
                confidence=confidence,
                card_number=card_number,
                card_set=card_set,
                set_code=set_code,
                card_type=card_type,
                grading_cost=grading_cost,
            ),
            timeout=35.0
        )
    except asyncio.TimeoutError:
        return JSONResponse(content={
            "available": False,
            "mode": "click_only",
            "error": "Timeout after 35 seconds",
            "message": "Market context request took too long. Try again.",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "disclaimer": "Informational market context only. Not financial advice."
        })

async def _market_context_impl(
    card_name: str,
    predicted_grade: str,
    confidence: float,
    card_number: Optional[str],
    card_set: Optional[str],
    set_code: Optional[str],
    card_type: Optional[str],
    grading_cost: float,
):
    clean_name = _norm_ws(card_name or "")
    clean_type = _norm_ws(card_type or "")
    clean_set = _norm_ws(card_set or "")
    clean_code = _norm_ws(set_code or "").upper()
    clean_number_display = _clean_card_number_display(card_number or "")

    # If set code exists but set name missing, resolve from PokemonTCG sets endpoint
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

    pokemontcg_out = None
    anchor_prices: List[float] = []
    anchor_stats = {"avg": 0, "median": 0, "low": 0, "high": 0, "sample_size": 0}
    currency_anchor = "unknown"

    # PokemonTCG is PRIMARY for Pokemon cards
    if POKEMONTCG_API_KEY and ("pokemon" in clean_type.lower() or clean_type == ""):
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
            currency_anchor = (ptcg_prices.get("currencies") or ["unknown"])[0]
            sources.append("PokemonTCG API (TCGplayer/Cardmarket)")
            pokemontcg_out = {
                "id": pid,
                "name": (card or {}).get("name", ""),
                "number": (card or {}).get("number", ""),
                "rarity": (card or {}).get("rarity", ""),
                "set": (card or {}).get("set", {}) or {},
                "images": (card or {}).get("images", {}) or {},
                "prices": ptcg_prices,
            }

    raw_prices: List[float] = []
    raw_currency = ""

    # Optional eBay (OFF by default)
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
            "message": "No market anchor data available from PokemonTCG (and eBay is disabled or returned no samples).",
            "sources": sources,
            "pokemontcg": pokemontcg_out,
            "meta": {"urls": meta_urls, "use_ebay": USE_EBAY},
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "disclaimer": "Informational market context only. Not financial advice."
        })

    raw_stats = _stats(raw_prices)
    liquidity = _liquidity_label(max(raw_stats["sample_size"], anchor_stats["sample_size"]))
    currency_out = raw_currency or currency_anchor

    baseline = 0.0
    if anchor_stats["sample_size"] >= 1:
        baseline = anchor_stats["median"] or anchor_stats["avg"]
    elif raw_stats["sample_size"] >= 1:
        baseline = raw_stats["median"] or raw_stats["avg"]

    baseline_minus_grading = None
    if baseline and grading_cost:
        try:
            baseline_minus_grading = round(float(baseline) - float(grading_cost or 0.0), 2)
        except Exception:
            baseline_minus_grading = None

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
            "currency": currency_out,
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
            "baseline_minus_grading_cost": baseline_minus_grading,
        },
        "pokemontcg": pokemontcg_out,
        "sources": sources,
        "meta": {"urls": meta_urls, "use_ebay": USE_EBAY},
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "disclaimer": "Informational market context only. Figures are third-party anchors and/or observed sales and may vary. Not financial advice."
    })

# Legacy alias (keep older frontend working)
@app.post("/api/market-intelligence")
@safe_endpoint
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
    # Attach deprecation hint when JSONResponse
    if isinstance(res, JSONResponse):
        try:
            payload = json.loads(res.body.decode("utf-8"))
            payload["deprecated"] = True
            payload["deprecated_message"] = "Use /api/market-context instead."
            return JSONResponse(content=payload, status_code=res.status_code)
        except Exception:
            return res
    return res

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
