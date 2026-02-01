"""
The Collectors League Australia - Scan API
Futureproof v6.4.1 (2026-01-31)

What changed vs v6.3.x
- ✅ Adds PriceCharting API as primary market source for cards + sealed/memorabilia (current prices).
- ✅ Keeps PokemonTCG.io for identification + metadata enrichment (NOT price history).
- ✅ Adds weekly snapshot option (store PriceCharting CSV/API snapshots on a schedule) to build your own price history.
- ✅ eBay API scaffolding included but DISABLED by default (waiting for your dev account approval).
- ✅ Market endpoints return "click-only" informational context + no ROI language.

Env vars
- OPENAI_API_KEY (required for vision grading/ID)
- POKEMONTCG_API_KEY (optional; used for Pokemon metadata enrichment)
- PRICECHARTING_TOKEN (recommended; enables pricing for cards + sealed/memorabilia)
- USE_EBAY_API=0/1 (default 0) + EBAY_APP_ID/EBAY_CERT_ID/EBAY_DEV_ID/EBAY_OAUTH_TOKEN (optional; scaffold only)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from statistics import mean, median
from functools import wraps
import base64
import os
import json
import sqlite3
import csv
import secrets
import hashlib
import re
import traceback
from pathlib import Path

import httpx
import time
import math

# Optional image processing for 2-pass defect enhancement
try:
    from PIL import Image, ImageEnhance, ImageOps, ImageFilter
except Exception:
    Image = None
    ImageEnhance = None
    ImageOps = None
    ImageFilter = None

# Simple in-memory caches (per-process)
_FX_CACHE = {"ts": 0, "usd_aud": None}
_EBAY_CACHE = {}  # key -> {ts, data}
FX_CACHE_SECONDS = int(os.getenv("FX_CACHE_SECONDS", "3600"))

# ==============================
# App & CORS
# ==============================
app = FastAPI(title="Collectors League Scan API")

ALLOWED_ORIGINS = [
    "https://collectors-league.com",
    "https://www.collectors-league.com",
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

def safe_endpoint(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ {func.__name__} crashed: {e}")
            traceback.print_exc()
            return JSONResponse(content={"error": True, "endpoint": func.__name__, "message": str(e)}, status_code=500)
    return wrapper

# ==============================
# Config
# ==============================
APP_VERSION = os.getenv("CL_SCAN_VERSION", "2026-01-31-v6.5.0")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
POKEMONTCG_API_KEY = os.getenv("POKEMONTCG_API_KEY", "").strip()
PRICECHARTING_TOKEN = os.getenv("PRICECHARTING_TOKEN", "").strip()
ADMIN_TOKEN = os.getenv("CL_ADMIN_TOKEN", "").strip()  # optional


# ==============================
# FX + eBay helpers
# ==============================
UA = "CollectorsLeagueScan/6.6 (+https://collectors-league.com)"

async def _fx_usd_to_aud() -> float:
    """Return live-ish USD->AUD rate with a short cache. Falls back to 1.50 if unavailable."""
    try:
        now = int(time.time())
        if _FX_CACHE.get("usd_aud") and (now - int(_FX_CACHE.get("ts") or 0) < FX_CACHE_SECONDS):
            return float(_FX_CACHE["usd_aud"])
        # exchangerate.host is free and doesn't require an API key
        url = "https://api.exchangerate.host/latest?base=USD&symbols=AUD"
        async with httpx.AsyncClient(timeout=15.0, headers={"User-Agent": UA}) as client:
            r = await client.get(url)
            if r.status_code == 200:
                j = r.json()
                rate = float(((j.get("rates") or {}).get("AUD")) or 0.0)
                if rate > 0.5:
                    _FX_CACHE["usd_aud"] = rate
                    _FX_CACHE["ts"] = now
                    return rate
    except Exception:
        pass
    return 1.50

async 
def _ebay_completed_stats(keyword_query: str, limit: int = 120, days_lookback: int = 120) -> dict:
    """
    Fetch eBay completed/sold items stats using FindingService (AppID only).
    Returns:
      {
        count, currency,
        prices(list),
        low, high, median, avg,
        p20, p80,
        min, max,
        query
      }
    Notes:
      - low/high are p20/p80 (trim outliers)
      - prices are returned in the listing currency (usually USD/AUD)
    """
    q = _norm_ws(keyword_query or "")
    if not (USE_EBAY_API and EBAY_APP_ID and q):
        return {}

    # Cache per query+limit
    cache_key = f"ebay:{q}:{limit}:{days_lookback}"
    now = int(time.time())
    cached = _EBAY_CACHE.get(cache_key)
    if cached and (now - int(cached.get("ts") or 0) < 900):  # 15 min cache
        return cached.get("data") or {}

    # Finding API pagination
    per_page = min(100, max(10, int(limit)))
    max_pages = max(1, int(math.ceil(float(limit) / float(per_page))))
    max_pages = min(max_pages, 5)  # safety cap

    all_prices: list[float] = []
    currency = None
    total_count = 0

    # Use last N days if requested (Finding doesn't support absolute sold date filter cleanly),
    # but we can lightly prefer recency by sorting order and limiting pages.
    for page in range(1, max_pages + 1):
        params = {
            "OPERATION-NAME": "findCompletedItems",
            "SERVICE-VERSION": "1.13.0",
            "SECURITY-APPNAME": EBAY_APP_ID,
            "RESPONSE-DATA-FORMAT": "JSON",
            "REST-PAYLOAD": "true",
            "keywords": q,
            "itemFilter(0).name": "SoldItemsOnly",
            "itemFilter(0).value": "true",
            "paginationInput.entriesPerPage": str(per_page),
            "paginationInput.pageNumber": str(page),
            "sortOrder": "EndTimeSoonest",
        }

        try:
            url = "https://svcs.ebay.com/services/search/FindingService/v1"
            r = httpx.get(url, params=params, timeout=20.0)
            r.raise_for_status()
            j = r.json()
        except Exception:
            continue

        try:
            resp = (j.get("findCompletedItemsResponse") or [{}])[0]
            sr = (resp.get("searchResult") or [{}])[0]
            items = sr.get("item") or []
        except Exception:
            items = []

        if not items:
            break

        for it in items:
            try:
                selling = (it.get("sellingStatus") or [{}])[0]
                cur_price = (selling.get("currentPrice") or [{}])[0]
                val = float(cur_price.get("__value__"))
                cur = cur_price.get("@currencyId")
                if currency is None and cur:
                    currency = str(cur)
                if val > 0:
                    all_prices.append(val)
            except Exception:
                continue

        total_count = len(all_prices)
        if total_count >= limit:
            break

    # Trim to limit
    if len(all_prices) > limit:
        all_prices = all_prices[:limit]

    if not all_prices:
        data = {}
        _EBAY_CACHE[cache_key] = {"ts": now, "data": data}
        return data

    # Robust stats
    prices_sorted = sorted(all_prices)
    n = len(prices_sorted)
    def pct(p: float) -> float:
        if n == 1:
            return float(prices_sorted[0])
        i = (n - 1) * p
        lo = int(math.floor(i))
        hi = int(math.ceil(i))
        if lo == hi:
            return float(prices_sorted[lo])
        w = i - lo
        return float(prices_sorted[lo] * (1 - w) + prices_sorted[hi] * w)

    p20 = pct(0.20)
    p80 = pct(0.80)
    med = pct(0.50)
    avg = float(sum(prices_sorted) / n)
    mn = float(prices_sorted[0])
    mx = float(prices_sorted[-1])

    data = {
        "query": q,
        "count": n,
        "currency": currency or "USD",
        "prices": prices_sorted,
        "p20": round(p20, 2),
        "p80": round(p80, 2),
        "low": round(p20, 2),
        "high": round(p80, 2),
        "median": round(med, 2),
        "avg": round(avg, 2),
        "min": round(mn, 2),
        "max": round(mx, 2),
    }

    _EBAY_CACHE[cache_key] = {"ts": now, "data": data}
    return data


def _pc_db():
    con = sqlite3.connect(PRICECHARTING_DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def _pc_init_db():
    con = _pc_db()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pc_products (
            id TEXT PRIMARY KEY,
            category TEXT,
            product_name TEXT,
            console_name TEXT,
            loose_price REAL,
            cib_price REAL,
            new_price REAL,
            graded_price REAL,
            raw_json TEXT,
            updated_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pc_price_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id TEXT NOT NULL,
            category TEXT,
            snap_date TEXT NOT NULL,
            loose_price REAL,
            cib_price REAL,
            new_price REAL,
            graded_price REAL,
            source TEXT,
            row_hash TEXT,
            raw_json TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pc_snapshots_pid_date ON pc_price_snapshots(product_id, snap_date)""")
    con.commit()
    con.close()

_pc_init_db()

def _pc_row_hash(row: dict) -> str:
    try:
        stable = json.dumps(row, sort_keys=True, separators=(",", ":"))
    except Exception:
        stable = str(row)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()

def _pc_upsert_product(row: dict, category: str, now_iso: str):
    """Upsert latest product row into pc_products."""
    pid = str(row.get("id") or row.get("product-id") or row.get("product_id") or "").strip()
    if not pid:
        return

    def f(k):
        v = row.get(k)
        if v in (None, "", "null", "NULL"):
            return None
        try:
            return float(str(v).replace("$", "").replace(",", "").strip())
        except Exception:
            return None

    product_name = (row.get("product-name") or row.get("product_name") or row.get("name") or "").strip()
    console_name = (row.get("console-name") or row.get("console_name") or row.get("console") or "").strip()

    loose = f("loose-price") or f("loose_price")
    cib = f("cib-price") or f("cib_price")
    newp = f("new-price") or f("new_price")
    graded = f("graded-price") or f("graded_price")

    con = _pc_db()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO pc_products (id, category, product_name, console_name, loose_price, cib_price, new_price, graded_price, raw_json, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            category=excluded.category,
            product_name=excluded.product_name,
            console_name=excluded.console_name,
            loose_price=excluded.loose_price,
            cib_price=excluded.cib_price,
            new_price=excluded.new_price,
            graded_price=excluded.graded_price,
            raw_json=excluded.raw_json,
            updated_at=excluded.updated_at
        """,
        (pid, category, product_name, console_name, loose, cib, newp, graded, json.dumps(row), now_iso)
    )
    con.commit()
    con.close()

def _pc_insert_snapshot(row: dict, category: str, snap_date: str, source: str):
    pid = str(row.get("id") or row.get("product-id") or row.get("product_id") or "").strip()
    if not pid:
        return

    def f(k):
        v = row.get(k)
        if v in (None, "", "null", "NULL"):
            return None
        try:
            return float(str(v).replace("$", "").replace(",", "").strip())
        except Exception:
            return None

    loose = f("loose-price") or f("loose_price")
    cib = f("cib-price") or f("cib_price")
    newp = f("new-price") or f("new_price")
    graded = f("graded-price") or f("graded_price")

    rh = _pc_row_hash({"pid": pid, "snap": snap_date, "loose": loose, "cib": cib, "new": newp, "graded": graded})

    con = _pc_db()
    cur = con.cursor()
    # Dedupe: do not insert identical row_hash for same date
    cur.execute(
        """
        SELECT 1 FROM pc_price_snapshots
        WHERE product_id=? AND snap_date=? AND row_hash=?
        LIMIT 1
        """,
        (pid, snap_date, rh)
    )
    if cur.fetchone():
        con.close()
        return

    cur.execute(
        """
        INSERT INTO pc_price_snapshots (product_id, category, snap_date, loose_price, cib_price, new_price, graded_price, source, row_hash, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (pid, category, snap_date, loose, cib, newp, graded, source, rh, json.dumps(row))
    )
    con.commit()
    con.close()

def _pc_trend(product_id: str, days: int = 30) -> dict:
    """Compute a basic trend based on snapshot 'new_price' if present, else loose_price."""
    pid = (product_id or "").strip()
    if not pid:
        return {"available": False}

    con = _pc_db()
    cur = con.cursor()
    cur.execute(
        """
        SELECT snap_date, new_price, loose_price, graded_price
        FROM pc_price_snapshots
        WHERE product_id=?
        ORDER BY snap_date ASC
        """,
        (pid,)
    )
    rows = cur.fetchall()
    con.close()

    if not rows or len(rows) < 2:
        return {"available": False}

    # Convert to (date, value)
    series = []
    for r in rows:
        v = r["new_price"]
        if v is None:
            v = r["loose_price"]
        if v is None:
            continue
        series.append((r["snap_date"], float(v)))

    if len(series) < 2:
        return {"available": False}

    # Use last value and value from ~days ago (closest earlier)
    last_date, last_val = series[-1]
    # find earliest >= days back: since we store weekly, just pick first in window
    from datetime import datetime, timedelta
    try:
        last_dt = datetime.fromisoformat(last_date.replace("Z",""))
    except Exception:
        last_dt = None

    baseline_val = series[0][1]
    baseline_date = series[0][0]
    if last_dt:
        cutoff = (last_dt - timedelta(days=days)).date()
        for d, v in series:
            try:
                dt = datetime.fromisoformat(d.replace("Z","")).date()
            except Exception:
                continue
            if dt >= cutoff:
                baseline_date, baseline_val = d, v
                break

    change = last_val - baseline_val
    pct = (change / baseline_val) if baseline_val else None
    label = "stable"
    if pct is not None:
        if pct > 0.10:
            label = "increasing"
        elif pct < -0.10:
            label = "decreasing"

    return {
        "available": True,
        "window_days": days,
        "baseline": {"date": baseline_date, "value": round(baseline_val, 2)},
        "latest": {"date": last_date, "value": round(last_val, 2)},
        "change": round(change, 2),
        "change_pct": round(pct, 4) if pct is not None else None,
        "label": label,
    }


POKEMONTCG_BASE = "https://api.pokemontcg.io/v2"

# ==============================
# Set Code Mapping + Canonicalization
# ==============================
SET_CODE_MAP: Dict[str, str] = {
    "MEW": "Scarlet & Violet-151",
    "OBF": "Obsidian Flames",
    "SVI": "Scarlet & Violet",
    "PFL": "Phantasmal Flames",
    "BS": "Base Set",
}

def _canonicalize_set(set_code: Optional[str], set_name: Optional[str]) -> Dict[str, str]:
    sc = (set_code or "").strip().upper()
    sn = _norm_ws(set_name or "")
    mapped = SET_CODE_MAP.get(sc) if sc else None
    if mapped:
        return {"set_code": sc, "set_name": mapped, "set_source": "code_map"}
    if sc and sn:
        return {"set_code": sc, "set_name": sn, "set_source": "provided"}
    if sn:
        return {"set_code": sc or "", "set_name": sn, "set_source": "name_only"}
    if sc:
        return {"set_code": sc, "set_name": "", "set_source": "code_only"}
    return {"set_code": "", "set_name": "", "set_source": "unknown"}



# eBay API scaffolding (disabled by default)
USE_EBAY_API = os.getenv("USE_EBAY_API", "0").strip() in ("1", "true", "TRUE", "yes", "YES")
EBAY_APP_ID = os.getenv("EBAY_APP_ID", "").strip()
EBAY_OAUTH_TOKEN = os.getenv("EBAY_OAUTH_TOKEN", "").strip()

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set! Vision ID/assessment will fail.")
if not POKEMONTCG_API_KEY:
    print("INFO: POKEMONTCG_API_KEY not set (Pokemon enrichment disabled).")
if not PRICECHARTING_TOKEN:
    print("INFO: PRICECHARTING_TOKEN not set (PriceCharting pricing disabled).")
if USE_EBAY_API and not EBAY_APP_ID:
    print("INFO: USE_EBAY_API=1 set but eBay credentials are missing (will remain inactive).")

# ==============================
# Generic helpers
# ==============================
def _b64(img: bytes) -> str:
    return base64.b64encode(img).decode("utf-8")


def _make_defect_filter_variants(img_bytes: bytes) -> Dict[str, bytes]:
    """Create enhanced variants to help the second-pass detect print lines / scratches / whitening.
    Uses PIL if available. Returns dict name->jpeg_bytes. If PIL missing, returns empty dict.
    """
    if not img_bytes or Image is None:
        return {}
    try:
        from io import BytesIO
        im = Image.open(BytesIO(img_bytes)).convert("RGB")
        variants: Dict[str, bytes] = {}

        # Variant 1: grayscale + autocontrast (good for print lines / scratches)
        g = ImageOps.grayscale(im)
        g = ImageOps.autocontrast(g)
        g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        buf = BytesIO()
        g.save(buf, format="JPEG", quality=92)
        variants["gray_autocontrast"] = buf.getvalue()

        # Variant 2: boosted contrast + sharpness (good for edge wear and whitening)
        c = ImageEnhance.Contrast(im).enhance(1.6)
        c = ImageEnhance.Sharpness(c).enhance(1.8)
        c = c.filter(ImageFilter.UnsharpMask(radius=1, percent=130, threshold=2))
        buf = BytesIO()
        c.save(buf, format="JPEG", quality=92)
        variants["contrast_sharp"] = buf.getvalue()

        return variants
    except Exception:
        return {}

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _is_blankish(s: str) -> bool:
    s2 = _norm_ws(s or "").lower()
    return (not s2) or s2 in ("unknown", "n/a", "na", "none", "null", "undefined")

def _normalize_card_type(card_type: str) -> str:
    """Force card_type into the allowed enum."""
    s = _norm_ws(card_type or "").lower()
    if not s:
        return "Other"
    # common variants
    if "pokemon" in s or s in ("pkmn", "poke", "pokémon"):
        return "Pokemon"
    if "magic" in s or "mtg" in s:
        return "Magic"
    if "yug" in s or "yu-gi" in s or "yugi" in s:
        return "YuGiOh"
    if "one piece" in s or "onepiece" in s:
        return "OnePiece"
    if "sport" in s:
        return "Sports"
    if s in ("other", "other tcg", "tcg", "trading card", "tradingcard"):
        return "Other"
    # fallback: title-case but keep enum
    return "Other"


def _is_blankish(s: str) -> bool:
    s=(s or "").strip().lower()
    return s in ("", "unknown", "n/a", "na", "none", "not sure", "unsure")

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

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

def _clean_card_number_display(s: str) -> str:
    s = (s or "").strip().replace("#", "").strip()
    if not s:
        return ""
    return re.sub(r"\s+", "", s)  # keep leading zeros

def _card_number_for_query(s: str) -> str:
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

# ==============================
# OpenAI helper (vision)
# ==============================
async def _openai_chat(messages: List[Dict[str, Any]], max_tokens: int = 1200, temperature: float = 0.1) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    url = "https://api.openai.com/v1/chat/completions"
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "response_format": {"type": "json_object"}}

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code != 200:
                return {"error": True, "status": r.status_code, "message": r.text[:700]}
            data = r.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return {"error": False, "content": content}
    except Exception as e:
        return {"error": True, "status": 0, "message": str(e)}


async def _fetch_html(url: str) -> str:
    """Fetch text content from a URL (used for CSV downloads)."""
    try:
        async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": UA}) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return ""
            return r.text or ""
    except Exception:
        return ""

# ==============================
# PokemonTCG helpers (ID/metadata only)
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

async def _pokemontcg_resolve_card_id(card_name: str, set_code: str, card_number_display: str, set_name: str = "", set_id: str = "") -> str:
    if not POKEMONTCG_API_KEY:
        return ""
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

    for q in queries:
        data = await _pokemontcg_get("/cards", params={"q": q, "pageSize": 25})
        cards = data.get("data") or []
        if not cards:
            continue
        if card_number_display:
            want = num_q.split("/")[0] if num_q else ""
            for c in cards:
                if _same_number(str(c.get("number", "")), want):
                    return str(c.get("id", "")) or ""
        return str(cards[0].get("id", "")) or ""
    return ""

async def _pokemontcg_card_by_id(card_id: str) -> Dict[str, Any]:
    if not card_id:
        return {}
    data = await _pokemontcg_get(f"/cards/{card_id}")
    card = data.get("data") if isinstance(data, dict) else None
    return card if isinstance(card, dict) else {}

# ==============================
# PriceCharting API (pricing + downloadable CSV snapshots)
# Docs: https://www.pricecharting.com/api-documentation
# NOTE: PriceCharting prices are returned as integer pennies/cents.
# ==============================
PRICECHARTING_BASE = "https://www.pricecharting.com/api"

async def _pc_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not PRICECHARTING_TOKEN:
        return {}
    params = params or {}
    params["t"] = PRICECHARTING_TOKEN
    url = path if path.startswith("http") else f"{PRICECHARTING_BASE}{path}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, params=params, headers={"User-Agent": UA})
            if r.status_code != 200:
                return {}
            return r.json() if r.content else {}
    except Exception:
        return {}

def _pc_money(pennies: Any) -> Optional[float]:
    try:
        p = int(pennies)
        return round(p / 100.0, 2)
    except Exception:
        return None

def _pc_extract_price_fields(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Convert any *-price fields to floats. Keeps raw too."""
    out: Dict[str, Any] = {}
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        if k.endswith("-price") or k.endswith("_price") or k in ("price", "used_price", "new_price"):
            mv = _pc_money(v)
            if mv is not None:
                out[k] = mv
        # Keep certain identifiers
        if k in ("id", "product-id", "console-name", "product-name", "url", "category-name", "variant-name", "edition-name"):
            out[k] = v
    return out

async def _pc_search(q: str, category: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    if not PRICECHARTING_TOKEN:
        return []
    params = {"q": q}
    if category:
        params["category"] = category
    data = await _pc_get("/products", params=params)
    products = data.get("products") or data.get("data") or []
    if not isinstance(products, list):
        return []
    return products[: max(1, min(50, limit))]

async def _pc_product(product_id: str) -> Dict[str, Any]:
    if not product_id:
        return {}
    data = await _pc_get("/product", params={"id": product_id})
    # response shape can vary; try common keys
    if isinstance(data, dict):
        return data.get("product") or data
    return {}


# ==============================
# FX helper (USD -> AUD)
# ==============================
# User requested a simple constant multiplier for display (fast + reliable).
AUD_MULTIPLIER = float(os.getenv("CL_USD_TO_AUD_MULTIPLIER", "1.44"))

def _usd_to_aud_simple(amount: Any) -> Optional[float]:
    try:
        v = float(amount)
    except Exception:
        return None
    return round(v * AUD_MULTIPLIER, 2)


def _stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"avg": 0, "median": 0, "low": 0, "high": 0, "sample_size": 0}
    v = sorted(values)
    return {
        "avg": round(mean(v), 2),
        "median": round(median(v), 2),
        "low": round(min(v), 2),
        "high": round(max(v), 2),
        "sample_size": len(v),
    }

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


def _condition_bucket_from_pregrade(g: Optional[int]) -> str:
    """Map numeric pregrade into a human-friendly condition bucket."""
    if g is None:
        return "unknown"
    try:
        gi = int(g)
    except Exception:
        return "unknown"
    if gi >= 9:
        return "mint_like"
    if gi >= 7:
        return "near_mint"
    if gi >= 5:
        return "excellent"
    if gi >= 3:
        return "good"
    return "poor"

# ==============================
# Root & Health
# ==============================
@app.get("/")
def root():
    return {"status": "ok", "service": "cl-scan-api", "version": APP_VERSION}

@app.head("/")
def head_root():
    return Response(status_code=200)

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "cl-scan-api",
        "version": APP_VERSION,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_pokemontcg_key": bool(POKEMONTCG_API_KEY),
        "has_pricecharting_token": bool(PRICECHARTING_TOKEN),
        "use_ebay_api": bool(USE_EBAY_API and EBAY_APP_ID),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "allowed_origins": ALLOWED_ORIGINS,
        "supports": ["cards", "memorabilia", "sealed_products", "market_context_click_only"],
    }

@app.head("/health")
def head_health():
    return Response(status_code=200)


# ==============================
# Admin: PriceCharting weekly sync (CSV) + history
# ==============================
@app.post("/api/admin/pricecharting/sync")
@safe_endpoint
async def pricecharting_sync(
    category: str = Form("pokemon-cards"),
    snapshot_date: Optional[str] = Form(None),
    max_rows: int = Form(200000),
    admin_token: Optional[str] = Form(None),
):
    """
    Downloads your PriceCharting custom CSV for a category and stores:
    - latest product row in pc_products
    - snapshot row in pc_price_snapshots (deduped)
    Security:
      - pass admin_token in body OR set header X-Admin-Token.
      - set CL_ADMIN_TOKEN env to enable. If not set, endpoint returns 403.
    """
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Admin sync disabled (CL_ADMIN_TOKEN not set)")

    # accept token from form OR header
    from fastapi import Request
    request: Request = kwargs.get("request") if "kwargs" in locals() else None  # not used
    # FastAPI doesn't inject Request here by default; read from header via dependency-free approach:
    # We'll use httpx to get headers? not possible. So accept admin_token param.
    token = (admin_token or "").strip()
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")

    if not PRICECHARTING_TOKEN:
        raise HTTPException(status_code=400, detail="PRICECHARTING_TOKEN not configured")

    cat = (category or "pokemon-cards").strip()
    snap = (snapshot_date or "").strip()
    if not snap:
        snap = datetime.utcnow().date().isoformat() + "T00:00:00Z"

    url = f"https://www.pricecharting.com/price-guide/download-custom?t={PRICECHARTING_TOKEN}&category={cat}"
    csv_text = await _fetch_html(url)
    if not csv_text or len(csv_text) < 100:
        return JSONResponse(content={
            "ok": False,
            "message": "Failed to download CSV (empty response). Check token/category.",
            "url": url,
        }, status_code=502)

    # Save raw CSV
    fname = os.path.join(PRICECHARTING_CACHE_DIR, f"{cat}-{snap[:10]}.csv")
    try:
        with open(fname, "w", encoding="utf-8", newline="") as f:
            f.write(csv_text)
    except Exception:
        pass

    # Parse
    reader = csv.DictReader(csv_text.splitlines())
    n = 0
    now_iso = datetime.utcnow().isoformat() + "Z"
    for row in reader:
        n += 1
        _pc_upsert_product(row, cat, now_iso)
        _pc_insert_snapshot(row, cat, snap, source="pricecharting_csv")
        if n >= int(max_rows):
            break

    return {
        "ok": True,
        "category": cat,
        "snapshot_date": snap,
        "rows_processed": n,
        "db_path": PRICECHARTING_DB_PATH,
        "saved_csv": os.path.basename(fname) if 'fname' in locals() else None,
    }

@app.get("/api/pricecharting/trend")
@safe_endpoint
async def pricecharting_trend(product_id: str, days: int = 30):
    return _pc_trend(product_id, int(days or 30))

@app.get("/api/pricecharting/history")
@safe_endpoint
async def pricecharting_history(product_id: str, limit: int = 52):
    pid = (product_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="product_id required")

    con = _pc_db()
    cur = con.cursor()
    cur.execute(
        """
        SELECT snap_date, loose_price, cib_price, new_price, graded_price, source
        FROM pc_price_snapshots
        WHERE product_id=?
        ORDER BY snap_date DESC
        LIMIT ?
        """,
        (pid, int(limit or 52))
    )
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return {"ok": True, "product_id": pid, "rows": rows}


# ==============================
# Card: Identify (AI + optional PokemonTCG enrichment)
# ==============================
@app.post("/api/identify")
@safe_endpoint
async def identify(front: UploadFile = File(...)):
    img = await front.read()
    if not img or len(img) < 200:
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
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(img)}", "detail": "high"}},
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
            "pokemontcg": None,
            "pokemontcg_enriched": False,
        })

    data = _parse_json_or_none(result.get("content", "")) or {}

    # ------------------------------
    # Second-pass (defect enhanced) analysis
    # ------------------------------
    second_pass = {"enabled": True, "ran": False, "skipped_reason": None, "glare_suspects": [], "defect_candidates": []}
    try:
        # Only run for cards; memorabilia uses a different endpoint.
        front_vars = _make_defect_filter_variants(front_bytes)
        back_vars = _make_defect_filter_variants(back_bytes)

        if not front_vars and not back_vars:
            second_pass["enabled"] = False
            second_pass["skipped_reason"] = "image_filters_unavailable"
        else:
            sp_prompt = f"""You are a meticulous trading card defect inspector.
You are analyzing ENHANCED/FILTERED variants created to make defects stand out (print lines, scratches, whitening, dents).
You may also receive an ANGLED image used to rule out glare/light refraction.

CRITICAL:
- Do NOT treat holo sheen / glare as whitening. If the angled shot suggests the mark moves/disappears, flag it as glare_suspect.
- Card texture is NOT damage unless there is a true crease/indent/paper break.
- Print lines are typically straight and consistent; glare moves with angle.

Return ONLY valid JSON:
{{
  "defect_candidates": [
    {{"type":"print_line|scratch|whitening|dent|crease|tear|stain|other","severity":"minor|moderate|severe","note":"short plain description","confidence":0-1}}
  ],
  "glare_suspects": [
    {{"type":"whitening|scratch|print_line|other","note":"why it looks like glare","confidence":0-1}}
  ]
}}
"""

            content_parts = [{"type": "text", "text": sp_prompt}]

            # Include filtered variants first (strong signal)
            if front_vars.get("gray_autocontrast"):
                content_parts += [
                    {"type":"text","text":"FRONT (filtered: gray_autocontrast)"},
                    {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{_b64(front_vars['gray_autocontrast'])}", "detail":"high"}}
                ]
            if front_vars.get("contrast_sharp"):
                content_parts += [
                    {"type":"text","text":"FRONT (filtered: contrast_sharp)"},
                    {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{_b64(front_vars['contrast_sharp'])}", "detail":"high"}}
                ]
            if back_vars.get("gray_autocontrast"):
                content_parts += [
                    {"type":"text","text":"BACK (filtered: gray_autocontrast)"},
                    {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{_b64(back_vars['gray_autocontrast'])}", "detail":"high"}}
                ]
            if back_vars.get("contrast_sharp"):
                content_parts += [
                    {"type":"text","text":"BACK (filtered: contrast_sharp)"},
                    {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{_b64(back_vars['contrast_sharp'])}", "detail":"high"}}
                ]

            # Include angled if available (glare check)
            if angled_bytes and len(angled_bytes) > 200:
                content_parts += [
                    {"type":"text","text":"OPTIONAL ANGLED IMAGE (glare check)"},
                    {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{_b64(angled_bytes)}", "detail":"high"}}
                ]

            sp_msg = [{"role":"user","content": content_parts}]
            sp_result = await _openai_chat(sp_msg, max_tokens=900, temperature=0.1)
            if not sp_result.get("error"):
                sp_data = _parse_json_or_none(sp_result.get("content","")) or {}
                if isinstance(sp_data, dict):
                    second_pass["ran"] = True
                    if isinstance(sp_data.get("glare_suspects"), list):
                        second_pass["glare_suspects"] = sp_data.get("glare_suspects")[:10]
                    if isinstance(sp_data.get("defect_candidates"), list):
                        second_pass["defect_candidates"] = sp_data.get("defect_candidates")[:20]
            else:
                second_pass["skipped_reason"] = "second_pass_ai_failed"
    except Exception:
        second_pass["skipped_reason"] = "second_pass_exception"

    card_name = _norm_ws(str(data.get("card_name", "Unknown")))
    card_type = _norm_ws(str(data.get("card_type", "Other")))
    card_type = _normalize_card_type(card_type)
    year = _norm_ws(str(data.get("year", "")))
    card_number_display = _clean_card_number_display(str(data.get("card_number", "")))
    set_code = _norm_ws(str(data.get("set_code", ""))).upper()
    set_name = _norm_ws(str(data.get("set_name", "")))
    if _is_blankish(set_name):
        set_name = ""
    if set_name.strip().lower() in ("unknown","n/a","na","none"):
        set_name = ""
    conf = _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0)
    notes = _norm_ws(str(data.get("notes", "")))

    # If the model says "Unknown", force confidence low.
    if card_name.strip().lower() == "unknown":
        conf = 0.0
        card_type = "Other"

    # optional PokemonTCG enrichment (Pokemon only)
    enriched = False
    ptcg_payload = None
    set_id = ""

    if POKEMONTCG_API_KEY and card_type.lower() == "pokemon" and card_name != "Unknown":
        ptcg_set = {}
        if set_code and not set_name:
            ptcg_set = await _pokemontcg_resolve_set_by_ptcgo(set_code)
            if ptcg_set.get("name"):
                set_name = _norm_ws(str(ptcg_set.get("name", "")))
                set_id = str(ptcg_set.get("id", ""))

        pid = await _pokemontcg_resolve_card_id(card_name, set_code, card_number_display, set_name=set_name, set_id=set_id)
        if pid:
            card = await _pokemontcg_card_by_id(pid)
            if card:
                enriched = True
                conf = min(1.0, conf + 0.15)
                set_name = _norm_ws(str((card.get("set") or {}).get("name", set_name)))
                set_code = _norm_ws(str((card.get("set") or {}).get("ptcgoCode", set_code))).upper()
                set_id = _norm_ws(str((card.get("set") or {}).get("id", set_id)))
                release_date = _norm_ws(str((card.get("set") or {}).get("releaseDate", "")))
                if release_date and not year:
                    year = release_date[:4]
                # keep the AI card_number display if present, else use API number
                if not card_number_display:
                    card_number_display = _clean_card_number_display(str(card.get("number", "")))
                ptcg_payload = {
                    "id": card.get("id", ""),
                    "name": card.get("name", ""),
                    "number": card.get("number", ""),
                    "rarity": card.get("rarity", ""),
                    "set": card.get("set", {}) or {},
                    "images": card.get("images", {}) or {},
                    "links": {
                        "tcgplayer": (card.get("tcgplayer") or {}).get("url", ""),
                        "cardmarket": (card.get("cardmarket") or {}).get("url", ""),
                    },
                }
                notes = f"Verified via PokemonTCG API (id {pid})"

    
    # Canonicalize set (prefer map / resolved name) + build canonical id for frontend
    set_info = _canonicalize_set(set_code, set_name)
    canonical_id = {
        "card_name": card_name,
        "card_type": card_type,
        "year": year,
        "card_number": card_number_display,
        "set_code": set_info["set_code"],
        "set_name": set_info["set_name"],
        "set_source": set_info.get("set_source", "unknown"),
        "confidence": conf,
    }

    return JSONResponse(content={
        "card_name": card_name,
        "card_type": card_type,
        "year": year,
        "card_number": card_number_display,
        "set_code": set_info["set_code"],
        "set_name": set_info["set_name"],
        "set_id": set_id,
        "confidence": conf,
        "notes": notes,
        "canonical_id": canonical_id,
        "identify_token": f"idt_{secrets.token_urlsafe(12)}",
        "pokemontcg": ptcg_payload,
        "pokemontcg_enriched": enriched,
    })

# ==============================
# Card: Verify (grading only)
# ==============================
@app.post("/api/verify")
@safe_endpoint
async def verify(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    angled: Optional[UploadFile] = File(None),  # new: angled/glare-check shot
    card_name: Optional[str] = Form(None),
    card_set: Optional[str] = Form(None),
    card_number: Optional[str] = Form(None),
    card_year: Optional[str] = Form(None),
    card_type: Optional[str] = Form(None),
    set_code: Optional[str] = Form(None),
):
    front_bytes = await front.read()
    back_bytes = await back.read()
    angled_bytes = await angled.read() if angled is not None else b""
    if not front_bytes or not back_bytes or len(front_bytes) < 200 or len(back_bytes) < 200:
        raise HTTPException(status_code=400, detail="Images are too small or empty")

    provided_name = _norm_ws(card_name or "")
    provided_set = _norm_ws(card_set or "")
    if _is_blankish(provided_set):
        provided_set = ""
    provided_num_display = _clean_card_number_display(card_number or "")
    provided_year = _norm_ws(card_year or "")
    provided_type = _normalize_card_type(_norm_ws(card_type or ""))
    provided_code = _norm_ws(set_code or "").upper()

    # Backfill set name if needed
    if POKEMONTCG_API_KEY and provided_code and (not provided_set or provided_set.lower()=='unknown'):
        ptcg_set = await _pokemontcg_resolve_set_by_ptcgo(provided_code)
        if ptcg_set.get("name"):
            provided_set = _norm_ws(str(ptcg_set.get("name", "")))

    context = ""
    if provided_name or provided_set or provided_num_display or provided_year or provided_type or provided_code:
        context = "\n\nKNOWN/PROVIDED CARD DETAILS (use as hints, do not force if images contradict):\n"
        if provided_name: context += f"- Card Name: {provided_name}\n"
        if provided_set: context += f"- Set Name: {provided_set}\n"
        if provided_code: context += f"- Set Code: {provided_code}\n"
        if provided_num_display: context += f"- Card Number: {provided_num_display}\n"
        if provided_year: context += f"- Year: {provided_year}\n"
        if provided_type: context += f"- Type: {provided_type}\n"

    prompt = f"""You are a professional trading card grader with 15+ years experience.

Analyze the provided images with EXTREME scrutiny.
You will receive FRONT and BACK images, and MAY receive a third ANGLED image used to rule out glare / light refraction artifacts (holo sheen) vs true whitening / scratches / print lines. Write as if speaking directly to a collector who needs honest, specific feedback about their card.

CRITICAL RULES:
1) **Be conversational and specific.** Write like you're examining the card in person and describing what you see:
   - BAD: "Minor edge wear present"
   - GOOD: "Looking at the front, I can see some very slight edge wear along the top edge, approximately 2mm from the top-left corner. The right edge is notably cleaner."

2) **Call out every single corner individually** with precise location and severity:
   - For EACH of the 8 corners (4 front + 4 back), describe what you observe
   - Examples: "Front top-left corner is perfectly sharp", "Back bottom-right shows minor whitening about 1mm deep"

3) **Grade must reflect worst visible defect** (conservative PSA-style):
   - Any crease/fold/tear/major dent → pregrade **4 or lower**
   - Any bend/ding/impression, heavy rounding → pregrade **5 or lower**
   - Moderate whitening across multiple corners/edges → pregrade **6-7**
   - Only grade 9-10 if truly exceptional


5) **Do NOT confuse holo sheen / light refraction / texture for damage**:
   - If a mark disappears or changes drastically in the ANGLED shot, treat it as glare/reflection, NOT whitening/damage.
   - Print lines are typically straight and consistent across lighting; glare moves with angle.
   - Card texture (especially modern) is not damage unless there is a true crease, indentation, or paper break.

4) **Write the assessment summary in first person, conversational style** (5-8 sentences):
   - Open with overall impression: "Looking at your card..."
   - Discuss specific observations: "The front presents beautifully, with..."
   - Compare front vs back: "While the front is near-perfect, the back shows..."
   - Explain grade rationale: "The grade of X is primarily limited by..."
   - End with realistic expectation: "If you're considering grading..."

{context}

Return ONLY valid JSON with this EXACT structure:

{{
  "pregrade": "1-10",
  "confidence": 0.0-1.0,
  "centering": {{
    "front": {{
      "grade": "55/45",
      "notes": "Detailed observation: slightly off-center towards [direction], approximately [measurement]. [Impact on grade]."
    }},
    "back": {{
      "grade": "60/40", 
      "notes": "Detailed observation: [specific description of centering quality]."
    }}
  }},
  "corners": {{
    "front": {{
      "top_left": {{
        "condition": "sharp/minor_whitening/whitening/bend/ding/crease",
        "notes": "Specific description: [exactly what you see, be detailed]"
      }},
      "top_right": {{
        "condition": "sharp/minor_whitening/whitening/bend/ding/crease",
        "notes": "Specific description: [exactly what you see]"
      }},
      "bottom_left": {{
        "condition": "sharp/minor_whitening/whitening/bend/ding/crease",
        "notes": "Specific description: [exactly what you see]"
      }},
      "bottom_right": {{
        "condition": "sharp/minor_whitening/whitening/bend/ding/crease",
        "notes": "Specific description: [exactly what you see]"
      }}
    }},
    "back": {{
      "top_left": {{"condition": "...", "notes": "..."}},
      "top_right": {{"condition": "...", "notes": "..."}},
      "bottom_left": {{"condition": "...", "notes": "..."}},
      "bottom_right": {{"condition": "...", "notes": "..."}}
    }}
  }},
  "edges": {{
    "front": {{
      "grade": "Mint/Near Mint/Excellent/Good/Poor",
      "notes": "Walk around all 4 edges: top edge [description], right edge [description], bottom [description], left [description]. Be specific about location and severity."
    }},
    "back": {{
      "grade": "Mint/Near Mint/Excellent/Good/Poor",
      "notes": "Detailed edge-by-edge assessment with locations."
    }}
  }},
  "surface": {{
    "front": {{
      "grade": "Mint/Near Mint/Excellent/Good/Poor",
      "notes": "Describe surface quality in detail: holographic pattern quality, any print lines, scratches (with location), scuffs, gloss level."
    }},
    "back": {{
      "grade": "Mint/Near Mint/Excellent/Good/Poor",
      "notes": "Detailed surface assessment."
    }}
  }},
  "defects": [
    "Each defect as a complete sentence: [SIDE] [precise location] shows [type of defect] [severity]. Example: 'Front top-left corner shows moderate whitening extending approximately 2mm into the card surface.'"
  ],
  "flags": [
    "Short flags for important issues (crease, bend, edge chipping, etc.)"
  ],
  "assessment_summary": "Write 5-8 sentences in first person, conversational style. Start with: 'Looking at your [card name]...' Then describe specific observations, compare front vs back, explain what limits the grade, and give realistic grading expectations. Be honest but professional.",
    "spoken_word": "A punchy spoken-word version of the assessment summary (about 20-45 seconds). First person, conversational. Mention the best features, the main grade limiters, and end with what grade you’d realistically expect.",
  "observed_id": {{
    "card_name": "best-effort from images",
    "set_code": "only if clearly visible",
    "set_name": "best-effort",
    "card_number": "preserve leading zeros",
    "year": "best-effort",
    "card_type": "Pokemon/Magic/YuGiOh/Sports/OnePiece/Other"
  }}
}}

CRITICAL REMINDERS:
- Every corner needs a detailed note explaining what you observe
- Every edge/surface needs location-specific observations  
- Assessment summary must be conversational (first person, like talking to the owner)
- Do NOT miss obvious damage - be brutally honest
- If you can't see something clearly due to glare/blur, say so in notes

Respond ONLY with JSON, no extra text.
"""

    msg = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(front_bytes)}", "detail": "high"}},
            {"type": "text", "text": "FRONT IMAGE ABOVE ☝️ | BACK IMAGE BELOW 👇"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(back_bytes)}", "detail": "high"}},
        ] + (
            [
                {"type": "text", "text": "OPTIONAL ANGLED IMAGE BELOW (use to rule out glare vs true defects) 👇"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(angled_bytes)}", "detail": "high"}},
            ] if angled_bytes and len(angled_bytes) > 200 else []
        ),
    }]

    result = await _openai_chat(msg, max_tokens=2200, temperature=0.1)
    if result.get("error"):
        return JSONResponse(content={"error": True, "message": "AI grading failed", "details": result.get("message", "")}, status_code=502)

    data = _parse_json_or_none(result.get("content", "")) or {}

    # ------------------------------
    # Second-pass (defect enhanced) analysis
    # ------------------------------
    second_pass = {"enabled": True, "ran": False, "skipped_reason": None, "glare_suspects": [], "defect_candidates": []}
    try:
        # Only run for cards; memorabilia uses a different endpoint.
        front_vars = _make_defect_filter_variants(front_bytes)
        back_vars = _make_defect_filter_variants(back_bytes)

        if not front_vars and not back_vars:
            second_pass["enabled"] = False
            second_pass["skipped_reason"] = "image_filters_unavailable"
        else:
            sp_prompt = f"""You are a meticulous trading card defect inspector.
You are analyzing ENHANCED/FILTERED variants created to make defects stand out (print lines, scratches, whitening, dents).
You may also receive an ANGLED image used to rule out glare/light refraction.

CRITICAL:
- Do NOT treat holo sheen / glare as whitening. If the angled shot suggests the mark moves/disappears, flag it as glare_suspect.
- Card texture is NOT damage unless there is a true crease/indent/paper break.
- Print lines are typically straight and consistent; glare moves with angle.

Return ONLY valid JSON:
{{
  "defect_candidates": [
    {{"type":"print_line|scratch|whitening|dent|crease|tear|stain|other","severity":"minor|moderate|severe","note":"short plain description","confidence":0-1}}
  ],
  "glare_suspects": [
    {{"type":"whitening|scratch|print_line|other","note":"why it looks like glare","confidence":0-1}}
  ]
}}
"""

            content_parts = [{"type": "text", "text": sp_prompt}]

            # Include filtered variants first (strong signal)
            if front_vars.get("gray_autocontrast"):
                content_parts += [
                    {"type":"text","text":"FRONT (filtered: gray_autocontrast)"},
                    {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{_b64(front_vars['gray_autocontrast'])}", "detail":"high"}}
                ]
            if front_vars.get("contrast_sharp"):
                content_parts += [
                    {"type":"text","text":"FRONT (filtered: contrast_sharp)"},
                    {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{_b64(front_vars['contrast_sharp'])}", "detail":"high"}}
                ]
            if back_vars.get("gray_autocontrast"):
                content_parts += [
                    {"type":"text","text":"BACK (filtered: gray_autocontrast)"},
                    {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{_b64(back_vars['gray_autocontrast'])}", "detail":"high"}}
                ]
            if back_vars.get("contrast_sharp"):
                content_parts += [
                    {"type":"text","text":"BACK (filtered: contrast_sharp)"},
                    {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{_b64(back_vars['contrast_sharp'])}", "detail":"high"}}
                ]

            # Include angled if available (glare check)
            if angled_bytes and len(angled_bytes) > 200:
                content_parts += [
                    {"type":"text","text":"OPTIONAL ANGLED IMAGE (glare check)"},
                    {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{_b64(angled_bytes)}", "detail":"high"}}
                ]

            sp_msg = [{"role":"user","content": content_parts}]
            sp_result = await _openai_chat(sp_msg, max_tokens=900, temperature=0.1)
            if not sp_result.get("error"):
                sp_data = _parse_json_or_none(sp_result.get("content","")) or {}
                if isinstance(sp_data, dict):
                    second_pass["ran"] = True
                    if isinstance(sp_data.get("glare_suspects"), list):
                        second_pass["glare_suspects"] = sp_data.get("glare_suspects")[:10]
                    if isinstance(sp_data.get("defect_candidates"), list):
                        second_pass["defect_candidates"] = sp_data.get("defect_candidates")[:20]
            else:
                second_pass["skipped_reason"] = "second_pass_ai_failed"
    except Exception:
        second_pass["skipped_reason"] = "second_pass_exception"

    # Normalize flags/defects and compute structural-damage indicator
    flags_raw = data.get("flags", [])
    if isinstance(flags_raw, list):
        flags_list_out = [str(f).lower().strip() for f in flags_raw if str(f).strip()]
    else:
        flags_list_out = []
    # De-duplicate while preserving order
    flags_list_out = list(dict.fromkeys(flags_list_out))

    defects_list_out = data.get("defects", [])
    if not isinstance(defects_list_out, list):
        defects_list_out = []

    
    # Normalize defects to strings (avoid [object Object] in frontend)
    _norm_def = []
    for d in defects_list_out:
        if isinstance(d, str):
            s = d.strip()
            if s:
                _norm_def.append(s)
        elif isinstance(d, dict):
            note = str(d.get("note") or d.get("text") or d.get("issue") or "").strip()
            if note:
                _norm_def.append(note)
    defects_list_out = list(dict.fromkeys(_norm_def))
# Merge second-pass defect candidates (print lines / scratches / whitening) and glare suspects.
    # - Add high-confidence print_line flags when detected.
    # - If glare suspects exist, annotate rather than over-penalize.
    if isinstance(second_pass, dict) and second_pass.get("ran"):
        cand = second_pass.get("defect_candidates") or []
        glare = second_pass.get("glare_suspects") or []

        # Add candidates into defects list (dedup by (type,note))
        seen = set(str(d).lower().strip() for d in defects_list_out if isinstance(d, str))
        for d in cand:
            if not isinstance(d, dict): 
                continue
            t = str(d.get("type","")).lower().strip()
            note = str(d.get("note","")).strip()
            if not t or not note:
                continue
            key = note.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            defects_list_out.append(note)

            # Promote print line detection into flags if confidence is decent
            try:
                c = float(d.get("confidence", 0))
            except Exception:
                c = 0.0
            if t == "print_line" and c >= 0.55 and "print_line" not in flags_list_out:
                flags_list_out.append("print_line")

        # Store glare suspects into data for frontend / summary
        data["glare_suspects"] = glare[:10]

    has_structural_damage = any(
        f in ("crease", "tear", "paper break", "structural bend", "hole")
        for f in flags_list_out
    )



    raw_pregrade = str(data.get("pregrade", "")).strip()
    g = _grade_bucket(raw_pregrade)
    # NOTE: We intentionally do NOT cap the AI-assessed pregrade here.
    # Any condition-based value adjustments are applied in /api/market-context only.

    pregrade_norm = str(g) if g is not None else ""


    # Ensure assessment_summary is detailed enough (UI-friendly)
    summary = _norm_ws(str(data.get("assessment_summary", "")))
    if len(summary.split()) < 35:
        # Build a fuller summary from structured fields (without inventing defects)
        flags_list = data.get("flags", []) if isinstance(data.get("flags", []), list) else []
        defects_list = data.get("defects", []) if isinstance(data.get("defects", []), list) else []
        cen = data.get("centering", {}) if isinstance(data.get("centering", {}), dict) else {}
        cen_f = (cen.get("front") or {}) if isinstance(cen.get("front") or {}, dict) else {}
        cen_b = (cen.get("back") or {}) if isinstance(cen.get("back") or {}, dict) else {}
        edges = data.get("edges", {}) if isinstance(data.get("edges", {}), dict) else {}
        surf = data.get("surface", {}) if isinstance(data.get("surface", {}), dict) else {}
        ef = (edges.get("front") or {}) if isinstance(edges.get("front") or {}, dict) else {}
        eb = (edges.get("back") or {}) if isinstance(edges.get("back") or {}, dict) else {}
        sf = (surf.get("front") or {}) if isinstance(surf.get("front") or {}, dict) else {}
        sb = (surf.get("back") or {}) if isinstance(surf.get("back") or {}, dict) else {}

        parts = []
        parts.append(f"Overall, this looks like a PSA-style {pregrade_norm or raw_pregrade or 'N/A'} estimate based on what is visible in the photos.")
        if cen_f.get("grade") or cen_b.get("grade"):
            parts.append(f"Centering appears around Front {cen_f.get('grade','').strip() or 'N/A'} and Back {cen_b.get('grade','').strip() or 'N/A'}.")
        if ef.get("grade") or eb.get("grade"):
            parts.append(f"Edges read as Front {ef.get('grade','').strip() or 'N/A'} / Back {eb.get('grade','').strip() or 'N/A'}; notes: {_norm_ws(str(ef.get('notes','')))} {_norm_ws(str(eb.get('notes','')))}".strip())
        if sf.get("grade") or sb.get("grade"):
            parts.append(f"Surface reads as Front {sf.get('grade','').strip() or 'N/A'} / Back {sb.get('grade','').strip() or 'N/A'}; notes: {_norm_ws(str(sf.get('notes','')))} {_norm_ws(str(sb.get('notes','')))}".strip())
        if defects_list:
            parts.append("Visible issues noted: " + "; ".join([_norm_ws(str(d)) for d in defects_list[:8]]) + ("" if len(defects_list) <= 8 else " (and more)."))
        if flags_list:
            parts.append("Key flags: " + ", ".join([_norm_ws(str(f)) for f in flags_list[:10]]) + ("" if len(flags_list) <= 10 else ", …") + ".")
        parts.append("Biggest grade limiters are the most severe corner/edge whitening/chipping, any surface scratches/print lines, and any bends/creases/dents if present.")
        summary = " ".join([p for p in parts if p]).strip()

        data["assessment_summary"] = summary
    return JSONResponse(content={
        "pregrade": pregrade_norm or "N/A",
        "confidence": _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0),
        "centering": data.get("centering", {"front": {"grade": "", "notes": ""}, "back": {"grade": "", "notes": ""}}),
        "corners": data.get("corners", {"front": {}, "back": {}}),
        "edges": data.get("edges", {"front": {"grade": "", "notes": ""}, "back": {"grade": "", "notes": ""}}),
        "surface": data.get("surface", {"front": {"grade": "", "notes": ""}, "back": {"grade": "", "notes": ""}}),
        "defects": defects_list_out,
        "flags": flags_list_out,
        "second_pass": second_pass,
        "glare_suspects": data.get("glare_suspects", []) if isinstance(data.get("glare_suspects", []), list) else [],
        "assessment_summary": _norm_ws(str(data.get("assessment_summary", ""))) or summary or "",
        "spoken_word": _norm_ws(str(data.get("spoken_word", ""))) or _norm_ws(str(data.get("assessment_summary", ""))) or summary or "",
        "observed_id": data.get("observed_id", {}) if isinstance(data.get("observed_id", {}), dict) else {},
        "verify_token": f"vfy_{secrets.token_urlsafe(12)}",
        "market_context_mode": "click_only",
        "has_structural_damage": bool(has_structural_damage),
    })

# ==============================
# Memorabilia / Sealed: Identify + Assess (NO pricing here)
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
    if not b1 or len(b1) < 200:
        raise HTTPException(status_code=400, detail="Image 1 is too small or empty")

    imgs = [b1]
    for f in (image2, image3, image4):
        if f:
            bb = await f.read()
            if bb and len(bb) >= 200:
                imgs.append(bb)

    prompt = """You are identifying a collectible item (sealed product or memorabilia) from photos.

Goal: be SPECIFIC. Name the exact product when possible, including brand/TCG, series/set, configuration, and any visible edition or language.

Return ONLY valid JSON with these exact fields:

{
  "item_type": "sealed booster box/sealed booster bundle/sealed pack/sealed tin/sealed case/elite trainer box/collection box/autographed memorabilia/game-used memorabilia/graded item/other",
  "brand": "brand/league/publisher if visible (e.g., Pokemon TCG, Panini, Topps, Upper Deck, Wizards of the Coast) else empty string",
  "product_name": "exact product name if visible (e.g., 'Scarlet & Violet—151 Booster Bundle') else best-effort specific name",
  "set_or_series": "set/series/expansion name if visible (e.g., 'Scarlet & Violet—151') else empty string",
  "year": "4 digit year if visible else empty string",
  "edition_or_language": "e.g., English/Japanese/1st Edition/Unlimited/Collector's Edition if visible else empty string",
  "special_attributes": ["short tags like Factory Sealed", "Pokemon Center", "Hobby Box", "1st Edition", "Case Fresh"],
  "description": "one clear paragraph describing exactly what it is and what can be seen (packaging, labels, markings)",
  "signatures": "names of any visible signatures or 'None visible'",
  "seal_condition": "Factory Sealed/Opened/Resealed/Damaged/Not applicable",
  "authenticity_notes": "authenticity indicators visible (holograms, stickers, COA) and any red flags",
  "notable_features": "unique features worth noting (promo contents, special print, chase set, serial numbering, COA, etc.)",
  "confidence": 0.0-1.0,
  "category_hint": "Pokemon/Magic/YuGiOh/Sports/OnePiece/Other"
}

Rules:
- If multiple products are plausible, choose the best match and explain uncertainty in authenticity_notes (briefly).
- Do NOT invent a year/edition/language if you cannot see it.
- If it appears sealed, describe the wrap (tight/loose, tears, holes, seams, bubbling). Use 'Factory Sealed' only if it looks consistent.
- If you cannot identify confidently, keep product_name generic and set confidence low.
Respond ONLY with JSON, no extra text.
"""

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for i, bb in enumerate(imgs):
        if i > 0:
            content.append({"type": "text", "text": f"IMAGE {i} ABOVE ☝️ | IMAGE {i+1} BELOW 👇"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(bb)}", "detail": "high"}})

    msg = [{"role": "user", "content": content}]
    result = await _openai_chat(msg, max_tokens=900, temperature=0.1)
    data = _parse_json_or_none(result.get("content", "")) if not result.get("error") else None
    data = data or {}

    return JSONResponse(content={
        "item_type": _norm_ws(str(data.get("item_type", "Unknown"))),
        "brand": _norm_ws(str(data.get("brand", ""))),
        "product_name": _norm_ws(str(data.get("product_name", ""))),
        "set_or_series": _norm_ws(str(data.get("set_or_series", ""))),
        "year": _norm_ws(str(data.get("year", ""))),
        "edition_or_language": _norm_ws(str(data.get("edition_or_language", ""))),
        "special_attributes": data.get("special_attributes", []) if isinstance(data.get("special_attributes", []), list) else [],
        "description": _norm_ws(str(data.get("description", ""))),
        "signatures": _norm_ws(str(data.get("signatures", "None visible"))),
        "seal_condition": _norm_ws(str(data.get("seal_condition", "Not applicable"))),
        "authenticity_notes": _norm_ws(str(data.get("authenticity_notes", ""))),
        "notable_features": _norm_ws(str(data.get("notable_features", ""))),
        "confidence": _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0),
        "category_hint": _norm_ws(str(data.get("category_hint", "Other"))),
        "identify_token": f"idt_{secrets.token_urlsafe(12)}",
    })

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
    if not b1 or not b2 or len(b1) < 200 or len(b2) < 200:
        raise HTTPException(status_code=400, detail="At least two images required")

    imgs = [b1, b2]
    for f in (image3, image4):
        if f:
            bb = await f.read()
            if bb and len(bb) >= 200:
                imgs.append(bb)

    ctx = ""
    if item_type or description:
        ctx = "\n\nKNOWN ITEM DETAILS:\n"
        if item_type: ctx += f"- Item Type: {_norm_ws(item_type)}\n"
        if description: ctx += f"- Description: {_norm_ws(description)}\n"

    prompt = f"""You are a professional memorabilia/collectibles grader.

You MUST identify what the item is (brand + product name + series/set) as specifically as the images allow, then grade condition conservatively.

Return ONLY valid JSON with this EXACT structure:

{{
  "condition_grade": "Mint/Near Mint/Excellent/Good/Fair/Poor",
  "confidence": 0.0-1.0,
  "condition_distribution": {{
    "Mint": 0.0-1.0,
    "Near Mint": 0.0-1.0,
    "Excellent": 0.0-1.0,
    "Good": 0.0-1.0,
    "Fair": 0.0-1.0,
    "Poor": 0.0-1.0
  }},
  "seal_integrity": {{
    "status": "Factory Sealed/Opened/Resealed/Compromised/Not Applicable",
    "notes": "detailed notes about seal/wrap (tightness, tears, holes, seams, bubbling). Mention exact locations."
  }},
  "packaging_condition": {{
    "grade": "Mint/Near Mint/Excellent/Good/Fair/Poor",
    "notes": "detailed notes about packaging wear: corners, dents, crushing, scratches, scuffs, sticker residue, window plastic, edges. Mention exact locations."
  }},
  "signature_assessment": {{
    "present": true/false,
    "quality": "Clear/Faded/Smudged/Not Applicable",
    "notes": "notes about signature placement/ink flow/bleeding and any authenticity concerns"
  }},
  "value_factors": ["short bullets: print run, desirability, era, sealed premium, athlete/popularity, scarcity, set demand"],
  "defects": ["each defect as a full sentence with location + severity"],
  "flags": ["short flags for important issues (reseal risk, crush damage, water, heavy dents, COA missing)"],
  "overall_assessment": "5-8 sentences in first person (start with: 'Looking at your [brand] [product_name]...'). Mention what it is, what looks strong, what issues you see, and what limits the condition grade.",
  "spoken_word": "A 20–45 second spoken-word script in first person. Format it like: Hook (1 line) → What it is (1–2 lines) → Best features (1–2 lines) → Biggest concerns (1–3 lines) → Bottom line grade + confidence (1 line). No hype, no guarantees.",
  "observed_id": {{
    "brand": "best-effort",
    "product_name": "best-effort",
    "set_or_series": "best-effort",
    "year": "best-effort",
    "edition_or_language": "best-effort",
    "item_type": "best-effort"
  }}
}}

{ctx}

Rules:
- Do NOT claim Factory Sealed unless the wrap/seal looks consistent. If uncertain, say so and reduce confidence.
- If glare/blur prevents certainty, say so and reduce confidence.
- Be specific with locations and avoid generic statements.
Respond ONLY with JSON, no extra text.
"""

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for i, bb in enumerate(imgs):
        if i > 0:
            content.append({"type": "text", "text": f"IMAGE {i} ABOVE ☝️ | IMAGE {i+1} BELOW 👇"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(bb)}", "detail": "high"}})

    msg = [{"role": "user", "content": content}]
    result = await _openai_chat(msg, max_tokens=2000, temperature=0.1)
    data = _parse_json_or_none(result.get("content", "")) if not result.get("error") else None
    data = data or {}

    flags_raw = data.get("flags", [])
    if isinstance(flags_raw, list):
        flags_list_out = [str(f).lower().strip() for f in flags_raw if str(f).strip()]
    else:
        flags_list_out = []
    flags_list_out = list(dict.fromkeys(flags_list_out))


    # Ensure condition_distribution exists (confidence-weighted)
    if not isinstance(data.get("condition_distribution"), dict):
        cg = _norm_ws(str(data.get("condition_grade", ""))).title()
        confv = _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0)
        ladder = ["Mint", "Near Mint", "Excellent", "Good", "Fair", "Poor"]
        try:
            i = ladder.index(cg) if cg in ladder else 2
        except Exception:
            i = 2
        # Put most weight on predicted bucket, spill to neighbors based on confidence
        p_main = 0.45 + 0.50 * confv
        rem = 1.0 - p_main
        dist = {k: 0.0 for k in ladder}
        dist[ladder[i]] = p_main
        if i-1 >= 0: dist[ladder[i-1]] += rem * 0.55
        if i+1 < len(ladder): dist[ladder[i+1]] += rem * 0.45
        total = sum(dist.values()) or 1.0
        data["condition_distribution"] = {k: round(v/total, 4) for k, v in dist.items()}


    # Ensure seal_integrity always has status (fixes UI "undefined" cases)
    seal = data.get("seal_integrity") if isinstance(data.get("seal_integrity"), dict) else {}
    if not seal.get("status"):
        seal["status"] = "Not Applicable"
    if not seal.get("notes"):
        seal["notes"] = ""
    if not seal.get("grade"):
        # Frontend may expect "grade" - mirror status for compatibility
        seal["grade"] = seal.get("status", "Not Applicable")

    return JSONResponse(content={
        "condition_grade": _norm_ws(str(data.get("condition_grade", "N/A"))),
        "confidence": _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0),
        "seal_integrity": seal,
        "packaging_condition": data.get("packaging_condition", {"grade": "N/A", "notes": ""}),
        "signature_assessment": data.get("signature_assessment", {"present": False, "quality": "Not Applicable", "notes": ""}),
        "value_factors": data.get("value_factors", []) if isinstance(data.get("value_factors", []), list) else [],
        "defects": data.get("defects", []) if isinstance(data.get("defects", []), list) else [],
        "overall_assessment": _norm_ws(str(data.get("overall_assessment", ""))),
        "spoken_word": _norm_ws(str(data.get("spoken_word", ""))) or _norm_ws(str(data.get("overall_assessment", ""))),
        "observed_id": data.get("observed_id", {}) if isinstance(data.get("observed_id", {}), dict) else {},
        "flags": flags_list_out,
        "verify_token": f"vfy_{secrets.token_urlsafe(12)}",
    })

# ==============================
# Market Context (Click-only) - Cards
# Primary: PriceCharting (current prices, includes graded where available)
# Secondary: PokemonTCG links (ID + marketplace links, not history)
# eBay API: scaffold only (disabled)
# ==============================

@app.post("/api/market-context")
@safe_endpoint
async def market_context(
    # Preferred keys (new)
    item_name: Optional[str] = Form(None),
    item_category: Optional[str] = Form("Pokemon"),  # Pokemon/Magic/YuGiOh/Sports/OnePiece/Other
    item_set: Optional[str] = Form(None),
    item_number: Optional[str] = Form(None),
    product_id: Optional[str] = Form(None),
    predicted_grade: Optional[str] = Form("9"),
    confidence: float = Form(0.0),
    grading_cost: float = Form(35.0),
    has_structural_damage: Optional[bool] = Form(False),  # new: hard gate for damaged cards

    item_type: Optional[str] = Form(None),
    assessed_pregrade: Optional[str] = Form(None),
    condition_multiplier: Optional[float] = Form(None),

    # Back-compat aliases from older frontends
    card_name: Optional[str] = Form(None),
    card_type: Optional[str] = Form(None),
    card_set: Optional[str] = Form(None),
    card_number: Optional[str] = Form(None),

    # Extra loose aliases (some JS sends these)
    name: Optional[str] = Form(None),
    query: Optional[str] = Form(None),
):
    """Click-only informational market context.

    Primary source: PriceCharting *custom CSV download* (your account token), cached for speed.
    Output shape is kept compatible with your current frontend renderer.
    """

    # --------------------------
    # Helpers (local to endpoint)
    # --------------------------
    async def _http_get_text(url: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(url, headers={"User-Agent": UA})
                if r.status_code != 200:
                    return ""
                return r.text or ""
        except Exception:
            return ""

    def _parse_money(v: Any) -> Optional[float]:
        if v is None:
            return None
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "null", "-"):
            return None
        s = s.replace("$", "").replace(",", "").strip()
        try:
            return float(s)
        except Exception:
            return None

    def _pc_csv_cache_path(category_slug: str) -> str:
        safe = re.sub(r"[^a-z0-9\-]+", "-", (category_slug or "pokemon-cards").lower()).strip("-")
        return os.path.join(PRICECHARTING_CACHE_DIR, f"pc_csv_{safe}_latest.csv")

    def _pc_norm(s: str) -> str:
        s = _norm_ws(str(s or "")).lower()
        s = s.replace("-", "-").replace("-", "-")
        s = re.sub(r"[^a-z0-9#\-/ ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _extract_card_no_short(card_number_display: str) -> Optional[int]:
        s = _pc_norm(card_number_display)
        if not s:
            return None
        if "/" in s:
            s = s.split("/", 1)[0]
        s = s.replace("#", "").strip()
        try:
            return int(s)
        except Exception:
            return None

    def _liquidity_label(sales_volume: Any) -> str:
        try:
            sv = int(str(sales_volume).strip() or "0")
        except Exception:
            sv = 0
        if sv >= 100:
            return "high"
        if sv >= 25:
            return "medium"
        if sv > 0:
            return "low"
        return "-"


    
    def _condition_multiplier_from_pregrade(g: Optional[int]) -> float:
        """Conservative 'as-is' multiplier from observed/assessed pregrade.

        IMPORTANT: This is intentionally harsh for low grades. A PSA 1–4 copy is often worth a small
        fraction of a near-mint reference sale.
        """
        if g is None:
            return 0.80
        try:
            gi = int(g)
        except Exception:
            return 0.80
        gi = max(1, min(10, gi))
        if gi >= 10: return 1.00
        if gi == 9:  return 0.90
        if gi == 8:  return 0.78
        if gi == 7:  return 0.65
        if gi == 6:  return 0.52
        if gi == 5:  return 0.35
        if gi == 4:  return 0.18
        if gi == 3:  return 0.10
        if gi == 2:  return 0.06
        return 0.03

    def _safe_money_mul(v: Optional[float], m: Optional[float]) -> Optional[float]:
        try:
            if v is None or m is None:
                return None
            return round(float(v) * float(m), 2)
        except Exception:
            return None
        try:
            gi = int(g)
        except Exception:
            return "unknown"
        if gi >= 9:
            return "mint_like"
        if gi >= 7:
            return "near_mint"
        if gi >= 5:
            return "excellent"
        if gi >= 3:
            return "good"
        return "poor"

    def _safe_money_mul(v: Optional[float], mult: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        try:
            m = float(mult) if mult is not None else 1.0
        except Exception:
            m = 1.0
        return round(float(v) * m, 2)



    def _holding_time(liquidity: str, trend: str) -> Tuple[str, str]:
        """Return (label, explanation). Labels: short/medium/long."""
        liq = (liquidity or "-").lower()
        tr = (trend or "-").lower()
        if liq == "high" and tr == "increasing":
            return ("short", "Higher activity and upward momentum can shorten the wait for a stronger sale price.")
        if liq in ("high", "medium") and tr in ("increasing", "stable"):
            return ("medium", "Reasonable activity; meaningful movement is more likely over months than weeks.")
        return ("long", "Lower activity and/or weaker momentum suggests a longer collector-style hold for meaningful movement.")

    def _grading_value_summary(
        card_label: str,
        predicted_grade_str: str,
        conf_in: float,
        assessed_grade_int: Optional[int],
        has_structural_damage: bool,
        grading_cost_aud: float,
        raw_as_is_aud: Optional[float],
        expected_graded_aud: Optional[float],
        liquidity: str,
        trend: str,
    ) -> Dict[str, Any]:
        """Create a spoken-word grading/value summary (non-promissory)."""
        # Parse grade
        try:
            g = float(str(predicted_grade_str or "").strip())
        except Exception:
            g = 0.0
        c = _clamp(_safe_float(conf_in, 0.0), 0.0, 1.0)

        # Hard gates: structural damage or very low assessed grade => do NOT recommend grading for value
        if has_structural_damage:
            return {
                "spoken_word": (
                    f"Because {card_label or 'this card'} shows structural damage like creasing or tearing, "
                    "high-grade market values do not apply to this specific copy. Grading for value is not recommended. "
                    "Treat market history below as general reference only for the title, not this condition."
                ),
                "recommendation": "do_not_grade",
                "grading_cost_aud": float(grading_cost_aud or 0.0),
                "suggested_holding_time": "long",
            }

        if isinstance(assessed_grade_int, int) and assessed_grade_int <= 4:
            # Low-grade condition: cap expectations and avoid high-grade anchors
            return {
                "spoken_word": (
                    f"Based on the visible condition, {card_label or 'this card'} looks like a low-grade copy. "
                    "In this range, grading rarely increases value unless the card is exceptionally rare, and it can reduce liquidity. "
                    "Market history for high grades is not applicable to this condition. If you grade, do it for protection or authenticity, not value."
                ),
                "recommendation": "do_not_grade",
                "grading_cost_aud": float(grading_cost_aud or 0.0),
                "suggested_holding_time": "long",
            }


        # Determine recommendation
        recommendation = "borderline"
        if g >= 9.0 and c >= 0.75:
            recommendation = "grade"
        elif g <= 7.5 and c >= 0.40:
            recommendation = "do_not_grade"

        # Compute net view (best-effort)
        net = None
        if isinstance(expected_graded_aud, (int, float)) and isinstance(raw_as_is_aud, (int, float)):
            net = round(float(expected_graded_aud) - float(raw_as_is_aud) - float(grading_cost_aud or 0.0), 2)

        hold_label, hold_reason = _holding_time(liquidity, trend)

        # Build spoken word (avoid guarantees / ROI language)
        parts = []
        parts.append(f"Here’s our take on grading {card_label or 'this card'} based on the current market history.")
        if g:
            parts.append(f"With a projected grade around {g:.1f} and confidence at {int(round(c*100))} percent, grading is {('generally worth considering' if recommendation=='grade' else 'more of a borderline call' if recommendation=='borderline' else 'probably not worth it for value')}.")

        if isinstance(raw_as_is_aud, (int, float)):
            parts.append(f"Using the visible condition as-is, the raw baseline is roughly ${raw_as_is_aud:,.2f} AUD.")
        if isinstance(expected_graded_aud, (int, float)):
            parts.append(f"A reasonable graded estimate comes out around ${expected_graded_aud:,.2f} AUD.")
        if net is not None:
            if net > 0:
                parts.append(f"After the ${grading_cost_aud:,.0f} AUD grading fee, there appears to be some headroom - but it’s not guaranteed.")
            else:
                parts.append(f"Once you include the ${grading_cost_aud:,.0f} AUD grading fee, the numbers look tight, so the value case for grading isn’t strong right now.")

        parts.append(f"Market activity looks {liquidity or '-'} and the trend reads {trend or '-'}. {hold_reason}")
        if hold_label == "short":
            parts.append("If you grade, think in terms of a shorter hold - around 3 to 12 months - rather than expecting an immediate jump.")
        elif hold_label == "medium":
            parts.append("If you grade, a medium hold - roughly 6 to 18 months - is a more realistic window to see meaningful movement.")
        else:
            parts.append("If you grade, treat it as a longer hold - 12 months plus - where the benefit is credibility and protection, not a quick outcome.")

        parts.append("This is informational only and markets can change quickly - use it as a guide, not a promise.")

        return {
            "spoken_word": " ".join(parts),
            "recommendation": recommendation,
            "grading_cost_aud": float(grading_cost_aud or 0.0),
            "suggested_holding_time": hold_label,
        }

    # --------------------------
    # Normalize inputs
    # --------------------------
    name_in = item_name or card_name or name or query or ""
    cat_in = item_category or card_type or "Other"
    set_in = item_set or card_set or ""
    num_in = item_number or card_number or ""
    pid_in = _norm_ws(str(product_id or ""))

    clean_name = _norm_ws(str(name_in))
    clean_cat = _normalize_card_type(_norm_ws(str(cat_in)))
    clean_set = _norm_ws(str(set_in))
    clean_num_display = _clean_card_number_display(str(num_in))


    # -----------------------------------------
    # eBay SOLD listings as PRIMARY pricing data
    # -----------------------------------------
    # We build a conservative "low/avg/high" from sold comps (p20/median/p80).
    # This is informational context only.
    base_query = " ".join([clean_name, clean_set, clean_num_display]).strip()
    if not base_query:
        base_query = " ".join([clean_name, clean_set]).strip() or clean_name

    # Fetch RAW sold stats first (always)
    raw_stats = _ebay_completed_stats(base_query, limit=120)
    # Determine an assessed grade bucket to pull graded comps around the *observed* condition
    g_ass = _grade_bucket(assessed_pregrade or "") or _grade_bucket(predicted_grade or "")
    g_ass = int(g_ass or 0) if str(g_ass).isdigit() else int(g_ass or 0)
    if g_ass <= 0:
        g_ass = 0

    graded_stats: Dict[str, Any] = {}
    # Only pull a small set of graded buckets to keep API calls sane
    grades_to_pull: List[int] = []
    if clean_cat.lower() in ("pokemon", "magic", "yugioh", "sports", "onepiece", "other"):
        if g_ass >= 8:
            grades_to_pull = [10, 9, 8]
        elif g_ass >= 5:
            grades_to_pull = [g_ass + 1, g_ass, 8]
        elif g_ass >= 1:
            grades_to_pull = [g_ass + 1, g_ass]
        else:
            grades_to_pull = [10, 9, 8]

        # de-dup + clamp
        grades_to_pull = sorted({max(1, min(10, int(x))) for x in grades_to_pull}, reverse=True)

        for g in grades_to_pull:
            qg = f"{base_query} PSA {g}"
            st = _ebay_completed_stats(qg, limit=90)
            if st:
                graded_stats[str(g)] = st

    # Convert to AUD (if needed) and compute a compact observed object
    def _to_aud_stats(st: dict) -> dict:
        if not st:
            return {"median": None, "avg": None, "low": None, "high": None, "count": 0, "currency": "AUD"}
        cur = str(st.get("currency") or "USD").upper()
        fx = 1.0
        if cur == "USD":
            fx = _usd_to_aud_rate()
        elif cur == "AUD":
            fx = 1.0
        # If other currency, treat as USD for now (rare)
        def conv(v):
            try:
                return round(float(v) * fx, 2) if v is not None else None
            except Exception:
                return None
        return {
            "median": conv(st.get("median")),
            "avg": conv(st.get("avg")),
            "low": conv(st.get("low")),
            "high": conv(st.get("high")),
            "count": int(st.get("count") or 0),
            "currency": "AUD",
        }

    raw_aud = _to_aud_stats(raw_stats)
    graded_aud = {k: _to_aud_stats(v) for k, v in (graded_stats or {}).items()}

    # If we have enough sold comps, return eBay-driven context immediately.
    # (PriceCharting remains as fallback if eBay has no data.)
    if raw_aud.get("count", 0) >= 6:
        # Damage lock: if verify flagged structural damage, kill EV/high-grade language
        damage_flag = bool(has_structural_damage or (str(assessed_pregrade or "").strip() in ("1", "2", "3") ))
        if damage_flag:
            return JSONResponse(content={
                "available": True,
                "mode": "damage_locked",
                "message": "This copy appears to have structural damage / heavy wear. High-grade values do not apply.",
                "used_query": base_query,
                "card": {"name": clean_name, "set": clean_set, "set_code": "", "year": "", "card_number": clean_num_display, "type": clean_cat},
                "observed": {
                    "currency": "AUD",
                    "liquidity": "-",
                    "trend": "-",
                    "raw": raw_aud,
                    "graded_psa": graded_aud,
                },
                "grade_impact": {
                    "expected_graded_value": None,
                    "raw_baseline_value": raw_aud.get("median"),
                    "grading_cost": float(grading_cost or 0.0),
                    "estimated_value_difference": None,
                },
                "grading_value_summary": {
                    "spoken_word": (
                        "Because this card shows heavy wear or possible structural damage, grading for value usually doesn't make sense. "
                        "The market prices for high grades don't apply to this specific copy. Treat these numbers as general reference only."
                    ),
                    "recommendation": "do_not_grade",
                },
                "meta": {"pricing_source": "ebay"},
                "disclaimer": "Informational market context only. Sold listings vary by platform, timing, and condition.",
            }, status_code=200)

        # Build a grade-aware summary using the observed grade (NOT a default 9)
        g_for_summary = g_ass if g_ass else None
        conf_in = _clamp(_safe_float(confidence, 0.0), 0.0, 1.0)

        # Choose a graded reference price closest to the assessed grade if we have it
        graded_ref = None
        if g_for_summary and str(g_for_summary) in graded_aud:
            graded_ref = graded_aud[str(g_for_summary)].get("median")
        elif graded_aud:
            # pick the lowest grade we fetched (closest to the assessed if below 8)
            pick = sorted([int(k) for k in graded_aud.keys()])[-1]
            graded_ref = graded_aud[str(pick)].get("median")

        raw_med = raw_aud.get("median")
        grading_fee = float(grading_cost or 0.0)
        expected = graded_ref
        diff = (expected - raw_med - grading_fee) if (expected and raw_med) else None

        spoken = (
            f"Here's our take based on recent sold listings. Your observed condition looks around a {g_for_summary or 'N/A'} "
            f"with confidence at {int(conf_in*100)} percent. "
        )
        if expected and raw_med:
            spoken += (
                f"Using that condition as-is, the raw baseline is roughly {raw_med:.2f} AUD. "
                f"A comparable graded copy around that level has a median near {expected:.2f} AUD. "
                f"After the {grading_fee:.0f} AUD fee, the headroom is {diff:.2f} AUD — but it's not guaranteed."
            )
        else:
            spoken += "We couldn't form a reliable graded comparison for this condition level yet. Use the raw sold range as your best guide for now."

        return JSONResponse(content={
            "available": True,
            "mode": "click_only",
            "message": "Market History Loaded",
            "used_query": base_query,
            "confidence": conf_in,
            "card": {"name": clean_name, "set": clean_set, "set_code": "", "year": "", "card_number": clean_num_display, "type": clean_cat},
            "observed": {
                "currency": "AUD",
                "liquidity": "-",
                "trend": "-",
                "raw": raw_aud,
                "graded_psa": graded_aud,
            },
            "grading_value_summary": {"spoken_word": spoken, "recommendation": "grade" if (g_for_summary and g_for_summary >= 8) else "consider"},
            "grade_impact": {
                "expected_graded_value": expected,
                "raw_baseline_value": raw_med,
                "grading_cost": grading_fee,
                "estimated_value_difference": diff,
            },
            "meta": {"pricing_source": "ebay", "counts": {"raw": raw_aud.get("count"), "graded": {k: v.get("count") for k, v in graded_aud.items()}}},
            "disclaimer": "Informational market context only. Sold listings vary by platform, timing, and condition.",
        }, status_code=200)
    damage_flag = str(has_structural_damage).strip().lower() in ("1","true","yes","y","on")



    
    # --------------------------
    # Hard gate: structural damage lock
    # --------------------------
    try:
        g_ass = _grade_bucket(str(assessed_pregrade or predicted_grade or "").strip())
    except Exception:
        g_ass = None

    if damage_flag:
        # We still return market history, but we DO NOT compute high-grade projections or ROI-style deltas.
        # Any high-grade values are not applicable to this specific damaged copy.
        # We'll continue to load market history later in the function and then override the grading summary.
        pass

# --------------------------
    # Sealed / memorabilia quick-path (PriceCharting API search)
    # --------------------------
    # Your CSV download is strongest for singles. Sealed/memorabilia often matches better via API search.
    sealed_hint = _norm_ws(str(item_type or "")).lower()
    sealed_keywords = ["booster box","booster bundle","bundle","display","sealed","tin","etb","elite trainer","case","box","pack","collection box","premium collection"]
    hay = (" ".join([clean_name, clean_set, sealed_hint])).lower()
    is_sealed_like = ("sealed" in sealed_hint) or ("memorabilia" in sealed_hint) or any(k in hay for k in sealed_keywords)

    if is_sealed_like and PRICECHARTING_TOKEN:
        try:
            q_api = " ".join([clean_name, clean_set]).strip() or clean_name
            products = await _pc_search(q_api, category=None, limit=8)
            if products:
                top = products[0]
                pid = str(top.get("id") or top.get("product-id") or "").strip()
                detail = await _pc_product(pid) if pid else {}
                prices = _pc_extract_price_fields(detail or top)
                vals = [v for v in prices.values() if isinstance(v, (int, float)) and v > 0]
                typical_usd = round(sum(vals)/len(vals), 2) if vals else None
                typical_aud = _usd_to_aud_simple(typical_usd) if typical_usd is not None else None

                # Apply condition multiplier ONLY to recommended buy guidance (not the assessment itself)
                g_ass = _grade_bucket(assessed_pregrade or "") or _grade_bucket(predicted_grade or "")
                mult = float(condition_multiplier) if isinstance(condition_multiplier, (int, float)) else _condition_multiplier_from_pregrade(g_ass)
                bucket = _condition_bucket_from_pregrade(g_ass)

                # Recommended buy (very simple for sealed): apply multiplier to typical
                rec_buy_usd = _safe_money_mul(typical_usd, mult) if typical_usd is not None else None
                rec_buy_aud = _usd_to_aud_simple(rec_buy_usd) if rec_buy_usd is not None else None

                return JSONResponse(content={
                    "available": True,
                    "mode": "click_only",
                    "message": "Market history loaded (sealed/memorabilia match)",
                    "used_query": q_api,
                    "query_ladder": [q_api],
                    "confidence": _clamp(_safe_float(confidence, 0.0), 0.0, 1.0),
                    "card": {"name": clean_name, "set": clean_set, "set_code": "", "year": "", "card_number": clean_num_display, "type": clean_cat},
                    "observed": {
                        "currency": "AUD",
                        "fx": {"usd_to_aud_multiplier": AUD_MULTIPLIER},
                        "usd_original": {"typical": typical_usd},
                        "raw": {"median": typical_aud, "avg": typical_aud, "range": None},
                        "graded_psa": {"10": None, "9": None, "8": None},
                        "as_is": {"multiplier": mult, "bucket": bucket, "recommended_buy_aud": rec_buy_aud},
                        "pricecharting_prices": prices,
                    },
                    "grade_impact": {
                        "expected_graded_value": None,
                        "expected_graded_value_aud": None,
                        "raw_baseline_value": typical_usd,
                        "grading_cost": float(grading_cost or 0.0),
                        "estimated_value_difference": None,
                        "recommended_buy": {
                            "currency": "AUD",
                            "recommended_purchase_price": rec_buy_usd,
                            "recommended_purchase_price_aud": rec_buy_aud,                            "notes": "Sealed/memorabilia guidance uses PriceCharting API typical price and applies the condition multiplier only for the recommended buy value.",
                        },
                    },
                    "meta": {
            "damage_locked": bool(damage_flag),
                        "match": {"id": pid, "product_name": (detail or top).get("product-name") or top.get("product-name"), "console_name": (detail or top).get("console-name") or top.get("console-name")},
                        "sources": ["pricecharting_api"],
                    },
                    "grading_value_summary": {
                        "spoken_word": "For sealed and memorabilia items, grading isn’t usually the main decision point. The bigger drivers are seal integrity, packaging condition, and overall demand. If you’re holding for a meaningful price move, focus on keeping it protected and expect changes to play out over months rather than days.",
                        "recommendation": "not_applicable",
                        "grading_cost_aud": float(grading_cost or 0.0),
                        "suggested_holding_time": "medium"
                    },
                    "disclaimer": "Informational market history only. Figures are third-party estimates and may vary. Not financial advice.",
                }, status_code=200)
        except Exception:
            pass
    if not clean_name:
        return JSONResponse(content={
            "available": False,
            "mode": "click_only",
            "message": "Missing item name. Please re-run identification or pass item_name (or card_name).",
            "used_query": "",
            "query_ladder": [],
            "confidence": _clamp(_safe_float(confidence, 0.0), 0.0, 1.0),
            "disclaimer": "Informational market context only. Not financial advice."
        }, status_code=200)

    # --------------------------
    # Category -> PriceCharting CSV category slug
    # --------------------------
    cat_map = {
        "Pokemon": "pokemon-cards",
        "Magic": "magic-cards",
        "YuGiOh": "yugioh-cards",
        "OnePiece": "one-piece-",
        "Sports": "sports-cards",
        "Other": "pokemon-cards",  # fallback
    }
    pc_cat = cat_map.get(clean_cat, "pokemon-cards")

    if not PRICECHARTING_TOKEN:
        return JSONResponse(content={
            "available": False,
            "mode": "click_only",
            "message": "PRICECHARTING_TOKEN not configured on backend.",
            "used_query": "",
            "query_ladder": [],
            "confidence": _clamp(_safe_float(confidence, 0.0), 0.0, 1.0),
            "disclaimer": "Informational market context only. Not financial advice."
        }, status_code=200)

    # --------------------------
    # Cached CSV download
    # --------------------------
    cache_path = _pc_csv_cache_path(pc_cat)
    max_age_hours = 24
    try:
        if os.path.exists(cache_path):
            mtime = datetime.utcfromtimestamp(os.path.getmtime(cache_path))
            if (datetime.utcnow() - mtime).total_seconds() < max_age_hours * 3600:
                csv_text = Path(cache_path).read_text(encoding="utf-8", errors="ignore")
            else:
                csv_text = ""
        else:
            csv_text = ""
    except Exception:
        csv_text = ""

    if not csv_text or len(csv_text) < 100:
        url = f"https://www.pricecharting.com/price-guide/download-custom?t={PRICECHARTING_TOKEN}&category={pc_cat}"
        csv_text = await _http_get_text(url)
        if csv_text and len(csv_text) > 100:
            try:
                os.makedirs(PRICECHARTING_CACHE_DIR, exist_ok=True)
                Path(cache_path).write_text(csv_text, encoding="utf-8")
            except Exception:
                pass

    if not csv_text or len(csv_text) < 100:
        return JSONResponse(content={
            "available": False,
            "mode": "click_only",
            "message": "Failed to download PriceCharting CSV (empty response). Check token/category.",
            "used_query": pc_cat,
            "query_ladder": [pc_cat],
            "confidence": _clamp(_safe_float(confidence, 0.0), 0.0, 1.0),
            "disclaimer": "Informational market context only. Not financial advice."
        }, status_code=200)

    # --------------------------
    # Parse + match best row
    # --------------------------
    rows = []
    try:
        reader = csv.DictReader(csv_text.splitlines())
        for r in reader:
            rows.append(r)
    except Exception:
        rows = []

    n_norm = _pc_norm(clean_name)
    s_norm = _pc_norm(clean_set)
    num_short = _extract_card_no_short(clean_num_display)

    name_tokens = [t for t in n_norm.split() if t not in ("pokemon", "card", "cards")]
    set_tokens = [t for t in s_norm.split() if t not in ("pokemon", "card", "cards")]
    set_tokens = set_tokens[:4]

    best = None
    best_score = -1

    for r in rows:
        # Exact match by product id if provided (CSV column "id")
        if pid_in:
            rid = _norm_ws(str(r.get("id", "")))
            if rid and rid == pid_in:
                best = r
                best_score = 999
                break

        pn = _pc_norm(r.get("product-name", ""))
        cn = _pc_norm(r.get("console-name", ""))

        score = 0
        if num_short is not None:
            if f"#{num_short}" in pn:
                score += 12
            elif str(num_short) in pn:
                score += 5

        if is_sealed_like:
            # Sealed/memorabilia rows often swap "set" vs "product" wording between inputs,
            # so we score tokens against BOTH fields.
            for tkn in name_tokens:
                if tkn and (tkn in pn or tkn in cn):
                    score += 3

            for tkn in set_tokens:
                if tkn and (tkn in pn or tkn in cn):
                    score += 2

            if s_norm and (s_norm in cn or s_norm in pn):
                score += 6

            # Extra boost if inputs look swapped (set appears in product-name and name appears in console-name)
            if n_norm and (n_norm in cn) and s_norm and (s_norm in pn):
                score += 6
        else:
            for tkn in name_tokens:
                if tkn and tkn in pn:
                    score += 3

            for tkn in set_tokens:
                if tkn and (tkn in cn or tkn in pn):
                    score += 2

            if s_norm and s_norm in cn:
                score += 6

        if score > best_score:
            best_score = score
            best = r

    min_score = 6 if is_sealed_like else 10

    if not best or best_score < min_score:
        return JSONResponse(content={
            "available": True,
            "mode": "click_only",
            "message": "No strong PriceCharting CSV match found for this item yet.",
            "used_query": f"{clean_name} {clean_set} {clean_num_display}".strip(),
            "query_ladder": [clean_name, f"{clean_name} {clean_set}".strip(), f"{clean_name} {clean_set} {clean_num_display}".strip()],
            "confidence": _clamp(_safe_float(confidence, 0.0), 0.0, 1.0),
            "card": {"name": clean_name, "set": clean_set, "set_code": "", "year": "", "card_number": clean_num_display, "type": clean_cat},
            "observed": {
                "currency": "AUD",
                "liquidity": "-",
                "trend": "-",
                "raw": {"median": None, "avg": None, "range": None},
                "graded_psa": {"10": None, "9": None, "8": None},
            },
            "grade_impact": {
                "expected_graded_value": None,
                "raw_baseline_value": None,
                "grading_cost": float(grading_cost or 0.0),
                "estimated_value_difference": None,
            },
            "meta": {            },
            "disclaimer": "Informational market context only. Figures are third-party estimates and may vary.",
        }, status_code=200)

    # --------------------------
    # Extract prices (CSV format)
    # --------------------------
    loose = _parse_money(best.get("loose-price"))
    newp = _parse_money(best.get("new-price"))
    graded = _parse_money(best.get("graded-price"))

    cond17 = _parse_money(best.get("condition-17-price"))
    cond16 = _parse_money(best.get("condition-16-price"))  # commonly present in your CSV
    cond18 = _parse_money(best.get("condition-18-price"))
    bgs10 = _parse_money(best.get("bgs-10-price"))

    retail_loose_buy = _parse_money(best.get("retail-loose-buy"))
    retail_loose_sell = _parse_money(best.get("retail-loose-sell"))

    # Raw baseline: prefer loose, else new
    raw_val = loose if loose is not None else newp

    # ---- "PSA-style" anchors (equivalency, not literal PSA comps) ----
    # IMPORTANT: We keep the UI labels familiar but we do NOT pretend these are official PSA comps.
    # Priority:
    # - PSA 10 equiv: BGS 10 if present, else condition-18, else graded
    # - PSA 9 equiv: condition-17, else graded
    # - PSA 8 equiv: graded
    psa10_equiv = bgs10 or cond18 or graded or newp or loose
    psa9_equiv = cond17 or graded or newp or loose
    psa8_equiv = cond18 or graded or newp or loose

    # --------------------------
    # Liquidity & trend (if we have a product id + history)
    # --------------------------
    pid = (best.get("id") or "").strip()
    liquidity = _liquidity_label(best.get("sales-volume"))
    trend_label = "-"
    try:
        if pid:
            tr = _pc_trend(pid, days=30)
            if isinstance(tr, dict) and tr.get("available") and tr.get("label"):
                trend_label = tr.get("label")
    except Exception:
        pass

    # --------------------------
    # Expected value (simple)
    # --------------------------
    conf_in = _clamp(_safe_float(confidence, 0.0), 0.0, 1.0)
    g_pred = _grade_bucket(assessed_pregrade) or _grade_bucket(predicted_grade) or 9
    dist = _grade_distribution(g_pred, conf_in)

    price_map = {"10": psa10_equiv, "9": psa9_equiv, "8": psa8_equiv}
    ev = 0.0
    ev_samples = 0
    for k, p in dist.items():
        v = price_map.get(k)
        if isinstance(v, (int, float)) and v > 0:
            ev += float(p) * float(v)
            ev_samples += 1
    expected_val = round(ev, 2) if ev_samples else None

    raw_base = raw_val if isinstance(raw_val, (int, float)) else None
    diff = (round(expected_val - raw_base - float(grading_cost or 0.0), 2) if (expected_val is not None and raw_base is not None) else None)


    # --------------------------
    # Condition-adjusted ("as-is") view
    # --------------------------
    g_ass = _grade_bucket(assessed_pregrade or "") or _grade_bucket(predicted_grade or "")
    try:
        mult = float(condition_multiplier) if condition_multiplier is not None else None
    except Exception:
        mult = None
    if mult is None:
        mult = _condition_multiplier_from_pregrade(g_ass)
    # Clamp to a sane range to avoid UI weirdness if a bad multiplier is posted
    mult = max(0.02, min(1.10, float(mult)))
    bucket = _condition_bucket_from_pregrade(g_ass)
    # --------------------------
    # Recommended purchase price
    # --------------------------
    try:
        pg = float(str(predicted_grade or "").strip() or 0)
    except Exception:
        pg = 0.0

    base_buy = retail_loose_buy
    if base_buy is None and isinstance(raw_base, (int, float)) and raw_base > 0:
        base_buy = round(raw_base * 0.70, 2)  # conservative fallback if retail buy not provided

    # condition factor based on assessed grade (simple, explainable)
    cond_factor = max(0.60, min(1.10, 0.40 + 0.07 * pg)) if pg else 0.85
    conf_factor = max(0.0, min(1.0, float(conf_in or 0.0)))

    assume_grading = False

    rec_buy = None
    note = []

    if base_buy is not None:
        rec_buy = float(base_buy) * float(cond_factor) * float(conf_factor)
        note.append("Baseline uses retail-loose-buy adjusted by assessed grade & confidence.")
    # cap recommended buy to observed raw value if we have one
    if rec_buy is not None and raw_base is not None:
        rec_buy = min(float(rec_buy), float(raw_base))

    
        # --- AUD conversion (simple multiplier) ---
    aud_raw = _usd_to_aud_simple(raw_val) if raw_val is not None else None
    aud_psa10 = _usd_to_aud_simple(psa10_equiv) if psa10_equiv is not None else None
    aud_psa9 = _usd_to_aud_simple(psa9_equiv) if psa9_equiv is not None else None
    aud_psa8 = _usd_to_aud_simple(psa8_equiv) if psa8_equiv is not None else None
    aud_expected = _usd_to_aud_simple(expected_val) if expected_val is not None else None
    aud_rec_buy = _usd_to_aud_simple(rec_buy) if rec_buy is not None else None

    rec_buy = (round(float(rec_buy), 2) if rec_buy is not None else None)

    # --- Grading value summary (spoken word) ---
    card_label = f"your {clean_name}" if clean_name else "this card"

    g_ass = _grade_bucket(assessed_pregrade or "") or _grade_bucket(predicted_grade or "")
    mult = float(condition_multiplier) if isinstance(condition_multiplier, (int, float)) else _condition_multiplier_from_pregrade(g_ass)
    raw_as_is_aud = (_usd_to_aud_simple(raw_val) * float(mult)) if (raw_val is not None and _usd_to_aud_simple(raw_val) is not None) else None

    # If structural damage is present, collapse values aggressively and disable graded EV math.
    if damage_flag:
        if isinstance(raw_as_is_aud, (int, float)):
            raw_as_is_aud = round(float(raw_as_is_aud) * 0.15, 2)  # damaged copies typically trade at a small fraction
        expected_graded_aud = None


    expected_graded_aud = _usd_to_aud_simple(expected_val) if expected_val is not None else None
    grading_value_summary = _grading_value_summary(
        card_label=card_label,
        predicted_grade_str=str(g_pred or assessed_pregrade or predicted_grade or ""),
        conf_in=conf_in,
        assessed_grade_int=g_ass,
        has_structural_damage=bool(damage_flag),
        grading_cost_aud=float(grading_cost or 0.0),
        raw_as_is_aud=raw_as_is_aud,
        expected_graded_aud=expected_graded_aud,
        liquidity=liquidity,
        trend=trend_label,
    )


    mode = "damage_locked" if damage_flag else "click_only"
    disclaimer = "Damaged-condition informational context only. High-grade values do not apply to this copy." if damage_flag else "Informational market context only. Figures are third-party estimates and may vary."

    # UI expects these keys
    return JSONResponse(content={
        "available": True,
        "mode": mode,
        "message": "Market history loaded",
        "used_query": f"{clean_name} {clean_set} {clean_num_display}".strip(),
        "query_ladder": [clean_name, f"{clean_name} {clean_set}".strip(), f"{clean_name} {clean_set} {clean_num_display}".strip()],
        "confidence": conf_in,

        "card": {
            "name": clean_name,
            "set": clean_set,
            "set_code": "",
            "year": "",
            "card_number": clean_num_display,
            "type": clean_cat,
        },

        "observed": {
            "currency": "AUD",
            "fx": {"usd_to_aud_multiplier": AUD_MULTIPLIER},
            "aud": {"raw_median": aud_raw, "psa10": aud_psa10, "psa9": aud_psa9, "psa8": aud_psa8},
            "liquidity": liquidity,
            "trend": trend_label,
            "raw": {
                "median": raw_val,
                "avg": raw_val,
                "range": None,
            },
            "graded_psa": {
                "10": psa10_equiv,
                "9": psa9_equiv,
                "8": psa8_equiv,
            },
            "as_is": {
                "multiplier": mult,
                "bucket": bucket,
                "raw_as_is_aud": _safe_money_mul(aud_raw, mult),
                "psa10_equiv_as_is_aud": _safe_money_mul(aud_psa10, mult),
                "psa9_equiv_as_is_aud": _safe_money_mul(aud_psa9, mult),
                "psa8_equiv_as_is_aud": _safe_money_mul(aud_psa8, mult),
                "recommended_buy_aud": aud_rec_buy,
            },
        },

        "grade_impact": {
            "expected_graded_value": expected_val,
            "expected_graded_value_aud": aud_expected,
            "raw_baseline_value": raw_base,
            "grading_cost": float(grading_cost or 0.0),
            "estimated_value_difference": diff,
            "recommended_buy": {
                "currency": "AUD",
                "retail_loose_buy": retail_loose_buy,
                "retail_loose_sell": retail_loose_sell,
                "recommended_purchase_price": rec_buy,
                "recommended_purchase_price_aud": aud_rec_buy,                "notes": " ".join(note) if isinstance(note, list) else "",
            },
        },

        "meta": {
            "match": {
                "id": pid,
                "product_name": best.get("product-name"),
                "console_name": best.get("console-name"),
                "sales_volume": best.get("sales-volume"),
            },            "sources": ["pricecharting_csv"],
            "raw_fields": {
                "loose_price": loose,
                "new_price": newp,
                "graded_price": graded,
                "condition_17_price": cond17,
                "condition_18_price": cond18,
                "bgs_10_price": bgs10,
            }
        },

        "grading_value_summary": grading_value_summary,

        "disclaimer": "Informational market context only. Figures are third-party estimates and may vary. Not financial advice.",
    }, status_code=200)

@app.post("/api/market-context-item")

@safe_endpoint
async def market_context_item(
    query: str = Form(...),
    category_hint: str = Form(""),  # optional: pokemon-cards, magic-cards, sports-cards, etc.
    condition: str = Form(""),
):
    q = _norm_ws(query)
    if not q:
        raise HTTPException(status_code=400, detail="query required")

    if not PRICECHARTING_TOKEN:
        return JSONResponse(content={
            "available": False,
            "mode": "click_only",
            "query": q,
            "message": "PRICECHARTING_TOKEN not configured.",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "disclaimer": "Informational market context only. Not financial advice.",
        })

    products = await _pc_search(q, category=category_hint or None, limit=10)
    if not products:
        return JSONResponse(content={
            "available": False,
            "mode": "click_only",
            "query": q,
            "message": "No matching products found on PriceCharting for this query.",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "disclaimer": "Informational market context only. Not financial advice.",
        })

    top = products[0]
    pid = str(top.get("id") or top.get("product-id") or "").strip()
    detail = await _pc_product(pid) if pid else {}
    prices = _pc_extract_price_fields(detail or top)
    url = (detail or {}).get("url") or top.get("url") or ""

    # Provide a "typical" figure by averaging any available price fields
    vals = [v for v in prices.values() if isinstance(v, (int, float))]
    typical = round(mean(vals), 2) if vals else None

    return JSONResponse(content={
        "available": True,
        "mode": "click_only",
        "query": q,
        "condition_hint": _norm_ws(condition),
        "pricecharting": {
            "best_match": top,
            "url": url,
            "prices": prices,
            "alternatives": products[:5],
        },
        "observed": {"typical_value_estimate_usd": typical, "typical_value_estimate_aud": _usd_to_aud_simple(typical) if typical is not None else None, "usd_to_aud_multiplier": AUD_MULTIPLIER},
        "sources": ["PriceCharting API"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "disclaimer": "Informational market context only. Figures are third-party estimates and may vary. Not financial advice.",
    })

# ==============================
# eBay API scaffold (disabled)
# ==============================
async def _ebay_sold_prices_stub(query: str) -> Dict[str, Any]:
    """Placeholder: wire this up once your eBay developer account is approved."""
    return {"available": False, "message": "eBay API not enabled.", "query": query}

# ==============================
# Runner
# ==============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
