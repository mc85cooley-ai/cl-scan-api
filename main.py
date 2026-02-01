"""
The Collectors League Australia — Scan API
Futureproof v6.4.0 (2026-01-31)

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
# Local Data Store (PriceCharting history)
# ==============================
# Use Render persistent disk (recommended) or any writable path.
# Configure in Render: add a Disk and set CL_DATA_DIR to that mount (e.g. /var/data).
CL_DATA_DIR = os.getenv("CL_DATA_DIR", "/var/data/cl-scan-api").strip() or "/var/data/cl-scan-api"
try:
    os.makedirs(CL_DATA_DIR, exist_ok=True)
except Exception:
    # Fallback to /tmp (ephemeral) if filesystem is locked
    CL_DATA_DIR = "/tmp/cl-scan-api"
    os.makedirs(CL_DATA_DIR, exist_ok=True)

PRICECHARTING_DB_PATH = os.path.join(CL_DATA_DIR, "pricecharting.sqlite")
PRICECHARTING_CACHE_DIR = os.path.join(CL_DATA_DIR, "pricecharting")
os.makedirs(PRICECHARTING_CACHE_DIR, exist_ok=True)

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
        return {"available": False, "samples": len(rows or [])}

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
        return {"available": False, "samples": len(series)}

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
        "samples": len(series),
    }


POKEMONTCG_BASE = "https://api.pokemontcg.io/v2"

# ==============================
# Set Code Mapping + Canonicalization
# ==============================
SET_CODE_MAP: Dict[str, str] = {
    "MEW": "Scarlet & Violet—151",
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
if USE_EBAY_API and not (EBAY_APP_ID and EBAY_OAUTH_TOKEN):
    print("INFO: USE_EBAY_API=1 set but eBay credentials are missing (will remain inactive).")

# ==============================
# Generic helpers
# ==============================
def _b64(img: bytes) -> str:
    return base64.b64encode(img).decode("utf-8")

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
# FX helper (USD -> AUD) with caching
# ==============================
_FX_CACHE = {"ts": 0.0, "rate": None}

async def _usd_to_aud_rate() -> Optional[float]:
    """Fetch USD->AUD rate. Best-effort; returns None on failure.
    Tries multiple providers to avoid a single-point failure.
    Caches for 1 hour.
    """
    import time
    now = time.time()
    if _FX_CACHE.get("rate") and (now - _FX_CACHE.get("ts", 0.0) < 3600):
        return _FX_CACHE["rate"]
    urls = [
        # 1) exchangerate.host
        ("https://api.exchangerate.host/latest", {"base": "USD", "symbols": "AUD"}, ("rates", "AUD")),
        # 2) open.er-api.com
        ("https://open.er-api.com/v6/latest/USD", None, ("rates", "AUD")),
        # 3) frankfurter.app
        ("https://api.frankfurter.app/latest", {"from": "USD", "to": "AUD"}, ("rates", "AUD")),
    ]
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            for url, params, path in urls:
                try:
                    r = await client.get(url, params=params)
                    if r.status_code != 200:
                        continue
                    data = r.json() if r.content else {}
                    node = data
                    for k in path:
                        if isinstance(node, dict) and k in node:
                            node = node[k]
                        else:
                            node = None
                            break
                    rate = None
                    try:
                        rate = float(node) if node else None
                    except Exception:
                        rate = None
                    if rate and rate > 0:
                        _FX_CACHE["ts"] = now
                        _FX_CACHE["rate"] = rate
                        return rate
                except Exception:
                    continue
    except Exception:
        return None
    return None

async def _usd_to_aud(amount: Any) -> Tuple[Optional[float], Optional[float]]:
    """Return (aud_amount, rate_used)."""
    try:
        amt = float(amount)
    except Exception:
        return (None, None)
    rate = await _usd_to_aud_rate()
    if not rate:
        return (None, None)
    return (round(amt * rate, 2), float(rate))


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

# ==============================
# Condition/value helpers
# ==============================
def _condition_bucket_from_pregrade(g: Optional[int]) -> str:
    """Map a 1-10 pregrade estimate to a simple condition bucket."""
    if g is None:
        return "Unknown"
    if g >= 9:
        return "Mint"
    if g == 8:
        return "Near Mint"
    if g == 7:
        return "Excellent"
    if g == 6:
        return "Good"
    if g == 5:
        return "Fair"
    return "Poor"

def _condition_multiplier_from_pregrade(g: Optional[int]) -> float:
    """Negative multiplier to convert 'mint' market context to 'as-is' value.
    Tuned so a very poor raw card can be ~0.16x of mint (e.g., $5 -> $0.80).
    """
    if g is None:
        return 1.0
    ladder = {10:1.00, 9:0.90, 8:0.80, 7:0.70, 6:0.60, 5:0.45, 4:0.30, 3:0.22, 2:0.18, 1:0.16}
    return float(ladder.get(int(g), 1.0))

def _safe_money_mul(v: Any, m: float) -> Optional[float]:
    try:
        if v is None:
            return None
        vv = float(v)
        if vv <= 0:
            return None
        return round(vv * float(m), 2)
    except Exception:
        return None


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
        "use_ebay_api": bool(USE_EBAY_API and EBAY_APP_ID and EBAY_OAUTH_TOKEN),
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
    card_name: Optional[str] = Form(None),
    card_set: Optional[str] = Form(None),
    card_number: Optional[str] = Form(None),
    card_year: Optional[str] = Form(None),
    card_type: Optional[str] = Form(None),
    set_code: Optional[str] = Form(None),
):
    front_bytes = await front.read()
    back_bytes = await back.read()
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

Analyze BOTH images (front + back) with EXTREME scrutiny. Write as if speaking directly to a collector who needs honest, specific feedback about their card.

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
    "spoken_word": "A punchy spoken-word version of the assessment summary (about 20–45 seconds). First person, conversational. Mention the best features, the main grade limiters, and end with what grade you’d realistically expect.",
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
        ],
    }]

    result = await _openai_chat(msg, max_tokens=2200, temperature=0.1)
    if result.get("error"):
        return JSONResponse(content={"error": True, "message": "AI grading failed", "details": result.get("message", "")}, status_code=502)

    data = _parse_json_or_none(result.get("content", "")) or {}

    raw_pregrade = str(data.get("pregrade", "")).strip()
    g = _grade_bucket(raw_pregrade)

    # Conservative safety net: if the model lists severe physical damage, cap the grade accordingly.
    # This helps avoid "Mint-ish" outputs when the card is clearly damaged.
    try:
        flags_txt = " ".join([str(x) for x in (data.get("flags") or [])]).lower()
        defects_txt = " ".join([str(x) for x in (data.get("defects") or [])]).lower()
        dmg_txt = (flags_txt + " " + defects_txt).strip()

        if g is not None:
            if any(k in dmg_txt for k in ["tear", "ripped", "major crease", "strong crease", "severe crease", "fold", "water damage"]):
                g = min(g, 4)
            elif any(k in dmg_txt for k in ["crease", "bent", "bend", "dent", "ding", "impression", "peeling", "lifted foil"]):
                g = min(g, 5)
            elif any(k in dmg_txt for k in ["heavy whitening", "heavy edge", "edge chipping", "rounded corners", "corner rounding"]):
                g = min(g, 6)
    except Exception:
        pass


    # Additional conservative caps based on structured fields (corners/edges/surface).
    # Helps catch cases where obvious damage isn't captured in flags/defects.
    try:
        corners = data.get("corners") if isinstance(data.get("corners"), dict) else {}
        cfront = corners.get("front") if isinstance(corners.get("front"), dict) else {}
        cback = corners.get("back") if isinstance(corners.get("back"), dict) else {}

        def _corner_blob(side: dict) -> str:
            parts = []
            for k in ("top_left", "top_right", "bottom_left", "bottom_right"):
                v = side.get(k) if isinstance(side.get(k), dict) else {}
                parts.append(str(v.get("condition", "")))
                parts.append(str(v.get("notes", "")))
            return " ".join(parts).lower()

        corner_txt = (_corner_blob(cfront) + " " + _corner_blob(cback)).strip()

        edges = data.get("edges") if isinstance(data.get("edges"), dict) else {}
        ef = edges.get("front") if isinstance(edges.get("front"), dict) else {}
        eb = edges.get("back") if isinstance(edges.get("back"), dict) else {}
        edge_txt = " ".join([
            str(ef.get("grade", "")), str(ef.get("notes", "")),
            str(eb.get("grade", "")), str(eb.get("notes", ""))
        ]).lower()

        surface = data.get("surface") if isinstance(data.get("surface"), dict) else {}
        sf = surface.get("front") if isinstance(surface.get("front"), dict) else {}
        sb = surface.get("back") if isinstance(surface.get("back"), dict) else {}
        surf_txt = " ".join([
            str(sf.get("grade", "")), str(sf.get("notes", "")),
            str(sb.get("grade", "")), str(sb.get("notes", ""))
        ]).lower()

        full_txt = " ".join([corner_txt, edge_txt, surf_txt]).strip()

        if g is not None:
            if any(k in full_txt for k in ["tear", "ripped", "water damage", "mold", "major crease", "severe crease", "fold"]):
                g = min(g, 3)
            if any(k in full_txt for k in ["crease", "bent", "bend", "dent", "ding", "impression"]):
                g = min(g, 4)
            if ("poor" in edge_txt) or ("poor" in surf_txt):
                g = min(g, 4)
            if full_txt.count("whitening") >= 3 or any(k in full_txt for k in ["edge chipping", "rounded", "rounding", "heavy whitening"]):
                g = min(g, 6)
    except Exception:
        pass

    pregrade_norm = str(g) if g is not None else ""

    # Ensure assessment_summary is detailed enough (UI-friendly)
    summary = _norm_ws(str(data.get("assessment_summary", "")))
    if len(summary.split()) < 35:
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
        parts.append(
            f"Overall, this looks like a PSA-style {pregrade_norm or raw_pregrade or 'N/A'} estimate based on what is visible in the photos."
        )
        if cen_f.get("grade") or cen_b.get("grade"):
            parts.append(
                f"Centering appears around Front {str(cen_f.get('grade','')).strip() or 'N/A'} and Back {str(cen_b.get('grade','')).strip() or 'N/A'}."
            )
        if ef.get("grade") or eb.get("grade"):
            parts.append(
                f"Edges read as Front {str(ef.get('grade','')).strip() or 'N/A'} / Back {str(eb.get('grade','')).strip() or 'N/A'}; notes: "
                f"{_norm_ws(str(ef.get('notes','')))} {_norm_ws(str(eb.get('notes','')))}"
            )
        if sf.get("grade") or sb.get("grade"):
            parts.append(
                f"Surface reads as Front {str(sf.get('grade','')).strip() or 'N/A'} / Back {str(sb.get('grade','')).strip() or 'N/A'}; notes: "
                f"{_norm_ws(str(sf.get('notes','')))} {_norm_ws(str(sb.get('notes','')))}"
            )
        if defects_list:
            parts.append(
                "Visible issues noted: "
                + "; ".join([_norm_ws(str(d)) for d in defects_list[:8]])
                + ("" if len(defects_list) <= 8 else " (and more).")
            )
        if flags_list:
            parts.append(
                "Key flags: "
                + ", ".join([_norm_ws(str(f)) for f in flags_list[:10]])
                + ("" if len(flags_list) <= 10 else ", …")
                + "."
            )
        parts.append(
            "Biggest grade limiters are the most severe corner/edge whitening/chipping, any surface scratches/print lines, and any bends/creases/dents if present."
        )
        summary = " ".join([p for p in parts if p]).strip()
        data["assessment_summary"] = summary

    g_for_mult = _grade_bucket(pregrade_norm) if pregrade_norm else _grade_bucket(raw_pregrade)
    return JSONResponse(content={
        "pregrade": pregrade_norm or "N/A",
        "confidence": _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0),
        "condition_distribution": data.get("condition_distribution", {}),
        "centering": data.get("centering", {"front": {"grade": "", "notes": ""}, "back": {"grade": "", "notes": ""}}),
        "corners": data.get("corners", {"front": {}, "back": {}}),
        "edges": data.get("edges", {"front": {"grade": "", "notes": ""}, "back": {"grade": "", "notes": ""}}),
        "surface": data.get("surface", {"front": {"grade": "", "notes": ""}, "back": {"grade": "", "notes": ""}}),
        "defects": data.get("defects", []) if isinstance(data.get("defects", []), list) else [],
        "flags": data.get("flags", []) if isinstance(data.get("flags", []), list) else [],
        "assessment_summary": _norm_ws(str(data.get("assessment_summary", ""))) or summary or "",
        "spoken_word": _norm_ws(str(data.get("spoken_word", ""))) or _norm_ws(str(data.get("assessment_summary", ""))) or summary or "",
        "observed_id": data.get("observed_id", {}) if isinstance(data.get("observed_id", {}), dict) else {},
        "condition_bucket": _condition_bucket_from_pregrade(g_for_mult),
        "condition_multiplier": _condition_multiplier_from_pregrade(g_for_mult),
        "verify_token": f"vfy_{secrets.token_urlsafe(12)}",
        "market_context_mode": "click_only",
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

Return ONLY valid JSON with these exact fields:

{
  "item_type": "sealed booster box/sealed pack/sealed tin/sealed case/autographed memorabilia/game-used memorabilia/graded item/other",
  "brand": "brand/league/publisher if visible (e.g., Pokemon, Panini, Topps, Upper Deck) else empty string",
  "product_name": "the main product name (e.g., 'Scarlet & Violet 151 Booster Box') else empty string",
  "set_or_series": "set/series name if visible else empty string",
  "year": "4 digit year if visible else empty string",
  "edition_or_language": "e.g., English/Japanese/1st Edition/Unlimited if visible else empty string",
  "special_attributes": ["e.g., Factory Sealed", "Pokemon Center", "Hobby Box", "1st Edition", "Shadowless"],
  "description": "detailed one-paragraph description of the item in plain English",
  "signatures": "names of any visible signatures or 'None visible'",
  "seal_condition": "Factory Sealed/Opened/Resealed/Damaged/Not applicable",
  "authenticity_notes": "authenticity indicators visible (holograms, stickers, COA) and any red flags",
  "notable_features": "unique features worth noting",
  "confidence": 0.0-1.0,
  "category_hint": "Pokemon/Magic/YuGiOh/Sports/OnePiece/Other"
}

Rules:
- Be specific. Do not invent a year/edition if you cannot see it.
- If it appears sealed, describe the wrap (tight/loose, tears, holes, bubbling) and label as Factory Sealed only if it looks consistent.
- If you cannot identify confidently, keep product_name empty and set confidence low.
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

    brand = _norm_ws(str(data.get("brand", "")))
    product_name = _norm_ws(str(data.get("product_name", "")))
    set_or_series = _norm_ws(str(data.get("set_or_series", "")))
    year = _norm_ws(str(data.get("year", "")))
    edition_or_language = _norm_ws(str(data.get("edition_or_language", "")))
    special_attributes = data.get("special_attributes", [])
    if not isinstance(special_attributes, list):
        special_attributes = []
    special_attributes = [_norm_ws(str(x)) for x in special_attributes if _norm_ws(str(x))]

    canonical_item_id = {
        "category": "sealed/memorabilia",
        "brand": brand,
        "product_name": product_name,
        "set_or_series": set_or_series,
        "year": year,
        "edition_or_language": edition_or_language,
        "special_attributes": special_attributes,
    }

    return JSONResponse(content={
        "item_type": _norm_ws(str(data.get("item_type", "Unknown"))),
        "brand": brand,
        "product_name": product_name,
        "set_or_series": set_or_series,
        "year": year,
        "edition_or_language": edition_or_language,
        "special_attributes": special_attributes,
        "canonical_item_id": canonical_item_id,
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

Analyze ALL images with the same strictness a card grader would use. Be conservative and specific.

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
    "notes": "detailed notes about seal/wrap (tightness, tears, holes, bubbling, seams)"
  }},
  "packaging_condition": {{
    "grade": "Mint/Near Mint/Excellent/Good/Fair/Poor",
    "notes": "detailed notes about box/packaging wear: corners, dents, crushing, scratches, scuffs, sticker residue"
  }},
  "signature_assessment": {{
    "present": true/false,
    "quality": "Clear/Faded/Smudged/Not Applicable",
    "notes": "notes about signature placement/ink flow/bleeding and any authenticity concerns"
  }},
  "value_factors": ["short bullets: print run, desirability, era, sealed premium, athlete/popularity, etc."],
  "defects": ["each defect as a full sentence with location + severity"],
  "flags": ["short flags for important issues (reseal risk, crush damage, water, heavy dents, COA missing)"],
  "overall_assessment": "5-8 sentences in first person (\"Looking at your item...\") explaining condition and what limits it.",
  "spoken_word": "A 20–45 second spoken-word version: best features, main issues, realistic condition grade.",
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
- Do NOT claim Factory Sealed unless the wrap/seal looks consistent.
- If glare/blur prevents certainty, say so and reduce confidence.
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
        "condition_distribution": data.get("condition_distribution", {}),
        "seal_integrity": seal,
        "packaging_condition": data.get("packaging_condition", {"grade": "N/A", "notes": ""}),
        "signature_assessment": data.get("signature_assessment", {"present": False, "quality": "Not Applicable", "notes": ""}),
        "value_factors": data.get("value_factors", []) if isinstance(data.get("value_factors", []), list) else [],
        "defects": data.get("defects", []) if isinstance(data.get("defects", []), list) else [],
        "overall_assessment": _norm_ws(str(data.get("overall_assessment", ""))),
        "spoken_word": _norm_ws(str(data.get("spoken_word", ""))) or _norm_ws(str(data.get("overall_assessment", ""))),
        "observed_id": data.get("observed_id", {}) if isinstance(data.get("observed_id", {}), dict) else {},
        "flags": data.get("flags", []) if isinstance(data.get("flags", []), list) else [],
        "verify_token": f"vfy_{secrets.token_urlsafe(12)}",
    })

# ==============================
# Market Context (Click-only) — Cards
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
    predicted_grade: Optional[str] = Form("9"),
    confidence: float = Form(0.0),
    grading_cost: float = Form(55.0),

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
        s = s.replace("—", "-").replace("–", "-")
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
        return "—"

    # --------------------------
    # Normalize inputs
    # --------------------------
    name_in = item_name or card_name or name or query or ""
    cat_in = item_category or card_type or "Other"
    set_in = item_set or card_set or ""
    num_in = item_number or card_number or ""

    clean_name = _norm_ws(str(name_in))
    clean_cat = _normalize_card_type(_norm_ws(str(cat_in)))
    clean_set = _norm_ws(str(set_in))
    clean_num_display = _clean_card_number_display(str(num_in))

    # Sealed/memorabilia heuristic (to choose matching strategy)
    itype = _norm_ws(str(item_type or "")) if item_type is not None else ""
    sealed_keywords = ["booster box","booster bundle","bundle","display","sealed","tin","etb","elite trainer","case","box","pack"]
    hay = (" ".join([clean_name, clean_set, itype])).lower()
    sealed_like = any(k in hay for k in sealed_keywords)

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
    # Sealed/memorabilia: try PriceCharting API search first (handles sealed categories better than CSV slugs)
    # --------------------------
    if sealed_like:
        try:
            q_api = " ".join([clean_name, clean_set]).strip() or clean_name
            products = await _pc_search(q_api, category=None, limit=8)
            if products:
                top = products[0]
                pid = str(top.get("id") or top.get("product-id") or "").strip()
                detail = await _pc_product(pid) if pid else {}
                prices = _pc_extract_price_fields(detail or top)
                vals = [v for v in prices.values() if isinstance(v, (int, float)) and v > 0]
                typical = round(sum(vals)/len(vals), 2) if vals else None

                g_ass = _grade_bucket(assessed_pregrade or "") or _grade_bucket(predicted_grade or "")
                mult = float(condition_multiplier) if isinstance(condition_multiplier, (int, float)) else _condition_multiplier_from_pregrade(g_ass)
                bucket = _condition_bucket_from_pregrade(g_ass)

                aud_rate = None
                aud_typical = None
                aud_as_is = None
                if typical is not None:
                    aud_typical, aud_rate = await _usd_to_aud(typical)
                    aud_as_is, _ = await _usd_to_aud(typical * mult)

                return JSONResponse(content={
                    "available": True,
                    "mode": "click_only",
                    "message": "Market context loaded (PriceCharting API)",
                    "used_query": q_api,
                    "query_ladder": [q_api],
                    "confidence": _clamp(_safe_float(confidence, 0.0), 0.0, 1.0),
                    "card": {"name": clean_name, "set": clean_set, "set_code": "", "year": "", "card_number": clean_num_display, "type": clean_cat},
                    "observed": {
                        "currency": ("AUD" if aud_rate else "USD"),
                        "fx": {"aud_rate": aud_rate, "aud_timestamp": datetime.utcnow().isoformat() + "Z"} if aud_rate else {},
                        "usd_original": {"raw_median": typical, "as_is_value": _safe_money_mul(typical, mult)},
                        "raw": {"median": (round(aud_typical, 2) if aud_rate and aud_typical is not None else typical), "avg": (round(aud_typical, 2) if aud_rate and aud_typical is not None else typical), "range": None, "samples": len(vals) if vals else 0},
                        "as_is": {"multiplier": mult, "bucket": bucket, "value": _safe_money_mul(typical, mult), "value_aud": round(aud_as_is,2) if aud_as_is is not None else None},
                        "pricecharting_prices": prices,
                    },
                    "disclaimer": "Informational market context only. Figures are third-party estimates and may vary. Not financial advice.",
                }, status_code=200)
        except Exception:
            pass

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
        pn = _pc_norm(r.get("product-name", ""))
        cn = _pc_norm(r.get("console-name", ""))

        score = 0
        if num_short is not None:
            if f"#{num_short}" in pn:
                score += 12
            elif str(num_short) in pn:
                score += 5

        for t in name_tokens:
            if t and t in pn:
                score += 3

        for t in set_tokens:
            if t and (t in cn or t in pn):
                score += 2

        if s_norm and s_norm in cn:
            score += 6

        if score > best_score:
            best_score = score
            best = r

    if not best or best_score < 10:
        return JSONResponse(content={
            "available": True,
            "mode": "click_only",
            "message": "No strong PriceCharting CSV match found for this item yet.",
            "used_query": f"{clean_name} {clean_set} {clean_num_display}".strip(),
            "query_ladder": [clean_name, f"{clean_name} {clean_set}".strip(), f"{clean_name} {clean_set} {clean_num_display}".strip()],
            "confidence": _clamp(_safe_float(confidence, 0.0), 0.0, 1.0),
            "card": {"name": clean_name, "set": clean_set, "set_code": "", "year": "", "card_number": clean_num_display, "type": clean_cat},
            "observed": {
                "currency": ("AUD" if aud_rate else "USD"),
                "liquidity": "—",
                "trend": "—",
                "raw": {"median": None, "avg": None, "range": None, "samples": 0},
                "graded_psa": {"10": None, "9": None, "8": None},
            },
            "grade_impact": {
                "expected_graded_value": None,
                "raw_baseline_value": None,
                "grading_cost": float(grading_cost or 0.0),
                "estimated_value_difference": None,
            },
            "meta": {
                "equivalency_note": "No match yet.",
            },
            "disclaimer": "Informational market context only. Figures are third-party estimates and may vary.",
        }, status_code=200)

    # --------------------------
    # Extract prices (CSV format)
    # --------------------------
    loose = _parse_money(best.get("loose-price"))
    newp = _parse_money(best.get("new-price"))
    graded = _parse_money(best.get("graded-price"))

    cond17 = _parse_money(best.get("condition-17-price"))  # commonly present in your CSV
    cond18 = _parse_money(best.get("condition-18-price"))
    bgs10 = _parse_money(best.get("bgs-10-price"))

    retail_loose_buy = _parse_money(best.get("retail-loose-buy"))
    retail_loose_sell = _parse_money(best.get("retail-loose-sell"))

    # Raw baseline: prefer loose, else new
    raw_val = loose if loose is not None else newp

    # Condition adjustment (treat PriceCharting baseline as mint-ish; convert to as-is)
    g_ass = _grade_bucket(assessed_pregrade or "") or _grade_bucket(predicted_grade or "")
    mult = float(condition_multiplier) if isinstance(condition_multiplier, (int, float)) else _condition_multiplier_from_pregrade(g_ass)
    bucket = _condition_bucket_from_pregrade(g_ass)

    # ---- "PSA-style" anchors (equivalency, not literal PSA comps) ----
    # IMPORTANT: We keep the UI labels familiar but we do NOT pretend these are official PSA comps.
    # Priority:
    # - PSA 10 equiv: BGS 10 if present, else condition-18, else graded
    # - PSA 9 equiv: condition-17, else graded
    # - PSA 8 equiv: graded
    psa10_equiv = bgs10 or graded
    psa9_equiv = cond17 or graded
    psa8_equiv = cond18 or graded

    # --------------------------
    # Liquidity & trend (if we have a product id + history)
    # --------------------------
    pid = (best.get("id") or "").strip()
    liquidity = _liquidity_label(best.get("sales-volume"))
    trend_label = "—"
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
    g_pred = _grade_bucket(predicted_grade) or 9
    dist = _grade_distribution(g_pred, conf_in)

    price_map = {"10": (aud_psa10 if aud_rate and aud_psa10 is not None else psa10_equiv), "9": (aud_psa9 if aud_rate and aud_psa9 is not None else psa9_equiv), "8": psa8_equiv}
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

    assume_grading = bool(pg >= 8.5 and conf_factor >= 0.75 and expected_val is not None and float(grading_cost or 0.0) > 0)

    rec_buy = None
    note = []

    if base_buy is not None:
        rec_buy = float(base_buy) * float(cond_factor) * float(conf_factor)
        note.append("Baseline uses retail-loose-buy adjusted by assessed grade & confidence.")

    if assume_grading:
        net_after_grading = float(expected_val) - float(grading_cost or 0.0)
        if net_after_grading > 0:
            # leave margin for fees/time/risk; confidence reduces aggression
            max_buy_from_grading = net_after_grading * 0.75 * conf_factor
            note.append("Grading upside considered: expected graded value minus grading cost, then margin.")
            if rec_buy is None:
                rec_buy = max_buy_from_grading
            else:
                rec_buy = max(rec_buy, max_buy_from_grading)

    # cap recommended buy to observed raw value if we have one
    if rec_buy is not None and raw_base is not None:
        cap_val = _safe_money_mul(raw_base, mult) or float(raw_base)
        rec_buy = min(float(rec_buy), float(cap_val))

    
    # --- AUD conversion (best-effort) ---
    aud_rate = None
    aud_raw = None
    aud_psa10 = None
    aud_psa9 = None
    aud_psa8 = None
    aud_expected = None
    aud_rec_buy = None
    try:
        if raw_val is not None:
            aud_raw, aud_rate = await _usd_to_aud(raw_val)
        if psa10_equiv is not None:
            aud_psa10, aud_rate = await _usd_to_aud(psa10_equiv)
        if psa9_equiv is not None:
            aud_psa9, aud_rate = await _usd_to_aud(psa9_equiv)
        if psa8_equiv is not None:
            aud_psa8, aud_rate = await _usd_to_aud(psa8_equiv)
        if expected_val is not None:
            aud_expected, aud_rate = await _usd_to_aud(expected_val)
        if rec_buy is not None:
            aud_rec_buy, aud_rate = await _usd_to_aud(rec_buy)
    except Exception:
        pass

    rec_buy = (round(float(rec_buy), 2) if rec_buy is not None else None)

    # UI expects these keys
    return JSONResponse(content={
        "available": True,
        "mode": "click_only",
        "message": "Market context loaded",
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
            "currency": ("AUD" if aud_rate else "USD"),
            "usd_original": {"raw_median": raw_val, "psa10": psa10_equiv, "psa9": psa9_equiv, "psa8": psa8_equiv} if aud_rate else {},
            "fx": {"aud_rate": aud_rate, "aud_timestamp": datetime.utcnow().isoformat() + "Z"} if aud_rate else {},
            "aud": {"raw_median": aud_raw, "psa10": aud_psa10, "psa9": aud_psa9, "psa8": aud_psa8} if aud_rate else {},
            "liquidity": liquidity,
            "trend": trend_label,
            "raw": {
                "median": (aud_raw if aud_rate and aud_raw is not None else raw_val),
                "avg": (aud_raw if aud_rate and aud_raw is not None else raw_val),
                "range": None,
                "samples": 1 if raw_val is not None else 0,
            },
            "graded_psa": {
                "10": (aud_psa10 if aud_rate and aud_psa10 is not None else psa10_equiv),
                "9": (aud_psa9 if aud_rate and aud_psa9 is not None else psa9_equiv),
                "8": (aud_psa8 if aud_rate and aud_psa8 is not None else psa8_equiv),
            
            "as_is": {
                "multiplier": mult,
                "bucket": bucket,
                "raw_median": _safe_money_mul((aud_raw if aud_rate and aud_raw is not None else raw_val), mult),
                "psa10_equiv": _safe_money_mul((aud_psa10 if aud_rate and aud_psa10 is not None else psa10_equiv), mult),
                "psa9_equiv": _safe_money_mul((aud_psa9 if aud_rate and aud_psa9 is not None else psa9_equiv), mult),
                "psa8_equiv": _safe_money_mul((aud_psa8 if aud_rate and aud_psa8 is not None else psa8_equiv), mult)
            },
},
        },

        "grade_impact": {
            "expected_graded_value": (aud_expected if aud_rate and aud_expected is not None else expected_val),
            "expected_graded_value_aud": aud_expected if aud_rate else None,
            "raw_baseline_value": (aud_raw if aud_rate and aud_raw is not None else raw_base),
            "grading_cost": float(grading_cost or 0.0),
            "estimated_value_difference": (round((aud_expected or 0) - (aud_raw or 0) - float(grading_cost or 0.0), 2) if (aud_rate and aud_expected is not None and aud_raw is not None) else diff),
            "recommended_buy": {
                "currency": ("AUD" if aud_rate else "USD"),
                "retail_loose_buy": retail_loose_buy,
                "retail_loose_sell": retail_loose_sell,
                "usd_original": {"recommended_purchase_price": rec_buy} if aud_rate else {},
                "recommended_purchase_price": (aud_rec_buy if aud_rate and aud_rec_buy is not None else rec_buy),
                "recommended_purchase_price_aud": aud_rec_buy if aud_rate else None,
                "assume_grading": assume_grading,
                "notes": " ".join(note) if isinstance(note, list) else "",
            },
        },

        "meta": {
            "match": {
                "id": pid,
                "product_name": best.get("product-name"),
                "console_name": best.get("console-name"),
                "sales_volume": best.get("sales-volume"),
            },
            "equivalency_note": "Front-end labels use PSA-style equivalents (est.). Mapping: BGS 10 → PSA 10 (est.), condition-17 → PSA 9 (est.), condition-18 → PSA 8 (est.). Not official PSA comps.",
            "sources": ["pricecharting_csv"],
            "raw_fields": {
                "loose_price": loose,
                "new_price": newp,
                "graded_price": graded,
                "condition_17_price": cond17,
                "condition_18_price": cond18,
                "bgs_10_price": bgs10,
            }
        },

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
        "observed": {"typical_value_estimate": typical},
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
