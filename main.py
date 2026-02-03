"""
The Collectors League Australia - Scan API
Futureproof v6.7.4 (2026-02-03)

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
import statistics
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
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
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
            return JSONResponse(
                content={"error": True, "endpoint": func.__name__, "message": str(e)},
                status_code=500
            )
    return wrapper


# ==============================
# Config
# ==============================
APP_VERSION = os.getenv("CL_SCAN_VERSION", "2026-01-31-v6.5.0")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
POKEMONTCG_API_KEY = os.getenv("POKEMONTCG_API_KEY", "").strip()
PRICECHARTING_TOKEN = os.getenv("PRICECHARTING_TOKEN", "").strip()
ADMIN_TOKEN = os.getenv("CL_ADMIN_TOKEN", "").strip()  # optional

# PriceCharting local storage (even if eBay is primary, some endpoints still reference these)
PRICECHARTING_CACHE_DIR = os.getenv("PRICECHARTING_CACHE_DIR", "/tmp/pricecharting_cache").strip() or "/tmp/pricecharting_cache"
PRICECHARTING_DB_PATH = os.getenv("PRICECHARTING_DB_PATH", "/tmp/pricecharting.db").strip() or "/tmp/pricecharting.db"
try:
    os.makedirs(PRICECHARTING_CACHE_DIR, exist_ok=True)
except Exception:
    pass


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


async def _ebay_completed_stats(keyword_query: str, limit: int = 120, days_lookback: int = 120) -> dict:
    """
    Fetch eBay completed/sold items stats using FindingService (AppID only).

    Returns:
      {
        source: "ebay",
        query: str,
        count: int,
        currency: "AUD",
        prices: list[float],
        low: float|None,   # p20
        median: float|None,# p50
        high: float|None,  # p80
        avg: float|None,
        p20: float|None,
        p80: float|None,
        min: float|None,
        max: float|None
      }

    Notes:
      - Uses SoldItemsOnly completed listings.
      - Prices are converted to AUD when currency is USD.
      - low/high are p20/p80 to reduce outlier impact.
    """
    q = _norm_ws(keyword_query or "")
    if not q or not USE_EBAY_API or not EBAY_APP_ID:
        return {}

    # Cache per query+limit+lookback
    cache_key = f"ebay:{q}:{limit}:{days_lookback}"
    now = int(time.time())
    cached = _EBAY_CACHE.get(cache_key)
    if cached and (now - int(cached.get("ts", 0))) < 1800:  # 30 min
        return cached.get("data", {}) or {}

    # eBay FindingService pages: 1..N, 100 per page
    target = max(1, int(limit))
    pages = min(5, (target + 99) // 100)  # safety cap

    prices: list[float] = []

    url = "https://svcs.ebay.com/services/search/FindingService/v1"
    headers = {"User-Agent": UA} if "UA" in globals() else None

    async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
        for page in range(1, pages + 1):
            params = {
                "OPERATION-NAME": "findCompletedItems",
                "SERVICE-VERSION": "1.13.0",
                "SECURITY-APPNAME": EBAY_APP_ID,
                "RESPONSE-DATA-FORMAT": "JSON",
                "REST-PAYLOAD": "true",
                "keywords": q,
                "itemFilter(0).name": "SoldItemsOnly",
                "itemFilter(0).value": "true",
                "paginationInput.entriesPerPage": "100",
                "paginationInput.pageNumber": str(page),
            }

            try:
                r = await client.get(url, params=params)
                r.raise_for_status()
                j = r.json()
            except Exception:
                continue

            try:
                items = (
                    j.get("findCompletedItemsResponse", [{}])[0]
                     .get("searchResult", [{}])[0]
                     .get("item", [])
                )
            except Exception:
                items = []

            for it in items or []:
                try:
                    selling = it.get("sellingStatus", [{}])[0]
                    cp = selling.get("currentPrice", [{}])[0]
                    val = float(cp.get("__value__", 0.0))
                    cur = str(cp.get("@currencyId", "")).upper()
                    if val <= 0:
                        continue
                    if cur == "USD":
                        val = float(_usd_to_aud_simple(val) or 0.0)
                    # Ignore insane values (protect against parsing weird lots)
                    if val <= 0 or val > 1_000_000:
                        continue
                    prices.append(val)
                except Exception:
                    continue

            if len(prices) >= target:
                break

    prices = sorted(prices)
    count = len(prices)
    if count == 0:
        out = {"source": "ebay", "query": q, "count": 0, "currency": "AUD", "prices": []}
        _EBAY_CACHE[cache_key] = {"ts": now, "data": out}
        return out

    def pct(p: float) -> float:
        # p in [0..1]
        idx = int(round(p * (count - 1)))
        idx = max(0, min(count - 1, idx))
        return float(prices[idx])

    out = {
        "source": "ebay",
        "query": q,
        "count": count,
        "currency": "AUD",
        "prices": prices[:target],
        "low": pct(0.20),
        "median": pct(0.50),
        "high": pct(0.80),
        "avg": float(sum(prices) / count),
        "p20": pct(0.20),
        "p80": pct(0.80),
        "min": float(prices[0]),
        "max": float(prices[-1]),
    }
    _EBAY_CACHE[cache_key] = {"ts": now, "data": out}
    return out


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

try:
    _pc_init_db()
except Exception as _e:
    print(f"INFO: PriceCharting DB init skipped: {_e}")


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
# Finding/Legacy (SOLD) uses "Dev ID" as AppID for FindingService
EBAY_APP_ID = os.getenv("EBAY_APP_ID", "").strip()

# Browse/Buy APIs (ACTIVE) use OAuth application token.
# Prefer auto-fetch via client credentials; EBAY_OAUTH_TOKEN remains as a fallback/manual override.
EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID", "").strip()
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET", "").strip()
EBAY_OAUTH_TOKEN = os.getenv("EBAY_OAUTH_TOKEN", "").strip()

# In-process OAuth cache (per Render instance). Tokens are short-lived; we refresh automatically.
_EBAY_OAUTH_STATE = {"token": "", "exp": 0, "source": "none"}  # exp = epoch seconds
_EBAY_OAUTH_LOCK = None  # created lazily to avoid event-loop issues

def _ebay_oauth_lock():
    global _EBAY_OAUTH_LOCK
    if _EBAY_OAUTH_LOCK is None:
        import asyncio
        _EBAY_OAUTH_LOCK = asyncio.Lock()
    return _EBAY_OAUTH_LOCK

async def _get_ebay_app_token(force_refresh: bool = False) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Returns (token, debug). Uses client-credentials to fetch an *application* token for Buy/Browse.
    Falls back to EBAY_OAUTH_TOKEN if client creds are not set.
    """
    now = int(time.time())
    dbg: Dict[str, Any] = {"ok": False, "source": None, "status": None, "error": None}

    # Manual override token (legacy) if provided and no client creds, or if caching is disabled.
    if (not EBAY_CLIENT_ID or not EBAY_CLIENT_SECRET) and EBAY_OAUTH_TOKEN:
        dbg.update({"ok": True, "source": "env:EBAY_OAUTH_TOKEN"})
        return EBAY_OAUTH_TOKEN, dbg

    if not EBAY_CLIENT_ID or not EBAY_CLIENT_SECRET:
        dbg.update({"ok": False, "source": "none", "error": "missing EBAY_CLIENT_ID/EBAY_CLIENT_SECRET"})
        return None, dbg

    # Cached token still valid (60s safety window)
    if not force_refresh and _EBAY_OAUTH_STATE.get("token") and now < int(_EBAY_OAUTH_STATE.get("exp", 0) - 60):
        dbg.update({"ok": True, "source": _EBAY_OAUTH_STATE.get("source", "cache")})
        return str(_EBAY_OAUTH_STATE["token"]), dbg

    lock = _ebay_oauth_lock()
    async with lock:
        # Double-check after waiting
        now2 = int(time.time())
        if not force_refresh and _EBAY_OAUTH_STATE.get("token") and now2 < int(_EBAY_OAUTH_STATE.get("exp", 0) - 60):
            dbg.update({"ok": True, "source": _EBAY_OAUTH_STATE.get("source", "cache")})
            return str(_EBAY_OAUTH_STATE["token"]), dbg

        token_url = "https://api.ebay.com/identity/v1/oauth2/token"
        # Minimal scope for public data
        scope = os.getenv("EBAY_OAUTH_SCOPE", "https://api.ebay.com/oauth/api_scope").strip() or "https://api.ebay.com/oauth/api_scope"

        auth = base64.b64encode(f"{EBAY_CLIENT_ID}:{EBAY_CLIENT_SECRET}".encode("utf-8")).decode("ascii")
        headers = {
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": UA,
        }
        data = {"grant_type": "client_credentials", "scope": scope}

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.post(token_url, data=data, headers=headers)
                dbg["status"] = r.status_code
                if r.status_code != 200:
                    dbg["error"] = (r.text or "")[:400]
                    return None, dbg
                j = r.json()
        except Exception as e:
            dbg["error"] = str(e)
            return None, dbg

        tok = str(j.get("access_token") or "").strip()
        exp_in = int(j.get("expires_in") or 0)
        if not tok or exp_in <= 0:
            dbg["error"] = "token response missing access_token/expires_in"
            return None, dbg

        _EBAY_OAUTH_STATE["token"] = tok
        _EBAY_OAUTH_STATE["exp"] = int(time.time()) + exp_in
        _EBAY_OAUTH_STATE["source"] = "oauth:client_credentials"
        dbg.update({"ok": True, "source": "oauth:client_credentials"})
        return tok, dbg

# eBay marketplace + display defaults
EBAY_MARKETPLACE_ID = os.getenv("EBAY_MARKETPLACE_ID", os.getenv("EBAY_MARKETPLACE", "EBAY_AU")).strip() or "EBAY_AU"
DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "AUD").strip().upper() or "AUD"
EXCLUDE_GRADED_DEFAULT = os.getenv("EXCLUDE_GRADED_DEFAULT", "1").strip() in ("1","true","TRUE","yes","YES")
STRICT_CARD_NUMBER = os.getenv("STRICT_CARD_NUMBER", "1").strip() in ("1","true","TRUE","yes","YES")


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
if USE_EBAY_API:
    if not EBAY_APP_ID:
        print("INFO: USE_EBAY_API=1 set but EBAY_APP_ID is missing (SOLD comps via FindingService will be inactive).")
    if not (EBAY_CLIENT_ID and EBAY_CLIENT_SECRET) and not EBAY_OAUTH_TOKEN:
        print("INFO: USE_EBAY_API=1 set but OAuth credentials are missing (ACTIVE comps via Browse API will be inactive).")

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


async def _ebay_active_stats(keyword_query: str, limit: int = 120) -> dict:
    """
    Fetch eBay ACTIVE listings stats using FindingService (AppID only).
    Returns same shape as _ebay_completed_stats but represents current asking prices.
    """
    q = _norm_ws(keyword_query or "")
    if not q or not USE_EBAY_API or not EBAY_APP_ID:
        return {}

    cache_key = f"ebay_active:{q}:{limit}"
    now = int(time.time())
    cached = _EBAY_CACHE.get(cache_key)
    if cached and (now - int(cached.get("ts", 0))) < 900:  # 15 min
        return cached.get("data", {}) or {}

    target = max(1, int(limit))
    pages = min(5, max(1, (target + 99) // 100))
    prices = []

    url = "https://svcs.ebay.com/services/search/FindingService/v1"
    headers = {"User-Agent": UA}

    async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
        for page in range(1, pages + 1):
            params = {
                "OPERATION-NAME": "findItemsAdvanced",
                "SERVICE-VERSION": "1.13.0",
                "SECURITY-APPNAME": EBAY_APP_ID,
                "RESPONSE-DATA-FORMAT": "JSON",
                "REST-PAYLOAD": "true",
                "keywords": q,
                "paginationInput.entriesPerPage": "100",
                "paginationInput.pageNumber": str(page),
                # Prefer fixed price + auction; don't restrict too hard.
                "itemFilter(0).name": "HideDuplicateItems",
                "itemFilter(0).value": "true",
            }
            try:
                r = await client.get(url, params=params)
                r.raise_for_status()
                j = r.json()
            except Exception:
                continue

            try:
                items = (
                    j.get("findItemsAdvancedResponse", [{}])[0]
                     .get("searchResult", [{}])[0]
                     .get("item", [])
                )
            except Exception:
                items = []

            for it in items:
                try:
                    selling = (it.get("sellingStatus") or [{}])[0]
                    cp = (selling.get("currentPrice") or [{}])[0]
                    p = float(cp.get("__value__", 0.0))
                    cur = cp.get("@currencyId", "USD")
                    if p <= 0:
                        continue
                    if str(cur).upper() == "USD":
                        p = _usd_to_aud_simple(p)
                    prices.append(p)
                except Exception:
                    continue

            if len(prices) >= target:
                break

    prices = sorted(prices)
    count = len(prices)
    if count == 0:
        out = {"source": "ebay", "query": q, "count": 0, "currency": "AUD", "prices": []}
        _EBAY_CACHE[cache_key] = {"ts": now, "data": out}
        return out

    def pct(p: float) -> float:
        idx = max(0, min(count - 1, int(round(p * (count - 1)))))
        return float(prices[idx])

    out = {
        "source": "ebay",
        "query": q,
        "count": count,
        "currency": "AUD",
        "prices": prices,
        "low": pct(0.20),
        "median": pct(0.50),
        "high": pct(0.80),
        "avg": float(sum(prices) / count),
        "p20": pct(0.20),
        "p80": pct(0.80),
        "min": float(prices[0]),
        "max": float(prices[-1]),
    }
    _EBAY_CACHE[cache_key] = {"ts": now, "data": out}
    return out


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

def _safe_end_time(item: dict) -> str:
    """Extract end time (ISO-ish) from eBay Finding API item dict if present."""
    try:
        li = item.get("listingInfo") or item.get("listinginfo") or None
        if isinstance(li, list) and li:
            li0 = li[0] or {}
        elif isinstance(li, dict):
            li0 = li
        else:
            li0 = {}
        et = li0.get("endTime") or li0.get("endtime") or ""
        if isinstance(et, list):
            return str(et[0] if et else "")
        return str(et)
    except Exception:
        return ""

def _parse_grade_from_title(title: str) -> Optional[float]:
    """Best-effort parse of a numeric grade from a listing title (brand-agnostic)."""
    t = (title or "").lower()
    if any(x in t for x in ["ungraded", "not graded", "no grade", "no grading"]):
        return None
    patterns = [
        r"\b(?:psa|bgs|cgc|sgc|beckett)\s*(10|9\.5|9|8|7|6|5|4|3|2|1)\b",
        r"\b(?:grade|graded)\s*(10|9\.5|9|8|7|6|5|4|3|2|1)\b",
        r"\b(?:gem\s*mint|gm)\s*(10)\b",
        r"\b(?:pristine)\s*(10)\b",
        r"\b(?:mint)\s*(9\.5|9)\b",
        r"\b(?:near\s*mint)\s*(8|9)\b",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None

def _grade_bucket_key(g: Optional[float]) -> Optional[str]:
    """Map parsed grade into display buckets (e.g., 10, 9, 8, 9.5)."""
    if g is None:
        return None
    if abs(g - round(g)) < 1e-6:
        return str(int(round(g)))
    s = str(g).rstrip("0").rstrip(".")
    return s if s else None

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
    # Accept token from form body (admin_token). You can extend this to read headers via Request injection later.
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
    # SECOND PASS GUARANTEE:
    # Enhanced filtered images (grayscale/autocontrast + contrast/sharpness)
    # are ALWAYS generated when PIL is available and are fed back into
    # grading logic to surface print lines, whitening, scratches, and dents.

    second_pass = {"enabled": True, "ran": False, "skipped_reason": None, "glare_suspects": [], "defect_candidates": []}
    angled_bytes = b""  # identify() has no angled image
    try:
        # Only run for cards; memorabilia uses a different endpoint.
        front_vars = _make_defect_filter_variants(img)
        back_vars = {}

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
1) Be conversational and specific. Write like you're examining the card in person and describing what you see:
   - BAD: "Minor edge wear present"
   - GOOD: "Looking at the front, I can see some very slight edge wear along the top edge, approximately 2mm from the top-left corner. The right edge is notably cleaner."

2) Call out every single corner individually with precise location and severity:
   - For EACH of the 8 corners (4 front + 4 back), describe what you observe
   - Examples: "Front top-left corner is perfectly sharp", "Back bottom-right shows minor whitening about 1mm deep"

3) Grade must reflect worst visible defect (conservative PSA-style):
   - Any crease/fold/tear/major dent → pregrade 4 or lower
   - Any bend/ding/impression, heavy rounding → pregrade 5 or lower
   - Moderate whitening across multiple corners/edges → pregrade 6-7
   - Only grade 9-10 if truly exceptional


5) Do NOT confuse holo sheen / light refraction / texture for damage:
   - If a mark disappears or changes drastically in the ANGLED shot, treat it as glare/reflection, NOT whitening/damage.
   - Print lines are typically straight and consistent across lighting; glare moves with angle.
   - Card texture (especially modern) is not damage unless there is a true crease, indentation, or paper break.

4) Write the assessment summary in first person, conversational style (5-8 sentences):
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
    # SECOND PASS GUARANTEE:
    # Enhanced filtered images (grayscale/autocontrast + contrast/sharpness)
    # are ALWAYS generated when PIL is available and are fed back into
    # grading logic to surface print lines, whitening, scratches, and dents.

    second_pass = {"enabled": True, "ran": False, "skipped_reason": None, "glare_suspects": [], "defect_candidates": []}
    try:
        # Only run for cards; memorabilia uses a different endpoint.
        front_vars = _make_defect_filter_variants(img)
        back_vars = {}  # identify() only has a front image

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

    # Condition anchor for downstream market logic (trust gate)
    g_int = None
    try:
        g_int = int(round(float(pregrade_norm))) if pregrade_norm else None
    except Exception:
        g_int = None
    condition_anchor = "damaged" if (has_structural_damage or (g_int is not None and g_int <= 4)) else ("low" if (g_int is not None and g_int <= 6) else ("mid" if (g_int is not None and g_int <= 8) else "high"))
    

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
        "condition_anchor": condition_anchor,
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

    predicted_grade: Optional[str] = Form(None),
    confidence: float = Form(0.0),
    grading_cost: float = Form(35.0),

    # Trust gates / coupling
    has_structural_damage: Optional[bool] = Form(False),
    assessed_pregrade: Optional[str] = Form(None),

    # Back-compat aliases from older frontends
    card_name: Optional[str] = Form(None),
    card_type: Optional[str] = Form(None),
    card_set: Optional[str] = Form(None),
    card_number: Optional[str] = Form(None),

    # Extra loose aliases (some JS sends these)
    name: Optional[str] = Form(None),
    query: Optional[str] = Form(None),
    item_type: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    seal_condition: Optional[str] = Form(None),

    # Controls (optional)
    exclude_graded: Optional[bool] = Form(None),   # if None, uses EXCLUDE_GRADED_DEFAULT
):
    """
    Market context (ACTIVE + GRADED comps) — SOLD history intentionally disabled.

    Why:
      - SOLD endpoints were not returning consistent results in production.
      - We keep the UX clean (no empty "Sold (historic)" tiles).
      - We can re-introduce sold history later once a reliable source is confirmed.

    Output:
      - active: top-5 closest ACTIVE comps + low/median/high
      - graded: grade buckets (10/9/8) from graded ACTIVE listings (brand-agnostic)
      - grade_outcome: expected value if it lands the assessed grade (uses graded bucket median; falls back to active median)
      - market_summary: a richer, Pokémon-enthusiast paragraph with buy-target guidance
    """

    # --------------------------
    # Identity resolution
    # --------------------------
    n = _norm_ws(item_name or card_name or name or "").strip()
    sname = _norm_ws(item_set or card_set or "").strip()
    num_display = _clean_card_number_display(item_number or card_number or "").strip()

    ctype = _normalize_card_type(_norm_ws(card_type or item_category or "").strip())

    # Memorabilia/sealed fallback query
    if _is_blankish(n):
        n = _norm_ws(description or query or "").strip()

    if not n:
        return {
            "available": False,
            "message": "No item details available yet. Please run Identify/Verify first.",
            "used_query": "",
            "query_ladder": [],
        }

    # --------------------------
    # Simple numeric grade parsing
    # --------------------------
    def _as_int_grade(s: Optional[str]) -> Optional[int]:
        if s is None:
            return None
        ss = str(s).strip()
        if not ss:
            return None
        try:
            return int(round(float(ss)))
        except Exception:
            return None

    g_ass = _as_int_grade(assessed_pregrade)
    g_pred = _as_int_grade(predicted_grade)
    resolved_grade = g_ass if g_ass is not None else g_pred
    structural = bool(has_structural_damage)

    # --------------------------
    # Query ladder (Google-like)
    # --------------------------
    set_code_hint = _norm_ws((item_set or "")).strip()
    ladder = _build_ebay_query_ladder(n, sname, set_code_hint, num_display, ctype)
    ladder = ladder[:12] if ladder else [_norm_ws(n)]

    exclude_graded_effective = EXCLUDE_GRADED_DEFAULT if exclude_graded is None else bool(exclude_graded)

    # --------------------------
    # eBay active fetcher (Browse API)
    # --------------------------
    async def _price_to_aud(amount: Any, currency: str) -> Optional[float]:
        try:
            v = float(amount)
        except Exception:
            return None
        cur = (currency or "").upper().strip() or DEFAULT_CURRENCY
        if cur == "AUD":
            return round(v, 2)
        if cur == "USD":
            rate = await _fx_usd_to_aud()
            return round(v * float(rate), 2)
        return round(v, 2)

    ebay_call_debug = {
        "active": {"calls": 0, "non200": [], "token": {}, "marketplace": EBAY_MARKETPLACE_ID},
    }

    async def _browse_active(query_str: str, limit: int = 50) -> List[Dict[str, Any]]:
        qs = _norm_ws(query_str)
        if not qs or not USE_EBAY_API:
            return []

        url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
        params = {"q": qs, "limit": str(max(1, min(50, int(limit or 50))))}

        token, tok_dbg = await _get_ebay_app_token(force_refresh=False)
        ebay_call_debug["active"]["token"] = tok_dbg
        if not token:
            return []

        async def _do_call(tok: str) -> Tuple[int, Optional[Dict[str, Any]], str]:
            headers = {
                "Authorization": f"Bearer {tok}",
                "X-EBAY-C-MARKETPLACE-ID": EBAY_MARKETPLACE_ID,
                "User-Agent": UA,
            }
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    r = await client.get(url, params=params, headers=headers)
                    txt = (r.text or "")
                    if r.status_code != 200:
                        return r.status_code, None, txt[:400]
                    return r.status_code, r.json(), ""
            except Exception as e:
                return 0, None, str(e)[:400]

        ebay_call_debug["active"]["calls"] += 1
        status, j, errtxt = await _do_call(token)
        if status in (401, 403):
            token2, tok_dbg2 = await _get_ebay_app_token(force_refresh=True)
            ebay_call_debug["active"]["token"] = tok_dbg2
            if token2:
                ebay_call_debug["active"]["calls"] += 1
                status, j, errtxt = await _do_call(token2)

        if status != 200 or not isinstance(j, dict):
            ebay_call_debug["active"]["non200"].append({"status": status, "err": errtxt, "q": qs})
            return []

        out: List[Dict[str, Any]] = []
        for it in (j.get("itemSummaries") or []):
            try:
                item_id = str(it.get("itemId") or "")
                title = str(it.get("title") or "")
                web_url = str(it.get("itemWebUrl") or "")
                price = (it.get("price") or {})
                pv = price.get("value")
                pc = price.get("currency")
                cond = str(it.get("condition") or it.get("conditionId") or "")
                aud = await _price_to_aud(pv, pc)
                if aud is None or aud <= 0:
                    continue
                out.append({
                    "itemId": item_id,
                    "title": title,
                    "url": web_url,
                    "condition": cond,
                    "price_aud": aud,
                    "currency": "AUD",
                    "raw_currency": pc,
                })
            except Exception:
                continue
        return out

    # --------------------------
    # Candidate pooling + scoring
    # --------------------------
    NEGATIVE_TERMS = [
        "proxy", "reprint", "custom", "fake", "fan made", "fanmade", "replica",
        "digital", "print", "download", "poster", "art",
    ]
    LOT_TERMS = ["lot", "bundle", "collection", "job lot", "joblot", "bulk", "assorted"]

    def _tokenize(s: str) -> List[str]:
        s = re.sub(r"[^a-z0-9/#]+", " ", (s or "").lower())
        return [t for t in s.split() if t and len(t) > 1]

    expected = {
        "name": n,
        "name_tokens": set(_tokenize(n)),
        "set": sname,
        "set_tokens": set(_tokenize(sname)),
        "num": num_display,
        "num_variants": _normalize_num_variants(num_display),
        "type": ctype,
    }

    # strict Pokemon card number gate (optional)
    expected_num_pair = None
    mnum = re.search(r"(\d+)\s*/\s*(\d+)", num_display)
    if mnum:
        try:
            expected_num_pair = (int(mnum.group(1)), int(mnum.group(2)))
        except Exception:
            expected_num_pair = None
    strict_number = bool(STRICT_CARD_NUMBER and ctype == "Pokemon" and expected_num_pair is not None)

    GRADED_TERMS = ["psa", "bgs", "cgc", "sgc", "beckett", "graded", "slab", "gem mint", "mint 10", "gm 10"]

    def _contains_any(hay: str, needles: List[str]) -> bool:
        h = (hay or "").lower()
        return any(x in h for x in needles)

    def _score_candidate(title: str, allow_graded: bool = False) -> Tuple[int, Dict[str, Any]]:
        t = (title or "").lower()
        toks = set(_tokenize(t))
        score = 0
        dbg = {"pos": [], "neg": []}

        if _contains_any(t, NEGATIVE_TERMS):
            score -= 80
            dbg["neg"].append("negative_term")
        if _contains_any(t, LOT_TERMS):
            score -= 25
            dbg["neg"].append("lot_bundle")

        is_graded = _contains_any(t, GRADED_TERMS)
        if is_graded and not allow_graded:
            score -= 35
            dbg["neg"].append("graded_listing")

        # Name tokens
        name_hits = len(expected["name_tokens"].intersection(toks))
        if name_hits:
            score += min(35, 10 + name_hits * 5)
            dbg["pos"].append(f"name_tokens:{name_hits}")

        # Set tokens
        set_hits = len(expected["set_tokens"].intersection(toks)) if expected["set_tokens"] else 0
        if set_hits:
            score += min(22, 8 + set_hits * 4)
            dbg["pos"].append(f"set_tokens:{set_hits}")

        # Card number strictness for Pokémon singles
        if strict_number:
            cand_pairs = re.findall(r"(\d{1,4})\s*/\s*(\d{1,4})", t)
            if cand_pairs:
                try:
                    ok = any((int(a) == expected_num_pair[0] and int(b) == expected_num_pair[1]) for (a, b) in cand_pairs)
                except Exception:
                    ok = False
                if ok:
                    score += 40
                    dbg["pos"].append("number_exact")
                else:
                    score -= 120
                    dbg["neg"].append("number_mismatch")
            else:
                score -= 20
                dbg["neg"].append("number_missing")
        else:
            # soft variants
            for nv in (expected.get("num_variants") or []):
                if nv and nv.lower() in t:
                    score += 20
                    dbg["pos"].append("number_match")
                    break

        return score, dbg

    def _collect_top5(candidates: List[Dict[str, Any]], allow_graded: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        for c in candidates:
            title = str(c.get("title") or "")
            sc, dbg = _score_candidate(title, allow_graded=allow_graded)

            if exclude_graded_effective and (not allow_graded) and ("graded_listing" in dbg.get("neg", [])):
                rejected.append({"title": title[:120], "reason": "graded_excluded", "score": sc})
                continue
            if strict_number and ("number_mismatch" in dbg.get("neg", [])):
                rejected.append({"title": title[:120], "reason": "number_mismatch", "score": sc})
                continue
            if sc < 0:
                rejected.append({"title": title[:120], "reason": "low_score", "score": sc})
                continue

            scored.append({
                "title": title,
                "price": float(c.get("price_aud") or 0),
                "price_aud": float(c.get("price_aud") or 0),
                "url": c.get("url") or "",
                "condition": c.get("condition") or "",
                "score": sc,
                "_dbg": dbg,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:5]

        prices = [x["price"] for x in top if isinstance(x.get("price"), (int, float)) and x["price"] > 0]
        prices_sorted = sorted(prices)
        if prices_sorted:
            low = prices_sorted[0]
            med = float(statistics.median(prices_sorted))
            high = prices_sorted[-1]
        else:
            low = med = high = None

        stats = {
            "count": len(top),
            "low": round(low, 2) if low is not None else None,
            "median": round(med, 2) if med is not None else None,
            "high": round(high, 2) if high is not None else None,
            "currency": "AUD",
        }

        debug = {"rejected_examples": rejected[:12]}
        for x in top:
            x.pop("_dbg", None)
        return top, stats, debug

    # --------------------------
    # Build ACTIVE pool + select best query
    # --------------------------
    used_query = ladder[0] if ladder else _norm_ws(n)
    active_pool: List[Dict[str, Any]] = []
    best_matches: List[Dict[str, Any]] = []
    best_stats: Dict[str, Any] = {}
    best_debug: Dict[str, Any] = {}

    for q in ladder:
        cand = await _browse_active(q, limit=50)
        if not cand:
            continue
        # merge pool
        active_pool.extend(cand)
        matches, stats, dbg = _collect_top5(cand, allow_graded=False)
        if stats and stats.get("median") is not None and len(matches) >= 3:
            used_query = q
            best_matches, best_stats, best_debug = matches, stats, dbg
            break
        # keep first non-empty as fallback
        if not best_matches:
            used_query = q
            best_matches, best_stats, best_debug = matches, stats, dbg

    active_matches = best_matches
    active_stats = best_stats or {"count": 0, "low": None, "median": None, "high": None, "currency": "AUD"}
    active_debug = best_debug or {}

    # --------------------------
    # Graded buckets (ACTIVE only; brand-agnostic)
    # --------------------------
    GRADE_RX = re.compile(r'(?:\bpsa\b|\bbgs\b|\bcgc\b|\bsgc\b|\bgraded\b|\bslab\b|\bgem\s*mint\b|\bpristine\b)?\s*([0-9]{1,2}(?:\.[05])?)\b', re.IGNORECASE)
    PSA10_RX = re.compile(r'\bpsa\s*10\b|\bpsa10\b|\bgem\s*mint\s*10\b|\b10\s*gem\b', re.IGNORECASE)

    def _parse_grade_from_title(title: str) -> Optional[float]:
        t = (title or "").lower()
        if PSA10_RX.search(t):
            return 10.0
        m = GRADE_RX.search(title or "")
        if not m:
            return None
        try:
            g = float(m.group(1))
            if g < 1 or g > 10:
                return None
            return g
        except Exception:
            return None

    def _bucket_key(g: Optional[float]) -> Optional[str]:
        if g is None:
            return None
        if g >= 9.5:
            return "10"
        if g >= 9.0:
            return "9"
        if g >= 8.0:
            return "8"
        return None

    # Pull a separate graded pool (do NOT exclude graded)
    graded_query_base = " ".join([x for x in [n, sname] if x]).strip()
    graded_query = _norm_ws(f"{graded_query_base} graded psa cgc bgs")
    graded_pool = await _browse_active(graded_query, limit=50)

    graded_buckets = {"10": [], "9": [], "8": []}
    for it in graded_pool:
        g = _parse_grade_from_title(str(it.get("title") or ""))
        bk = _bucket_key(g)
        if not bk:
            continue
        graded_buckets[bk].append(it)

    def _bucket_stats(pool: List[Dict[str, Any]]) -> Dict[str, Any]:
        prices = [float(x.get("price_aud") or 0) for x in pool if float(x.get("price_aud") or 0) > 0]
        prices_sorted = sorted(prices)
        if not prices_sorted:
            return {"count": 0, "low": None, "median": None, "high": None, "currency": "AUD"}
        low = prices_sorted[0]
        med = float(statistics.median(prices_sorted))
        high = prices_sorted[-1]
        return {"count": len(prices_sorted), "low": round(low, 2), "median": round(med, 2), "high": round(high, 2), "currency": "AUD"}

    grade_market: Dict[str, Any] = {}
    for k in ["10", "9", "8"]:
        st = _bucket_stats(graded_buckets[k])
        # Only include buckets with meaningful data
        grade_market[k] = {
            "stats": st,
            "matches": [{"title": str(x.get("title") or ""), "price_aud": float(x.get("price_aud") or 0), "url": x.get("url") or ""} for x in graded_buckets[k][:5]],
            "source": "active",
        }

    # --------------------------
    # Expected value + buy target (ACTIVE + GRADED)
    # --------------------------
    exp_grade_key = str(int(resolved_grade)) if isinstance(resolved_grade, int) else (str(resolved_grade) if resolved_grade is not None else "")
    expected_value_aud = None
    expected_source = "active"

    if exp_grade_key in grade_market and grade_market[exp_grade_key]["stats"].get("median") is not None:
        expected_value_aud = float(grade_market[exp_grade_key]["stats"]["median"])
        expected_source = "graded_active"
    elif active_stats.get("median") is not None:
        expected_value_aud = float(active_stats["median"])
        expected_source = "active"

    # Buy target heuristic (conservative, depends on grade + confidence)
    conf = float(confidence or 0.0)
    conf = max(0.0, min(1.0, conf))
    grade_tier = resolved_grade if isinstance(resolved_grade, int) else None
    base_mult = 0.78 + 0.07 * conf  # 0.78..0.85
    if grade_tier is not None:
        if grade_tier >= 9:
            base_mult = 0.82 + 0.06 * conf
        elif grade_tier >= 7:
            base_mult = 0.78 + 0.06 * conf
        else:
            base_mult = 0.70 + 0.06 * conf

    buy_target_aud = None
    if expected_value_aud is not None:
        buy_target_aud = round(expected_value_aud * base_mult, 2)

    grade_outcome = {
        "assessed_grade": exp_grade_key or None,
        "expected_value_aud": expected_value_aud,
        "expected_source": expected_source,
        "buy_target_aud": buy_target_aud,
    }

    # --------------------------
    # Build an enthusiast-style market summary (no fake sold-history claims)
    # --------------------------
    def _money(v: Optional[float]) -> str:
        try:
            if v is None:
                return "—"
            return f"${float(v):.0f}"
        except Exception:
            return "—"

    current_typ = active_stats.get("median")
    current_low = active_stats.get("low")
    current_high = active_stats.get("high")

    # trend proxy (active-only; honest language)
    trend_hint = "pretty steady right now"
    if current_low is not None and current_high is not None and current_typ is not None:
        spread = (float(current_high) - float(current_low)) / max(1.0, float(current_typ))
        if spread > 0.6:
            trend_hint = "kinda all over the shop (wide spread on asks)"
        elif spread < 0.25:
            trend_hint = "tight and steady (asks are clustering)"
    # grading potential language
    worth_grading = None
    if expected_value_aud is not None and current_typ is not None:
        try:
            uplift = float(expected_value_aud) - float(current_typ)
            worth_grading = (uplift > float(grading_cost or 0))
        except Exception:
            worth_grading = None

    slang_openers = [
        "Alright mate, here’s the vibe on this one:",
        "Okay, Pokéfam — quick market check:",
        "Righto, let’s suss the market real quick:",
    ]
    opener = slang_openers[0]

    grade_line = ""
    if exp_grade_key:
        grade_line = f" LeagAI’s calling it around a {exp_grade_key} in the current state."
    price_line = f" On current listings, it’s sitting around {_money(current_typ)} AUD (rough range {_money(current_low)}–{_money(current_high)})."
    trend_line = f" The market looks {trend_hint} based on live asks."
    graded_line = ""
    if exp_grade_key and exp_grade_key in grade_market:
        st = grade_market.get(exp_grade_key, {}).get("stats", {})
        if st and st.get("median") is not None:
            graded_line = f" If it lands that grade, you’re looking at roughly {_money(st.get('median'))} AUD in the graded lane (based on current graded listings)."
    grade_advice = ""
    if worth_grading is True:
        grade_advice = f" With grading cost around ${float(grading_cost or 0):.0f}, it looks worth a crack if your goal is long-term hold or a clean flip."
    elif worth_grading is False:
        grade_advice = f" With grading cost around ${float(grading_cost or 0):.0f}, it’s probably not worth sending in this condition unless you’re chasing the slab for the collection."
    buy_line = ""
    if buy_target_aud is not None:
        buy_line = f" If you’re buying, a good target entry is about {_money(buy_target_aud)} AUD or less for this condition — that’s where the risk/reward starts to feel spicy."

    market_summary = _norm_ws(f"{opener}{grade_line}{price_line} {trend_line}{graded_line} {grade_advice} {buy_line}")

    resp = {
        "available": True,
        "source": "ebay_active",
        "used_query": used_query,
        "query_ladder": ladder,
        "card": {"name": n, "set": sname, "number": num_display, "set_code": set_code_hint, "type": ctype},
        "has_structural_damage": structural,
        "assessed_pregrade": resolved_grade,
        "market_summary": market_summary,
        "grade_outcome": grade_outcome,
        "graded": grade_market,
        "active": active_stats,
        "active_matches": active_matches,
        "match_debug": {"active": active_debug, "ebay_calls": ebay_call_debug},
        "observed": {
            "currency": "AUD",
            "fx": {"usd_to_aud_rate": _FX_CACHE.get("usd_aud"), "cache_seconds": FX_CACHE_SECONDS},
            "active": active_stats,
            "raw": {  # legacy: point raw to ACTIVE stream now
                "source": "ebay_active_top5",
                "count": active_stats.get("count"),
                "low": active_stats.get("low"),
                "median": active_stats.get("median"),
                "high": active_stats.get("high"),
            },
        },
        "disclaimer": (
            "Informational market context only. ACTIVE listings are current asks and can differ from final sale prices. "
            "Figures are third-party estimates and may vary. Not financial advice."
        ),
    }
    return resp

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


def _normalize_num_variants(card_number: str) -> List[str]:
    """
    Return card number variants commonly used in listings:
      "004/102" -> ["4/102","4","#4"]
      "4/102"   -> ["4/102","4","#4"]
      "4"       -> ["4","#4"]
    """
    s = _norm_ws(str(card_number or ""))
    if not s:
        return []
    m = re.search(r"(\d+)\s*/\s*(\d+)", s)
    out = []
    if m:
        left = str(int(m.group(1)))
        right = str(int(m.group(2)))
        out.append(f"{left}/{right}")
        out.append(left)
        out.append(f"#{left}")
        return out
    m2 = re.search(r"(\d+)", s)
    if m2:
        left = str(int(m2.group(1)))
        return [left, f"#{left}"]
    return [s]

def _dedupe_tokens(q: str) -> str:
    """Remove immediate duplicate phrases like 'Base Set Base Set' and collapse whitespace."""
    q = _norm_ws(q)
    # remove duplicated bigrams/phrases conservatively
    q = re.sub(r"\b(Base Set)\s+\1\b", r"\1", q, flags=re.I)
    q = re.sub(r"\b(Shadowless)\s+\1\b", r"\1", q, flags=re.I)
    q = re.sub(r"\b(1st Edition)\s+\1\b", r"\1", q, flags=re.I)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def _build_ebay_query_ladder(card_name: str, set_name: str, set_code: str, card_number: str, card_type: str) -> List[str]:
    """
    Build a resilient eBay keyword ladder. Start specific, progressively relax:
      1) name + set + number (number variants)
      2) name + set (and 1st edition cues if present)
      3) name + number
      4) name + tcg + 'card'
      5) name only
    """
    name = _norm_ws(card_name)
    sname = _norm_ws(set_name)
    scode = _norm_ws(set_code)
    ctype = (_norm_ws(card_type) or "Pokemon")
    nums = _normalize_num_variants(card_number)

    # detect common variant cues in name/set
    is_first = bool(re.search(r"\b1st\b|\bfirst\b", name, flags=re.I))
    first_phrase = "1st Edition" if is_first else ""

    # prefer set_name; set_code is usually not in listings, but keep as fallback rung
    set_terms = [t for t in [sname, scode] if t]
    ladder = []

    # rung 1: name + set + number variants
    for st in set_terms[:1] or [""]:
        for nv in nums[:2] or [""]:
            if st and nv:
                ladder.append(_dedupe_tokens(f"{name} {st} {nv}"))
                if first_phrase:
                    ladder.append(_dedupe_tokens(f"{name} {first_phrase} {st} {nv}"))

    # rung 2: name + set
    for st in set_terms[:1] or [""]:
        if st:
            ladder.append(_dedupe_tokens(f"{name} {st}"))
            if first_phrase:
                ladder.append(_dedupe_tokens(f"{name} {first_phrase} {st}"))
        if st and "Base Set" in st and first_phrase:
            ladder.append(_dedupe_tokens(f"{name} {first_phrase} Base Set"))

    # rung 3: name + number only (good when set noise hurts)
    for nv in nums[:2]:
        ladder.append(_dedupe_tokens(f"{name} {nv}"))

    # rung 4: name + tcg card
    ladder.append(_dedupe_tokens(f"{name} {ctype} card"))

    # rung 5: name only
    ladder.append(_dedupe_tokens(f"{name}"))

    # remove empties + duplicates preserving order
    seen = set()
    out = []
    for q in ladder:
        q = _norm_ws(q)
        if not q:
            continue
        if q.lower() in seen:
            continue
        seen.add(q.lower())
        out.append(q)
    return out
