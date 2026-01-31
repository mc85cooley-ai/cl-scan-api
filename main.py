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
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}

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
    year = _norm_ws(str(data.get("year", "")))
    card_number_display = _clean_card_number_display(str(data.get("card_number", "")))
    set_code = _norm_ws(str(data.get("set_code", ""))).upper()
    set_name = _norm_ws(str(data.get("set_name", "")))
    conf = _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0)
    notes = _norm_ws(str(data.get("notes", "")))

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

    return JSONResponse(content={
        "card_name": card_name,
        "card_type": card_type,
        "year": year,
        "card_number": card_number_display,
        "set_code": set_code,
        "set_name": set_name,
        "set_id": set_id,
        "confidence": conf,
        "notes": notes,
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
    provided_num_display = _clean_card_number_display(card_number or "")
    provided_year = _norm_ws(card_year or "")
    provided_type = _norm_ws(card_type or "")
    provided_code = _norm_ws(set_code or "").upper()

    # Backfill set name if needed
    if POKEMONTCG_API_KEY and provided_code and not provided_set:
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

    prompt = f"""You are a professional trading card grader.

Analyze BOTH images (front + back) at high scrutiny. Your job is to surface *everything visible* — do NOT be polite or optimistic.

CRITICAL RULES (follow exactly):
1) **Do not miss obvious damage.** If you see any of these, you MUST mention them explicitly with SIDE + precise location:
   - bends / creases / folds / dents / dings / impressions
   - peeling / lifted foil / surface gouges
   - heavy whitening, corner rounding, edge chipping, tears
   - stains / marks / writing / residue
   - moisture warp / ripples / gloss loss / scuffs
2) **Grade must reflect the worst visible defect.** Use conservative PSA-style logic:
   - Any clear crease/fold, strong bend, tear, or major dent → pregrade MUST be **4 or lower**.
   - Any visible bend/ding/impression, heavy corner rounding, or multiple heavy edge chips → pregrade MUST be **5 or lower**.
   - Moderate whitening/edge wear across multiple edges/corners → pregrade typically **6–7** (not higher).
   - Only call corners "sharp" if they are truly sharp with no rounding, whitening, or bend.
3) **Be explicit and specific.** Write defects as clear sentences with side + location + severity (minor/moderate/severe).
4) If something cannot be confirmed due to blur/glare, state it in notes and lower confidence.

{context}

Return ONLY valid JSON with this EXACT structure:

{{
  "pregrade": "estimated PSA-style grade 1-10 (e.g. 8, 9, 10)",
  "confidence": 0.0-1.0,
  "centering": {{
    "front": {{"grade":"55/45","notes":"..."}} ,
    "back":  {{"grade":"60/40","notes":"..."}}
  }},
  "corners": {{
    "front": {{
      "top_left": {{"condition":"sharp/minor_whitening/whitening/bend/ding/crease","notes":"..."}}, 
      "top_right": {{"condition":"sharp/minor_whitening/whitening/bend/ding/crease","notes":"..."}}, 
      "bottom_left": {{"condition":"sharp/minor_whitening/whitening/bend/ding/crease","notes":"..."}}, 
      "bottom_right": {{"condition":"sharp/minor_whitening/whitening/bend/ding/crease","notes":"..."}}
    }},
    "back": {{
      "top_left": {{"condition":"sharp/minor_whitening/whitening/bend/ding/crease","notes":"..."}}, 
      "top_right": {{"condition":"sharp/minor_whitening/whitening/bend/ding/crease","notes":"..."}}, 
      "bottom_left": {{"condition":"sharp/minor_whitening/whitening/bend/ding/crease","notes":"..."}}, 
      "bottom_right": {{"condition":"sharp/minor_whitening/whitening/bend/ding/crease","notes":"..."}}
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
    "Each defect as a clear sentence with SIDE and location + severity (minor/moderate/severe)"
  ],
  "flags": [
    "Short bullet flags for important issues (crease, bend, dent, heavy whitening, edge chipping, peeling, stain, writing, warp)"
  ],
  "assessment_summary": "2-4 sentence narrative. Mention front vs back differences, the worst defect(s), and why they cap the grade.",
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
        "observed_id": data.get("observed_id", {}) if isinstance(data.get("observed_id", {}), dict) else {},
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

    prompt = """You are identifying a collectible item (memorabilia or sealed product) from photos.

Return ONLY valid JSON with these exact fields:

{
  "item_type": "sealed booster box/sealed pack/autographed memorabilia/game-used memorabilia/graded item/other",
  "description": "detailed description of the item (brand, set/series, year if visible)",
  "signatures": "names of any visible signatures or 'None visible'",
  "seal_condition": "Factory Sealed/Opened/Resealed/Damaged/Not applicable",
  "authenticity_notes": "any authenticity indicators visible (holograms, stickers, certificates) and red flags",
  "notable_features": "unique features worth noting",
  "confidence": 0.0-1.0,
  "category_hint": "Pokemon/Magic/YuGiOh/Sports/OnePiece/Other"
}

Rules:
- Be specific about condition and authenticity markers.
- Identify any COA/authentication labels if visible.
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

Analyze ALL images. Return ONLY valid JSON.

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
  "value_factors": ["..."],
  "defects": ["..."],
  "overall_assessment": "2-4 sentence narrative about condition and key factors",
  "flags": ["Important issues (damage, authenticity concerns, etc.)"]
}}

{ctx}

Important:
- If something doesn't apply, use "Not Applicable" or empty arrays.
Respond ONLY with JSON.
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

    # Back-compat aliases from older frontends
    card_name: Optional[str] = Form(None),
    card_type: Optional[str] = Form(None),
    card_set: Optional[str] = Form(None),
    card_number: Optional[str] = Form(None),

    # Extra loose aliases (some JS sends these)
    name: Optional[str] = Form(None),
    query: Optional[str] = Form(None),
):

    # Accept multiple field names from different frontend builds
    name_in = item_name or card_name or name or query or ""
    cat_in = item_category or card_type or "Other"
    set_in = item_set or card_set or ""
    num_in = item_number or card_number or ""

    clean_name = _norm_ws(str(name_in))
    clean_cat = _norm_ws(str(cat_in))
    clean_set = _norm_ws(str(set_in))
    clean_num_display = _clean_card_number_display(str(num_in))

    if not clean_name:
        # Never 422 — return a helpful payload the UI can show
        return JSONResponse(content={
            "available": False,
            "mode": "click_only",
            "message": "Missing item name. Please re-run identification or pass item_name (or card_name).",
            "received": {
                "item_name": item_name,
                "card_name": card_name,
                "name": name,
                "query": query,
                "item_set": item_set,
                "card_set": card_set,
                "item_number": item_number,
                "card_number": card_number,
                "item_category": item_category,
                "card_type": card_type,
            },
            "disclaimer": "Informational market context only. Not financial advice."
        }, status_code=200)

    # Grade model (for UI only)
    conf = _clamp(_safe_float(confidence, 0.0), 0.0, 1.0)
    g = _grade_bucket(predicted_grade) or 9
    dist = _grade_distribution(g, conf)

    sources: List[str] = []
    matched_products: List[Dict[str, Any]] = []

    # --- PriceCharting search ---
    pc_prices: Dict[str, Any] = {}
    pc_best: Dict[str, Any] = {}
    pc_url = ""
    if PRICECHARTING_TOKEN and clean_name:
        q_bits = [clean_name]
        if clean_set and clean_set.lower() not in clean_name.lower():
            q_bits.append(clean_set)
        if clean_num_display:
            q_bits.append(clean_num_display)
        q = " ".join([b for b in q_bits if b]).strip()

        # Category hint: PriceCharting uses category slugs; we keep it optional
        products = await _pc_search(q, category=None, limit=8)
        if products:
            sources.append("PriceCharting API")
            # Try to select the most relevant:
            # Prefer exact-ish name match, else first
            def _score(p: Dict[str, Any]) -> int:
                name = (p.get("product-name") or p.get("product_name") or p.get("name") or "").lower()
                sc = 0
                if clean_name.lower() in name:
                    sc += 3
                if clean_set and clean_set.lower() in name:
                    sc += 2
                if clean_num_display and clean_num_display.replace("#","").lower() in name:
                    sc += 2
                return sc

            products_sorted = sorted(products, key=_score, reverse=True)
            top = products_sorted[0]
            pid = str(top.get("id") or top.get("product-id") or "").strip()
            pc_best = top
            matched_products = products_sorted[:5]
            if pid:
                detail = await _pc_product(pid)
                pc_url = detail.get("url") or top.get("url") or ""
                pc_prices = _pc_extract_price_fields(detail or {})
            else:
                pc_prices = _pc_extract_price_fields(top)

    # Build useful "raw" vs graded extracts (best effort)
    raw_candidates: List[float] = []
    graded10: Optional[float] = None
    graded9: Optional[float] = None
    graded8: Optional[float] = None

    # Known-ish keys seen on PriceCharting vary by category; we grab what exists.
    for k, v in pc_prices.items():
        if not isinstance(v, (int, float)):
            continue
        lk = k.lower()
        # raw-ish
        if any(t in lk for t in ["ungraded", "loose", "new", "raw", "complete", "cib"]):
            raw_candidates.append(float(v))
        # graded
        if "psa-10" in lk or "grade-10" in lk or "10-price" in lk:
            graded10 = float(v)
        if "psa-9" in lk or "grade-9" in lk or "9-price" in lk:
            graded9 = float(v)
        if "psa-8" in lk or "grade-8" in lk or "8-price" in lk:
            graded8 = float(v)

    raw_value = round(mean(raw_candidates), 2) if raw_candidates else None

    available = bool(pc_prices)
    if not available:
        return JSONResponse(content={
            "available": False,
            "mode": "click_only",
            "item": {"name": clean_name, "category": clean_cat, "set": clean_set, "number": clean_num_display},
            "message": "No market data available (PriceCharting token missing or no match).",
            "sources": sources,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "disclaimer": "Informational market context only. Not financial advice.",
        })

    # Gentle recommendation: only if graded delta exists (no ROI/profit language)
    recommendation = "consider"
    if graded10 and raw_value and graded10 >= raw_value * 2.0:
        recommendation = "recommended"
    if graded10 and raw_value and graded10 <= raw_value * 1.2:
        recommendation = "not_recommended"

    return JSONResponse(content={
        "available": True,
        "mode": "click_only",
        "item": {"name": clean_name, "category": clean_cat, "set": clean_set, "number": clean_num_display},
        "pricecharting": {
            "best_match": pc_best,
            "url": pc_url,
            "prices": pc_prices,
            "alternatives": matched_products,
            "note": "Prices are current estimates from PriceCharting (not historical).",
        },
        "observed": {
            "currency": currency,
            "raw_value_estimate": raw_value,
            "graded": {"psa_8": graded8, "psa_9": graded9, "psa_10": graded10},
        },
        "grade_impact": {
            "predicted_grade": str(g),
            "confidence_input": conf,
            "grade_distribution": dist,
        },
        "grading_recommendation": {
            "label": recommendation,
            "note": "Context only: compares raw vs available graded anchors. Final decision depends on fees, risks, and grading outcomes.",
        },
        "sources": sources,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "disclaimer": "Informational market context only. Figures are third-party estimates and may vary. Not financial advice.",
    })

# ==============================
# Market Context (Click-only) — Memorabilia / Sealed
# Uses PriceCharting search across categories; returns best-match price fields.
# ==============================
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
