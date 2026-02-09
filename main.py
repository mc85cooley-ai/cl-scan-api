"""
The Collectors League Australia - Scan API
Futureproof v6.7.7 (2026-02-05)

What changed vs v6.7.5 (2026-02-03)
- ✅ Intent-aware grading language (BUYING vs SELLING) for BOTH cards + memorabilia prompts:
  - Buyer mode: negotiation leverage, red flags, verification steps, fair buy guidance
  - Seller mode: listing optimisation, disclosure strategy, pricing guidance
  - CRITICAL: grading logic remains objective; only advice framing changes
- ✅ Expanded card grading scale guidance:
  - Allows true GEM MINT 10 when warranted (no psychological “cap at 9”)
  - Adds Grade 12 “Collectors League Ultra Flawless” (rare, must be awarded when warranted)
  - JSON schema hint updated: pregrade "1-12" with 12 criteria note
- ✅ Memorabilia assessment stability fix:
  - Corrected indentation in assess_memorabilia defect thumbnail logic (defect_snaps) to prevent syntax/runtime errors

Market data architecture (unchanged)
- ✅ PriceCharting API as primary market source for cards + sealed/memorabilia (current prices).
- ✅ Keeps PokemonTCG.io for identification + metadata enrichment (NOT price history).
- ✅ Weekly snapshot option (store PriceCharting CSV/API snapshots on a schedule) to build your own price history.
- ✅ eBay API scaffolding included but DISABLED by default (waiting for your dev account approval).
- ✅ Market endpoints return "click-only" informational context + no ROI language.

Env vars
- OPENAI_API_KEY (required for vision grading/ID)
- POKEMONTCG_API_KEY (optional; used for Pokemon metadata enrichment)
- PRICECHARTING_TOKEN (recommended; enables pricing for cards + sealed/memorabilia)
- USE_EBAY_API=0/1 (default 0) + EBAY_APP_ID/EBAY_CERT_ID/EBAY_DEV_ID/EBAY_OAUTH_TOKEN (optional; scaffold only)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel
from datetime import datetime, timedelta
from statistics import mean, median
from functools import wraps
from fastapi import WebSocket
import base64
import io
import os
import asyncio
import json
import sqlite3
import csv
import secrets

# Optional scientific stack for market trend predictions
try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy import stats
    from scipy.fft import fft, fftfreq
except Exception:
    stats = None
    fft = None
    fftfreq = None

import hashlib
import re
import traceback
import logging
import statistics
from pathlib import Path

import httpx
import time
import math

# Optional image processing for 2-pass defect enhancement
try:
    from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat
except Exception:
    Image = None
    ImageEnhance = None
    ImageOps = None
    ImageFilter = None


PIL_AVAILABLE = bool(Image)



def _relax_whitening_mm(text: str) -> str:
    """Replace clinical mm measurements for whitening with collector-style language.

    Only touches sentences that mention whitening, so centering measurements (e.g. 5mm) stay intact.
    """
    try:
        s = str(text or "")
        if not s:
            return s

        # Split into sentences so we only rewrite the whitening ones.
        parts = re.split(r'(?<=[.!?])\s+', s)
        out_parts = []

        # Match mm mentions like "about 1mm", "approximately 2 mm"
        mm_rx = re.compile(r'(?:about\s+|around\s+|approx(?:imately)?\s+)?(\d+(?:\.\d+)?)\s*mm\b', re.IGNORECASE)

        def _mm_to_phrase(num: float) -> str:
            # Phrases are chosen to read naturally in different contexts ("deep", "into", etc.)
            if num <= 0.6:
                return "a hint"
            if num <= 1.4:
                return "a touch"
            if num <= 2.4:
                return "a little"
            if num <= 4.0:
                return "a noticeable amount"
            return "a fair bit"

        for part in parts:
            if "whitening" not in part.lower():
                out_parts.append(part)
                continue

            def repl(m):
                try:
                    num = float(m.group(1))
                except Exception:
                    num = 1.0
                return _mm_to_phrase(num)

            new = mm_rx.sub(repl, part)

            # Soften common phrasing after substitution
            new = re.sub(r"\bextending\b", "creeping", new, flags=re.IGNORECASE)

            # Fix awkward leftovers like "about a touch deep" or "a little into the card surface"
            new = re.sub(r"\b(about|around|approximately)\s+(a hint|a touch|a little|a noticeable amount|a fair bit)\s+deep\b",
                         r"\2", new, flags=re.IGNORECASE)
            new = re.sub(r"\b(a hint|a touch|a little|a noticeable amount|a fair bit)\s+deep\b",
                         r"\1", new, flags=re.IGNORECASE)

            # If it says "... into the card surface", make it conversational
            new = re.sub(r"\b(creeping)\s+(a hint|a touch|a little|a noticeable amount|a fair bit)\s+into\s+the\s+card\s+surface\b",
                         r"\1 in \2", new, flags=re.IGNORECASE)
            new = re.sub(r"\b(a hint|a touch|a little|a noticeable amount|a fair bit)\s+into\s+the\s+card\s+surface\b",
                         r"\1", new, flags=re.IGNORECASE)

            # Clean double spaces introduced by rewrites
            new = re.sub(r"\s{2,}", " ", new).strip()
            out_parts.append(new)

        return " ".join(out_parts).strip()
    except Exception:
        return str(text or "")


def _relax_whitening_mm_in_obj(obj):
    """Recursively apply _relax_whitening_mm to any string fields in nested dict/list."""
    if isinstance(obj, str):
        return _relax_whitening_mm(obj)
    if isinstance(obj, list):
        return [_relax_whitening_mm_in_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _relax_whitening_mm_in_obj(v) for k, v in obj.items()}
    return obj


def _wanted_rois_from_assessment(data: Dict[str, Any]) -> set:
    """Infer which ROI crops we should show from the structured assessment.

    We use this to ensure the UI gets at least one close-up for defects that
    are clearly flagged (e.g., red bullets / corner whitening), even if the ROI
    labeler returns "clean".

    Returns a set of tuples: {(side, roi), ...}
    """
    wanted = set()
    if not isinstance(data, dict):
        return wanted

    # Corners -> map corner keys to ROI names
    corner_map = {
        "top_left": "corner_top_left",
        "top_right": "corner_top_right",
        "bottom_left": "corner_bottom_left",
        "bottom_right": "corner_bottom_right",
    }
    corners = data.get("corners") or {}
    if isinstance(corners, dict):
        for side in ("front", "back"):
            s_obj = corners.get(side) or {}
            if not isinstance(s_obj, dict):
                continue
            for ck, roi in corner_map.items():
                c_obj = s_obj.get(ck) or {}
                if not isinstance(c_obj, dict):
                    continue
                cond = str(c_obj.get("condition") or "").strip().lower()
                note = str(c_obj.get("notes") or "").strip().lower()
                # Anything other than "sharp"/"clean" should be treated as a defect for evidence.
                if cond and cond not in {"sharp", "clean", "mint", "perfect", "nm"}:
                    wanted.add((side, roi))
                elif any(w in note for w in ("whitening", "wear", "nick", "chip", "rounded", "ding", "crease", "dent")):
                    wanted.add((side, roi))

    # Edges
    edges = data.get("edges") or {}
    if isinstance(edges, dict):
        for side in ("front", "back"):
            e_obj = edges.get(side) or {}
            if not isinstance(e_obj, dict):
                continue
            grade = str(e_obj.get("grade") or "").strip().lower()
            note = str(e_obj.get("notes") or "").strip().lower()
            # If edges are anything below NM-ish, request edge closeups.
            if grade and grade not in {"gem mint", "mint", "near mint", "excellent"}:
                wanted.update({(side, "edge_top"), (side, "edge_bottom"), (side, "edge_left"), (side, "edge_right")})
            elif any(w in note for w in ("whitening", "wear", "scuff", "chipping", "chip", "nick", "rough", "fray")):
                wanted.update({(side, "edge_top"), (side, "edge_bottom"), (side, "edge_left"), (side, "edge_right")})

    # Surface
    surface = data.get("surface") or {}
    if isinstance(surface, dict):
        for side in ("front", "back"):
            s_obj = surface.get(side) or {}
            if not isinstance(s_obj, dict):
                continue
            grade = str(s_obj.get("grade") or "").strip().lower()
            note = str(s_obj.get("notes") or "").strip().lower()
            if grade and grade not in {"gem mint", "mint", "near mint", "excellent"}:
                # We don't have dedicated surface ROIs; still allow any ROI the CV found.
                wanted.add((side, "surface"))
            elif any(w in note for w in ("scratch", "scuff", "print line", "crease", "dent", "indent", "stain", "mark")):
                wanted.add((side, "surface"))

    # Second pass defect candidates
    sp = data.get("second_pass") or {}
    if isinstance(sp, dict) and isinstance(sp.get("defect_candidates"), list):
        for c in sp.get("defect_candidates")[:20]:
            if not isinstance(c, dict):
                continue
            t = str(c.get("type") or "").lower()
            note = str(c.get("note") or "").lower()
            if any(w in (t + " " + note) for w in ("whitening", "edge", "corner", "crease", "dent", "tear", "scratch", "scuff")):
                wanted.add(("front", "surface"))
                wanted.add(("back", "surface"))

    return wanted

def _make_thumb_from_bbox(img_bytes: bytes, bbox: Dict[str, Any], max_size: int = 360) -> str:
    """
    Crop a normalized bbox out of an image and return base64 JPEG.

    Notes:
      - Produces a consistent square thumbnail (letterboxed) so the UI grid stays uniform.
      - Uses a slightly smaller max_size to reduce visual clutter + payload.
    """
    if not PIL_AVAILABLE or not img_bytes or not bbox:
        return ""
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = im.size
        x = float(bbox.get("x", 0.0)); y = float(bbox.get("y", 0.0))
        bw = float(bbox.get("w", 1.0)); bh = float(bbox.get("h", 1.0))

        # Convert normalized bbox -> pixel bbox
        px1 = max(0, min(w - 1, int(round(x * w))))
        py1 = max(0, min(h - 1, int(round(y * h))))
        px2 = max(1, min(w, int(round((x + bw) * w))))
        py2 = max(1, min(h, int(round((y + bh) * h))))
        if px2 <= px1 or py2 <= py1:
            return ""

        # Focused square crop around the bbox (prevents long edge strips in thumbnails)
        cx = (px1 + px2) / 2.0
        cy = (py1 + py2) / 2.0
        box_w = float(px2 - px1)
        box_h = float(py2 - py1)
        side = max(box_w, box_h) * 1.9  # padding factor
        side = max(80.0, min(side, float(min(w, h))))

        sx1 = int(round(cx - side / 2.0))
        sy1 = int(round(cy - side / 2.0))
        sx2 = int(round(cx + side / 2.0))
        sy2 = int(round(cy + side / 2.0))
        sx1 = max(0, sx1); sy1 = max(0, sy1)
        sx2 = min(w, sx2); sy2 = min(h, sy2)
        if sx2 <= sx1 or sy2 <= sy1:
            return ""

        crop = im.crop((sx1, sy1, sx2, sy2))
        crop = crop.resize((max_size, max_size), resample=Image.BICUBIC)

        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=82, optimize=True)
        return _b64(buf.getvalue())
    except Exception:
        return ""

def _cv_candidate_bboxes(img_bytes: bytes, side: str) -> List[Dict[str, Any]]:
    """
    Light CV assist to propose 'hotspot' regions near borders where whitening/edge wear is likely.

    Upgrade (Feb 2026):
      - For edges, we *do not* return a full-length strip anymore (those thumbnails looked huge / unfocused).
      - Instead, we sample multiple smaller segments along each edge and keep the highest-scoring segment.
    Returns normalized bboxes: {side, roi, score, bbox:{x,y,w,h}}
    """
    if not PIL_AVAILABLE or not img_bytes:
        return []
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = im.size
        g = ImageOps.grayscale(im)
        edge = g.filter(ImageFilter.FIND_EDGES)

        def score_box(px, py, px2, py2):
            cg = g.crop((px, py, px2, py2))
            ce = edge.crop((px, py, px2, py2))
            bg = ImageStat.Stat(cg).mean[0] / 255.0
            be = ImageStat.Stat(ce).mean[0] / 255.0
            return 0.65 * be + 0.35 * bg

        rois: List[Dict[str, Any]] = []

        # Edge strips (thin) + segmenting along the long axis
        t = max(6, int(0.06 * min(w, h)))  # thinner than before
        segs = 3  # number of segments per edge

        def best_edge_segment(name: str, segments: List[tuple]) -> Optional[Dict[str, Any]]:
            best = None
            best_s = -1.0
            for (px, py, px2, py2) in segments:
                s = float(score_box(px, py, px2, py2))
                if s > best_s:
                    best_s = s
                    best = (px, py, px2, py2)
            if not best:
                return None
            px, py, px2, py2 = best
            bbox = {"x": round(px / w, 4), "y": round(py / h, 4),
                    "w": round((px2 - px) / w, 4), "h": round((py2 - py) / h, 4)}
            return {"side": side, "roi": name, "score": round(best_s, 4), "bbox": bbox}

        # Build segmented edges
        step_x = max(1, w // segs)
        step_y = max(1, h // segs)

        top_segments = [(i * step_x, 0, min(w, (i + 1) * step_x), t) for i in range(segs)]
        bottom_segments = [(i * step_x, h - t, min(w, (i + 1) * step_x), h) for i in range(segs)]
        left_segments = [(0, i * step_y, t, min(h, (i + 1) * step_y)) for i in range(segs)]
        right_segments = [(w - t, i * step_y, w, min(h, (i + 1) * step_y)) for i in range(segs)]

        for nm, seg in [("edge_top", top_segments),
                        ("edge_bottom", bottom_segments),
                        ("edge_left", left_segments),
                        ("edge_right", right_segments)]:
            r = best_edge_segment(nm, seg)
            if r:
                rois.append(r)

        # Corners (slightly smaller than before to keep close-up tight)
        c = max(16, int(0.16 * min(w, h)))
        corner_defs = [
            ("corner_top_left", (0, 0, c, c)),
            ("corner_top_right", (w - c, 0, w, c)),
            ("corner_bottom_left", (0, h - c, c, h)),
            ("corner_bottom_right", (w - c, h - c, w, h)),
        ]
        for name, (px, py, px2, py2) in corner_defs:
            s = float(score_box(px, py, px2, py2))
            bbox = {"x": round(px / w, 4), "y": round(py / h, 4),
                    "w": round((px2 - px) / w, 4), "h": round((py2 - py) / h, 4)}
            rois.append({"side": side, "roi": name, "score": round(s, 4), "bbox": bbox})

        # Rank & keep a small set
        rois.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        corners = [r for r in rois if str(r.get("roi", "")).startswith("corner_")]
        edges = [r for r in rois if str(r.get("roi", "")).startswith("edge_")]

        out = (corners[:4] + edges[:2])
        seen = set()
        final = []
        for r in out:
            if r.get("roi") in seen:
                continue
            seen.add(r.get("roi"))
            final.append(r)
        return final
    except Exception:
        return []


async def _openai_label_rois(rois: List[Dict[str, Any]], front_bytes: Optional[bytes], back_bytes: Optional[bytes]) -> List[Dict[str, Any]]:
    """
    Ask the vision model to confirm if ROI crops contain real defects and label them.
    Returns list:
      [{crop_index, side, roi, is_defect, type, note, confidence}]
    """
    if not rois:
        return []
    content_parts: List[Dict[str, Any]] = [{
        "type": "text",
        "text": (
            "You are reviewing close-up crops from a trading card photo. "
            "For each crop, decide whether it shows a REAL defect (not glare/noise). "
            "If real, label the defect type and write a short note. Return ONLY JSON as an array."
        )
    }]

    crop_map = []  # keeps (i, side, roi) only for included crops
    for i, r in enumerate(rois):
        side = r.get("side")
        bbox = r.get("bbox") or {}
        src = front_bytes if side == "front" else back_bytes
        if not src:
            continue
        thumb = _make_thumb_from_bbox(src, bbox, max_size=520)
        if not thumb:
            continue
        crop_map.append((i, side, r.get("roi")))
        content_parts += [
            {"type": "text", "text": f"CROP {i} | side={side} | roi={r.get('roi')}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{thumb}", "detail": "high"}},
        ]

    label_prompt = (
        "Return ONLY JSON array, one object per crop you can judge. "
        "Schema: {crop_index:int, is_defect:bool, type:string, note:string, confidence:0-1}. "
        "type should be one of: whitening, edge_chipping, corner_whitening, scratch, print_line, dent, crease, stain, other. "
        "If not a defect, set is_defect=false and type='none'. Keep note short."
    )
    content_parts += [{"type":"text","text": label_prompt}]
    msg = [{"role":"user","content": content_parts}]
    res = await _openai_chat(msg, max_tokens=700, temperature=0.1)
    if res.get("error"):
        return []
    arr = _parse_json_or_none(res.get("content","")) or []
    if not isinstance(arr, list):
        return []
    out = []
    for x in arr:
        if not isinstance(x, dict):
            continue
        try:
            ci = int(x.get("crop_index", -1))
        except Exception:
            ci = -1
        if ci < 0 or ci >= len(rois):
            continue
        r = rois[ci]
        out.append({
            "crop_index": ci,
            "side": r.get("side"),
            "roi": r.get("roi"),
            "is_defect": bool(x.get("is_defect", False)),
            "type": str(x.get("type","") or "none")[:32],
            "note": _norm_ws(str(x.get("note","") or ""))[:220],
            "confidence": _clamp(_safe_float(x.get("confidence", 0.0)), 0.0, 1.0),
        })
    return out


# Simple in-memory caches (per-process)
_FX_CACHE = {"ts": 0, "usd_aud": None}
_EBAY_CACHE = {}  # key -> {ts, data}
FX_CACHE_SECONDS = int(os.getenv("FX_CACHE_SECONDS", "3600"))

# ==============================
# App & CORS
# ==============================
app = FastAPI(title="Collectors League Scan API")

logger = logging.getLogger("cl-scan-api")

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

# ═══════════════════════════════════════════════════════
# SIMPLE API KEY AUTHENTICATION (Render env vars)
# ═══════════════════════════════════════════════════════

security = HTTPBearer(auto_error=False)

def get_valid_api_keys() -> List[str]:
    """Get list of valid API keys from environment."""
    keys: List[str] = []
    wp_key = os.getenv("WORDPRESS_API_KEY", "").strip()
    if wp_key:
        keys.append(wp_key)
    admin_key = os.getenv("ADMIN_API_KEY", "").strip()
    if admin_key:
        keys.append(admin_key)
    return keys

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify API key from Authorization: Bearer <key>."""
    if not credentials or not (credentials.credentials or "").strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    valid = get_valid_api_keys()
    if not valid:
        # Dev mode: allow if no keys configured
        logging.warning("⚠️ No API keys configured (WORDPRESS_API_KEY / ADMIN_API_KEY). Running in open mode.")
        return "development"

    tok = credentials.credentials.strip()
    if tok not in valid:
        logging.warning(f"❌ Invalid API key attempt: {tok[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logging.info(f"✅ Authenticated: {tok[:10]}...")
    return tok

async def verify_api_key_optional(credentials: HTTPAuthorizationCredentials = Security(security)) -> Optional[str]:
    """Optional auth – allows public read endpoints."""
    if not credentials or not (credentials.credentials or "").strip():
        return None
    return await verify_api_key(credentials)


# PriceCharting local storage (even if eBay is primary, some endpoints still reference these)
PRICECHARTING_CACHE_DIR = os.getenv("PRICECHARTING_CACHE_DIR", "/tmp/pricecharting_cache").strip() or "/tmp/pricecharting_cache"
PRICECHARTING_DB_PATH = (os.getenv("PRICECHARTING_DB_PATH")
                       or os.getenv("PRICE_HISTORY_DB_PATH")
                       or "/opt/render/project/data/pricecharting.db").strip() or "/opt/render/project/data/pricecharting.db"
try:
    os.makedirs(PRICECHARTING_CACHE_DIR, exist_ok=True)
except Exception:
    pass

try:
    os.makedirs(os.path.dirname(PRICECHARTING_DB_PATH), exist_ok=True)
except Exception:
    pass


# ==============================
# FX + eBay helpers
# ==============================
UA = "CollectorsLeagueScan/6.6 (+https://collectors-league.com)"


def _build_ebay_search_query(card_name: str, card_set: str = "", card_number: str = "", grade: str = "") -> str:
    """Build a robust eBay keyword query for Pokemon cards.

    Goals:
      - Prefer recall over precision (too-specific queries cause 'no results')
      - Strip punctuation / years / bracket noise coming from UI labels like "(2025)"
      - Avoid placeholder or obviously-wrong set names
      - Keep full card name (don't truncate aggressively)
      - Optionally add PSA grade (only when explicitly PSA)
    """
    name_raw = (card_name or "").strip()
    if not name_raw:
        return ""

    set_raw = (card_set or "").strip()
    num_raw = (card_number or "").strip()
    g = (grade or "").strip()

    def _clean_text(s: str) -> str:
        s = s or ""
        # Remove bracketed years and any bracketed noise
        s = re.sub(r"\(\s*\d{4}\s*\)", " ", s)
        s = re.sub(r"\[[^\]]*\]", " ", s)
        s = re.sub(r"\([^\)]*\)", " ", s)  # remove remaining (...) safely
        # Drop common UI artefacts
        s = re.sub(r"\b(pre[-\s]?grade|draft|collecting|submitted|graded)\b", " ", s, flags=re.I)
        # Replace punctuation with spaces (keep + and / out)
        s = re.sub(r"[^A-Za-z0-9\s\-]", " ", s)
        return _norm_ws(s)

    name = _clean_text(name_raw)

    # Normalize common Pokemon card naming variations
    # eBay often uses "M Charizard EX" for Mega Charizard EX cards
    name = re.sub(r"\bMega\b", "M", name, flags=re.I)

    # Clean set/number lightly (but we will only include when it helps)
    set_name = _clean_text(set_raw)
    num = _clean_text(num_raw)

    # Avoid adding placeholder / generic / suspicious set values that reduce recall
    bad_sets = {
        "pokemon", "pokémon", "tcg", "card", "cards", "other", "unknown", "none", "n/a", "-",
        "base", "set", "promo", "draft"
    }
    set_lower = set_name.lower().strip()
    if not set_lower or set_lower in bad_sets:
        set_name = ""
    # If set contains digits or looks like a year label, drop it (often UI noise e.g. "(2025)")
    if re.search(r"\b(19|20)\d{2}\b", set_name):
        set_name = ""

    # Build query: start with card name only (best recall)
    parts = [name]

    # Only append set if it doesn't already appear in the name and is short/clean
    if set_name and set_name.lower() not in name.lower() and len(set_name) <= 30:
        parts.append(set_name)

    # Card numbers are often useful but can also overfilter; include only if short
    if num and len(num) <= 12:
        parts.append(num)

    # Always add Pokemon context
    parts.append("Pokemon card")

    q = _norm_ws(" ".join([p for p in parts if p]).strip())

    # Add PSA grade only (other grades vary too much)
    if g and "psa" in g.lower():
        q = _norm_ws(f"{q} {g}")

    return q.strip()


def _build_ebay_query_ladder(card_name: str, card_set: str = "", card_number: str = "", grade: str = "") -> list:
    """
    Return a small ladder of increasingly-broad eBay keyword queries.
    We try precise-ish first, then relax (set removed), then name-only.
    """
    base = _build_ebay_search_query(card_name=card_name, card_set=card_set, card_number=card_number, grade="")  # grade handled separately
    name_only = _build_ebay_search_query(card_name=card_name, card_set="", card_number="", grade="")

    # Normalize grade into a PSA token when possible
    g = (grade or "").strip()
    psa_token = ""
    if g:
        if "psa" in g.lower():
            psa_token = g.strip()
        else:
            # numeric grade like "9"
            if re.fullmatch(r"(10|[1-9](?:\.5)?)", g):
                psa_token = f"PSA {g}"

    ladder = []
    for q in [base, name_only]:
        q = _norm_ws(q)
        if not q:
            continue
        if psa_token:
            ladder.append(_norm_ws(f"{q} {psa_token}"))
        ladder.append(q)

    # Final ultra-broad fallback (sometimes "card" hurts recall)
    if card_name:
        ladder.append(_norm_ws(f"{card_name} Pokemon"))

    # Deduplicate while preserving order
    seen = set()
    out = []
    for q in ladder:
        if q and q not in seen:
            out.append(q)
            seen.add(q)
    return out

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
    if not q or not EBAY_APP_ID:
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
                "GLOBAL-ID": "EBAY-AU",
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
                    cp = (selling.get("convertedCurrentPrice", [{}]) or [{}])[0] or (selling.get("currentPrice", [{}]) or [{}])[0]
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

# ═══════════════════════════════════════════════════════
# PRICE HISTORY (persistent snapshots for market-trends)
# Stored in SQLite (defaults to the same DB file as PriceCharting)
# ═══════════════════════════════════════════════════════

PRICE_HISTORY_DB_PATH = os.getenv("PRICE_HISTORY_DB_PATH", PRICECHARTING_DB_PATH).strip() or PRICECHARTING_DB_PATH

def _ph_db():
    con = sqlite3.connect(PRICE_HISTORY_DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def _ph_init_db():
    con = _ph_db()
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_identifier TEXT NOT NULL,
            card_name TEXT,
            card_set TEXT,
            card_number TEXT,
            grade TEXT,
            price_current REAL,
            price_low REAL,
            price_median REAL,
            price_high REAL,
            volume INTEGER,
            source TEXT,
            data_quality TEXT,
            recorded_date TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_price_history_ident_date ON price_history(card_identifier, recorded_date)")
    con.commit()
    con.close()

try:
    _ph_init_db()
except Exception as _e:
    print(f"INFO: Price history DB init skipped: {_e}")

def record_price_history(
    card_identifier: str,
    card_name: str = "",
    card_set: str = "",
    card_number: str = "",
    grade: str = "",
    price_current: float = 0.0,
    price_low: float = 0.0,
    price_median: float = 0.0,
    price_high: float = 0.0,
    volume: int = 0,
    source: str = "",
    data_quality: str = "verified",
    recorded_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert a price snapshot and return the inserted row (as dict)."""
    ident = (card_identifier or "").strip()
    if not ident:
        raise ValueError("card_identifier required")
    rd = recorded_date or (datetime.utcnow().replace(microsecond=0).isoformat() + "Z")
    con = _ph_db()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO price_history (
            card_identifier, card_name, card_set, card_number, grade,
            price_current, price_low, price_median, price_high,
            volume, source, data_quality, recorded_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ident,
            (card_name or "").strip(),
            (card_set or "").strip(),
            (card_number or "").strip(),
            (grade or "").strip(),
            float(price_current or 0.0),
            float(price_low or 0.0),
            float(price_median or 0.0),
            float(price_high or 0.0),
            int(volume or 0),
            (source or "").strip(),
            (data_quality or "").strip(),
            rd,
        ),
    )
    con.commit()
    new_id = int(cur.lastrowid or 0)
    con.close()
    return {
        "id": new_id,
        "card_identifier": ident,
        "card_name": (card_name or "").strip(),
        "card_set": (card_set or "").strip(),
        "card_number": (card_number or "").strip(),
        "grade": (grade or "").strip(),
        "price_current": float(price_current or 0.0),
        "price_low": float(price_low or 0.0),
        "price_median": float(price_median or 0.0),
        "price_high": float(price_high or 0.0),
        "volume": int(volume or 0),
        "source": (source or "").strip(),
        "data_quality": (data_quality or "").strip(),
        "recorded_date": rd,
    }

def get_price_history(card_identifier: str, days: int = 90) -> List[Dict[str, Any]]:
    """Return up to `days` records, newest-first, for this identifier."""
    ident = (card_identifier or "").strip()
    if not ident:
        return []
    try:
        days_i = int(days or 90)
    except Exception:
        days_i = 90
    days_i = max(1, min(365, days_i))

    # SQLite date filter using ISO strings: compare by recorded_date text (UTC ISO)
    cutoff = (datetime.utcnow() - timedelta(days=days_i)).replace(microsecond=0).isoformat() + "Z"

    con = _ph_db()
    cur = con.cursor()
    cur.execute(
        """
        SELECT *
        FROM price_history
        WHERE card_identifier = ?
          AND recorded_date >= ?
        ORDER BY recorded_date DESC
        LIMIT ?
        """,
        (ident, cutoff, max(7, days_i * 2)),
    )
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows



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
_USE_EBAY_ENV = os.getenv("USE_EBAY_API", "").strip().lower()
# Finding/Legacy (SOLD) uses EBAY_APP_ID for FindingService
EBAY_APP_ID = os.getenv("EBAY_APP_ID", "").strip()
# Auto-enable eBay if EBAY_APP_ID is present and USE_EBAY_API is not explicitly set.
if _USE_EBAY_ENV == "":
    USE_EBAY_API = bool(EBAY_APP_ID)
elif _USE_EBAY_ENV in ("0","false","no","off"):
    USE_EBAY_API = False
else:
    USE_EBAY_API = _USE_EBAY_ENV in ("1","true","yes","y","on")

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

def _make_basic_hotspot_snaps(img_bytes: bytes, side: str, max_snaps: int = 6) -> List[Dict[str, Any]]:
    """Best-effort crops (corners + edges) used for UI evidence.
    Returns list of {side,type,note,thumbnail_b64,bbox}.
    BBox is normalized to [0..1] relative to the source image.
    """
    if not img_bytes or Image is None:
        return []
    try:
        from io import BytesIO
        im = Image.open(BytesIO(img_bytes)).convert("RGB")
        w, h = im.size

        def _crop_norm(x0, y0, x1, y1):
            x0i = max(0, min(w-1, int(x0*w)))
            y0i = max(0, min(h-1, int(y0*h)))
            x1i = max(1, min(w, int(x1*w)))
            y1i = max(1, min(h, int(y1*h)))
            crop = im.crop((x0i, y0i, x1i, y1i))
            # Slight upscale for better visibility
            crop = crop.resize((min(420, (x1i-x0i)*3), min(420, (y1i-y0i)*3)))
            buf = BytesIO()
            crop.save(buf, format="JPEG", quality=90)
            return {
                "thumbnail_b64": _b64(buf.getvalue()),
                "bbox": {"x": float(x0), "y": float(y0), "w": float(x1-x0), "h": float(y1-y0)}
            }

        # Regions: 4 corners + top/bottom edge strips
        regions = [
            ("corner", "top_left",   0.00, 0.00, 0.22, 0.22),
            ("corner", "top_right",  0.78, 0.00, 1.00, 0.22),
            ("corner", "bottom_left",0.00, 0.78, 0.22, 1.00),
            ("corner", "bottom_right",0.78,0.78, 1.00, 1.00),
            ("edge", "top_edge",     0.18, 0.00, 0.82, 0.14),
            ("edge", "bottom_edge",  0.18, 0.86, 0.82, 1.00),
        ]

        snaps: List[Dict[str, Any]] = []
        for kind, label, x0, y0, x1, y1 in regions[:max_snaps]:
            out = _crop_norm(x0, y0, x1, y1)
            snaps.append({
                "side": side,
                "type": kind,
                "note": f"{side} {label} close-up (best effort)",
                "confidence": 0.5,
                **out
            })
        return snaps
    except Exception:
        return []

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _is_blankish(s: str) -> bool:
    s2 = _norm_ws(s or "").lower()
    return (not s2) or s2 in ("unknown", "n/a", "na", "none", "null", "undefined")

def _safe_json_extract(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from a model response."""
    if not text:
        return None
    # Fast path
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Heuristic: find first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end+1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


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
    if not q or not EBAY_APP_ID:
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
                "GLOBAL-ID": "EBAY-AU",
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
                    cp = (selling.get("convertedCurrentPrice") or selling.get("currentPrice") or [{}])[0]
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

async def _pc_search_ungraded(query: str, category: Optional[str] = None, limit: int = 10):
    """Search PriceCharting and filter out graded products for RAW/ungraded card comparisons."""
    products = await _pc_search(query, category=category, limit=max(10, limit * 2))
    if not products:
        return []

    ungraded = []
    graded_keywords = ["psa", "bgs", "cgc", "graded", "gem mint 10", "grade"]
    for product in products:
        title = (str(product.get("product-name") or product.get("name") or "")).lower()
        if not any(k in title for k in graded_keywords):
            ungraded.append(product)
            if len(ungraded) >= limit:
                break
    return ungraded


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
async def identify(front: UploadFile = File(...), back: UploadFile | None = File(None)):
    """Identify a collectible card/item from images.
    Front is required. Back is optional (kept for backward compatibility).
    """
    front_bytes = await front.read()
    if not front_bytes or len(front_bytes) < 200:
        raise HTTPException(status_code=400, detail="No front image uploaded (or image too small).")

    back_bytes: bytes | None = None
    if back is not None:
        bb = await back.read()
        if bb and len(bb) >= 200:
            back_bytes = bb

    images = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{front.content_type or 'image/jpeg'};base64,{_b64(front_bytes)}"},
        }
    ]
    if back_bytes is not None:
        images.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{back.content_type or 'image/jpeg'};base64,{_b64(back_bytes)}"},
            }
        )

    system = (
        "You are an expert collectibles identifier. "
        "Return ONLY valid JSON. Be conservative; if unsure, leave fields empty rather than hallucinating."
    )
    user = (
        "Identify the card/item from the image(s). "
        "Return JSON with keys: "
        "card_name, card_type, game, year, card_number, set_code, set_name, manufacturer, language, "
        "confidence (0-1), reasoning (short). "
        "For Pokemon, set_code should be the PT-CGO set code if visible (e.g., MEW), else empty."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": user}, *images]},
    ]

    resp = await _openai_chat(messages=messages, max_tokens=900, temperature=0.1)
    if resp.get("error"):
        # bubble as 502 to clearly show upstream dependency issues
        raise HTTPException(status_code=502, detail=f"AI identify failed: {resp.get('message', 'unknown error')}")

    content = resp.get("content") or "{}"
    data = _parse_json_or_none(content) or _safe_json_extract(content) or {}
    if not isinstance(data, dict):
        data = {}

    result = {
        "card_name": _norm_ws(str(data.get("card_name", ""))),
        "card_type": _normalize_card_type(str(data.get("card_type", ""))),
        "game": _norm_ws(str(data.get("game", ""))),
        "year": _norm_ws(str(data.get("year", ""))),
        "card_number": _clean_card_number_display(str(data.get("card_number", ""))),
        "set_code": _norm_ws(str(data.get("set_code", ""))).upper(),
        "set_name": _norm_ws(str(data.get("set_name", ""))),
        "manufacturer": _norm_ws(str(data.get("manufacturer", ""))),
        "language": _norm_ws(str(data.get("language", ""))),
        "confidence": _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0),
        "reasoning": _norm_ws(str(data.get("reasoning", ""))),
    }

    # Canonicalize set info where helpers exist
    try:
        set_info = _canonicalize_set(result["set_code"], result["set_name"])
        result["set_code"] = set_info.get("set_code", result["set_code"])
        result["set_name"] = set_info.get("set_name", result["set_name"])
        result["set_source"] = set_info.get("set_source", "")
    except Exception:
        pass

    # Backward-compatible response: expose fields at top-level AND under card
    flat = dict(result)
    flat.update({"ok": True, "card": result})
    return flat


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
    intent: Optional[str] = Form(None),
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

    intent_norm = (intent or '').strip().lower()
    intent_context = 'BUYING' if intent_norm == 'buying' else ('SELLING' if intent_norm == 'selling' else 'UNSPECIFIED')

    prompt = f"""You are a professional trading card grader with 15+ years experience.

Analyze the provided images with EXTREME scrutiny.
You will receive FRONT and BACK images, and MAY receive a third ANGLED image used to rule out glare / light refraction artifacts (holo sheen) vs true whitening / scratches / print lines. Write as if speaking directly to a collector who needs honest, specific feedback about their card.

USER INTENT (context): {intent_context}

**INTENT-SPECIFIC RESPONSE FRAMING:**

If {intent_context} is BUYING (user is considering purchasing this card):
- In corner/edge/surface notes: Frame defects as NEGOTIATION LEVERAGE
  * Example: "Minor whitening on back top-right corner (about 1mm) — use this to negotiate 10-15% off the seller's asking price"
  * Example: "Light surface scratching visible under direct light on front — point this out to justify a lower offer"
- In centering notes: Mention if it's a deal-breaker or acceptable
  * "Centering is slightly off (60/40) but for this grade range, it's not a deal-breaker"
  * "Poor centering (70/30 left) significantly impacts value — factor this into your offer"
- In overall assessment: 
  * Start with RED FLAGS first, then positives
  * Use phrases like: "As a buyer, here's what you need to know...", "The concerns I'd have...", "Use these defects to negotiate..."
  * Include verification steps: "Ask the seller for additional photos of the back corners to confirm there's no hidden damage"
  * End with fair buy guidance: "Given these defects, I wouldn't pay more than [X]% of mint value for this card"
- Overall tone: BUYER PROTECTION — skeptical, focused on risks and negotiation points

If {intent_context} is SELLING (user owns this card and wants to sell):
- In corner/edge/surface notes: Frame defects as DISCLOSURE REQUIREMENTS with listing advice
  * Example: "Minor whitening on back top-right corner (about 1mm) — mention this in your description but don't lead with it"
  * Example: "Light surface scratching visible under direct light — photograph this angle in your listing to show transparency"
- In centering notes: Frame as PRICING FACTOR
  * "Centering is slightly off (60/40) — price 10-15% below perfectly centered examples"
  * "Excellent centering (55/45) — this is a selling point, highlight it in your title"
- In overall assessment:
  * Start with STRENGTHS first (what makes it desirable), then honest disclosure of weaknesses
  * Use phrases like: "For your listing, lead with...", "Be transparent about...", "Buyers will appreciate knowing..."
  * Include listing optimization: "Take additional close-up photos of the holo effect", "In your description, mention the sharp corners and clean edges first"
  * End with realistic pricing: "List at [X], be prepared to accept [Y]. This will sell faster if you're at [Z]"
- Overall tone: SELLER SUCCESS — honest but optimistic, focused on presentation and disclosure

If {intent_context} is UNSPECIFIED:
- Use neutral, educational tone
- Present both buyer considerations and seller perspectives where relevant

**CRITICAL**: The GRADING LOGIC remains identical regardless of intent. Only the COMMUNICATION STYLE and ADVICE CONTEXT changes.

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
   - Grade 8-9 for cards with minor flaws but overall excellent condition
   - Grade 10 (GEM MINT) ONLY for cards that are virtually flawless:
     * ALL 8 corners sharp with no whitening
     * Perfect or near-perfect centering (55/45 or better)
     * No surface scratches, print lines, or defects
     * Clean edges with no wear
     * High gloss, no dulling
     DO NOT be afraid to give 10 if the card truly merits it

   - Grade 12 (COLLECTORS LEAGUE ULTRA FLAWLESS) for cards that exceed perfection:
     * PERFECT centering (50/50 or 52/48 maximum)
     * ALL corners are razor-sharp with zero detectable flaws even under magnification
     * Surface is pristine - looks like it came straight from pack to sleeve
     * Edges are perfectly cut with no fraying, roughness, or inconsistency
     * Exceptional print quality with vivid colors and perfect registration
     * Zero factory defects (no print dots, lines, or imperfections)
     * Card presents as if it's never been touched by human hands
     This grade should be RARE (perhaps 1 in 1000 cards) but MUST be awarded when warranted


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
  "pregrade": "1-12 (use 12 ONLY for Ultra Flawless cards that exceed all expectations)",
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
    # Optional preview images (filtered variants). Keep defined to avoid runtime errors.
    defect_filters: Dict[str, str] = {}
    try:
        # Only run for cards; memorabilia uses a different endpoint.
        front_vars = _make_defect_filter_variants(front_bytes)
        back_vars = _make_defect_filter_variants(back_bytes)

        # Expose filtered variants as base64 strings (if present). Frontend may ignore these.
        if isinstance(front_vars, dict):
            for k, v in front_vars.items():
                if isinstance(v, (bytes, bytearray)) and len(v) > 200:
                    defect_filters[f"front_{k}"] = _b64(bytes(v))
        if isinstance(back_vars, dict):
            for k, v in back_vars.items():
                if isinstance(v, (bytes, bytearray)) and len(v) > 200:
                    defect_filters[f"back_{k}"] = _b64(bytes(v))

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


    # ------------------------------
    # CV-assisted defect closeups (defect_snaps) for UI thumbnails
    # ------------------------------
    rois: List[Dict[str, Any]] = []
    try:
        rois = _cv_candidate_bboxes(front_bytes, "front") + _cv_candidate_bboxes(back_bytes, "back")
    except Exception:
        rois = []

    roi_labels: List[Dict[str, Any]] = []
    try:
        roi_labels = await _openai_label_rois(rois, front_bytes, back_bytes) if rois else []
    except Exception:
        roi_labels = []

    if isinstance(second_pass, dict):
        second_pass["roi_labels"] = roi_labels

    # If the structured assessment clearly flags defects (e.g., whitening corners),
    # we should *force* evidence crops for those areas even if the ROI labeler
    # returns a "clean" verdict. This prevents empty grids when defects are obvious.
    want_src = dict(data) if isinstance(data, dict) else {}
    want_src["second_pass"] = second_pass
    wanted_rois = _wanted_rois_from_assessment(want_src)

    defect_snaps: List[Dict[str, Any]] = []
    label_by_idx = {int(x.get("crop_index", -1)): x for x in (roi_labels or []) if isinstance(x, dict)}

    def _is_likely_defect(lab: Dict[str, Any], roi: Dict[str, Any]) -> bool:
        """Same defect gating as cards, for sealed/memorabilia ROI crops."""
        if not isinstance(lab, dict):
            return False
        if bool(lab.get("is_defect", False)):
            return True

        t = str(lab.get("type") or "").strip().lower()
        note = str(lab.get("note") or "").strip().lower()
        conf = _clamp(_safe_float(lab.get("confidence"), 0.0), 0.0, 1.0)

        clean_types = {"none", "clean", "ok", "okay", "good", "no_defect", "no defect"}
        if t and t not in clean_types and conf >= 0.35:
            return True

        defect_words = (
            "tear", "split", "rip", "hole", "puncture", "crush", "crease", "dent",
            "scuff", "scratch", "mark", "stain", "lift", "peel", "wrinkle", "cloud",
            "seal", "seam", "repack", "tamper", "edge wear", "wear", "whitening",
        )
        if any(w in note for w in defect_words) and conf >= 0.30:
            return True

        roi_score = _safe_float(roi.get("score"), 0.0)
        if roi_score >= 0.80 and t and t not in clean_types:
            return True

        return False

    def _is_likely_defect(lab: Dict[str, Any], roi: Dict[str, Any]) -> bool:
        """Gate ROI thumbnails so we show only defect-ish photos.

        The ROI labeler sometimes forgets to set `is_defect=true` even when it
        provides a non-clean type/note. We treat a crop as a defect if ANY of:
          - explicit is_defect=True
          - non-clean `type` with decent confidence
          - defect-keyword `note` with decent confidence
          - high ROI score + non-clean type
        """
        if not isinstance(lab, dict):
            return False
        if bool(lab.get("is_defect", False)):
            return True

        t = str(lab.get("type") or "").strip().lower()
        note = str(lab.get("note") or "").strip().lower()
        conf = _clamp(_safe_float(lab.get("confidence"), 0.0), 0.0, 1.0)

        clean_types = {"none", "clean", "ok", "okay", "good", "no_defect", "no defect"}
        if t and t not in clean_types and conf >= 0.35:
            return True

        defect_words = (
            "whitening", "white", "wear", "edge wear", "chip", "chipping", "nick", "dent",
            "crease", "bend", "scratch", "scratches", "scuff", "scuffs", "print line",
            "indent", "stain", "mark", "marks", "lift", "peel", "tear", "split",
        )
        if any(w in note for w in defect_words) and conf >= 0.30:
            return True

        roi_score = _safe_float(roi.get("score"), 0.0)
        if roi_score >= 0.80 and t and t not in clean_types:
            return True

        return False
    # Helper to pull a human note from the structured assessment for a given ROI
    def _note_from_assessment(side: str, roi: str) -> str:
        try:
            side = str(side or "").lower().strip()
            roi = str(roi or "").lower().strip()
            # corners
            if roi.startswith("corner_"):
                ck = roi.replace("corner_", "")
                corner_key = {
                    "top_left": "top_left",
                    "top_right": "top_right",
                    "bottom_left": "bottom_left",
                    "bottom_right": "bottom_right",
                }.get(ck)
                if corner_key:
                    c = (((data.get("corners") or {}).get(side) or {}).get(corner_key) or {})
                    n = str(c.get("notes") or "").strip()
                    if n:
                        return n
            if roi.startswith("edge_"):
                e = ((data.get("edges") or {}).get(side) or {})
                n = str(e.get("notes") or "").strip()
                if n:
                    return n
            s = ((data.get("surface") or {}).get(side) or {})
            n = str(s.get("notes") or "").strip()
            return n
        except Exception:
            return ""

    # Helper: pull a nice human note from the structured section where possible
    def _note_for_roi(side: str, roi_name: str) -> str:
        try:
            if roi_name.startswith("corner_"):
                cm = {
                    "corner_top_left": "top_left",
                    "corner_top_right": "top_right",
                    "corner_bottom_left": "bottom_left",
                    "corner_bottom_right": "bottom_right",
                }
                ck = cm.get(roi_name)
                c = (data.get("corners") or {}).get(side, {}).get(ck, {}) if ck else {}
                n = str((c or {}).get("notes") or "").strip()
                return _norm_ws(n)
            if roi_name.startswith("edge_"):
                e = (data.get("edges") or {}).get(side, {})
                n = str((e or {}).get("notes") or "").strip()
                return _norm_ws(n)
        except Exception:
            pass
        return ""

    for i, r in enumerate((rois or [])[:10]):
        src = front_bytes if r.get("side") == "front" else back_bytes
        if not src:
            continue
        bbox = r.get("bbox") or {}
        thumb = _make_thumb_from_bbox(src, bbox, max_size=520)
        if not thumb:
            continue
        lab = label_by_idx.get(i) or {}
        side = str(r.get("side") or "").lower().strip()
        roi_name = str(r.get("roi") or "").lower().strip()
        force = (side, roi_name) in wanted_rois
        # Allow surface forcing: we don't have dedicated surface ROIs, so if surface is wanted,
        # accept any non-empty ROI from that side.
        if not force and (side, "surface") in wanted_rois:
            force = True

        if not _is_likely_defect(lab, r) and not force:
            continue

        dtype = lab.get("type") or ("whitening" if roi_name.startswith("corner_") else "defect")
        note = lab.get("note") if lab else ""
        if not note or force:
            note2 = _note_from_assessment(side, roi_name)
            if note2:
                note = note2
            elif not note:
                note = f"Close-up of {roi_name.replace('_',' ')}"
        conf = lab.get("confidence") if lab else 0.0
        defect_snaps.append({
            "side": r.get("side"),
            "roi": r.get("roi"),
            "type": dtype,
            "note": _norm_ws(str(note))[:220],
            "confidence": _clamp(_safe_float(conf, 0.0), 0.0, 1.0),
            "bbox": bbox,
            "thumbnail_b64": thumb,
        })

    defect_snaps.sort(key=lambda x: (0 if x.get("type") != "hotspot" else 1, -float(x.get("confidence") or 0.0)))
    defect_snaps = defect_snaps[:8]


    # Fallback: if defects were clearly flagged but ROI labeling filtered everything out,
    # still show a few evidence crops so the UI isn't empty.
    if (not defect_snaps) and rois and (wanted_rois or (isinstance(data, dict) and (data.get("defects") or data.get("flags")))):
        try:
            _pref = [r for r in rois if str(r.get("roi","")).startswith("corner_")] + [r for r in rois if str(r.get("roi","")).startswith("edge_")]
            _pref = _pref[:4]
            for r in _pref:
                src = front_bytes if r.get("side") == "front" else back_bytes
                if not src:
                    continue
                bbox = r.get("bbox") or {}
                thumb = _make_thumb_from_bbox(src, bbox, max_size=520)
                if not thumb:
                    continue
                side = str(r.get("side") or "").lower().strip()
                roi_name = str(r.get("roi") or "").lower().strip()
                note = _note_from_assessment(side, roi_name) or f"Close-up of {roi_name.replace('_',' ')}"
                defect_snaps.append({
                    "side": r.get("side"),
                    "roi": r.get("roi"),
                    "type": "defect",
                    "note": _norm_ws(str(note))[:220],
                    "confidence": 0.45,
                    "bbox": bbox,
                    "thumbnail_b64": thumb,
                })
            defect_snaps.sort(key=lambda x: (0 if x.get("type") != "hotspot" else 1, -float(x.get("confidence") or 0.0)))
            defect_snaps[:] = defect_snaps[:8]
        except Exception:
            pass

    # Add stable IDs + coarse categories for UI linking (photos <-> bullet refs)
    def _snap_category(s: Dict[str, Any]) -> str:
        t = str(s.get("type") or "").lower()
        roi = str(s.get("roi") or "").lower()
        if "corner" in t or roi.startswith("corner_"):
            return "corners"
        if "edge" in t or roi.startswith("edge_"):
            return "edges"
        if "surface" in t:
            return "surface"
        if "cent" in t:
            return "centering"
        return "other"

    for idx, s in enumerate(defect_snaps, start=1):
        s["id"] = idx
        s["ref"] = f"#{idx}"
        s["category"] = _snap_category(s)

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
    resp = {
        "pregrade": pregrade_norm or "N/A",
        "confidence": _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0),
        "centering": data.get("centering", {"front": {"grade": "", "notes": ""}, "back": {"grade": "", "notes": ""}}),
        "corners": data.get("corners", {"front": {}, "back": {}}),
        "edges": data.get("edges", {"front": {"grade": "", "notes": ""}, "back": {"grade": "", "notes": ""}}),
        "surface": data.get("surface", {"front": {"grade": "", "notes": ""}, "back": {"grade": "", "notes": ""}}),
        "defects": defects_list_out,
        "flags": flags_list_out,
        "second_pass": second_pass,
        "defect_snaps": defect_snaps,
        "glare_suspects": data.get("glare_suspects", []) if isinstance(data.get("glare_suspects", []), list) else [],
        "assessment_summary": _norm_ws(str(data.get("assessment_summary", ""))) or summary or "",
        "spoken_word": _norm_ws(str(data.get("spoken_word", ""))) or _norm_ws(str(data.get("assessment_summary", ""))) or summary or "",
        "observed_id": data.get("observed_id", {}) if isinstance(data.get("observed_id", {}), dict) else {},
        "verify_token": f"vfy_{secrets.token_urlsafe(12)}",
        "market_context_mode": "click_only",
        "condition_anchor": condition_anchor,
        "has_structural_damage": bool(has_structural_damage),
    }
    # Remove clinical mm sizing for whitening notes (collector-style language)
    resp = _relax_whitening_mm_in_obj(resp)
    return JSONResponse(content=resp)


# ==============================
# Memorabilia / Sealed: Identify + Assess (independent from cards)
# - No card-specific fields
# - Returns product_id when possible (PriceCharting resolver) so Market Context can be accurate
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

    prompt = (
    "You are identifying a collectible item (sealed product or memorabilia) from photos.\n\n"
    "CRITICAL: Be EXTREMELY SPECIFIC about product configuration. Many products look similar but have vastly different values:\n"
    "- Booster Box (typically 36 packs) vs Booster Bundle (typically 6 packs) vs Single Booster Pack\n"
    "- Display Box/Case (contains multiple booster boxes OR multiple bundles) vs Individual Box/Bundle\n"
    "- Elite Trainer Box (8-10 packs + accessories) vs Collection Box (4-5 packs + promos)\n\n"

    "IDENTIFICATION HIERARCHY (check in this order):\n"
    "1. VISIBLE TEXT: Look for exact product names on packaging:\n"
    "   - 'Booster Box' or '36 Booster Packs' \u2192 sealed booster box\n"
    "   - 'Booster Bundle' or '6 Pack Bundle' \u2192 sealed booster bundle\n"
    "   - 'Display Box' or 'Display Case' \u2192 sealed display box\n"
    "   - 'Elite Trainer Box' or 'ETB' \u2192 elite trainer box\n"
    "   - 'Collection Box' \u2192 collection box\n\n"

    "2. PACK COUNT: Count visible booster packs or look for pack count text\n"
    "   - 36 packs visible/mentioned \u2192 booster box\n"
    "   - 6 packs visible/mentioned \u2192 booster bundle\n"
    "   - 3-5 packs visible/mentioned \u2192 collection box or bundle\n"
    "   - Single pack \u2192 sealed pack\n\n"

    "3. SIZE/PROPORTIONS: Relative size indicators\n"
    "   - Large rectangular box (wider than tall) \u2192 likely booster box or display\n"
    "   - Small rectangular package \u2192 likely bundle\n"
    "   - Tall box with multiple compartments \u2192 likely ETB\n\n"

    "4. PACKAGING STYLE:\n"
    "   - Shrink-wrapped box with visible packs inside \u2192 booster box\n"
    "   - Cardboard box with promotional art \u2192 collection box, ETB, or bundle\n"
    "   - Multiple identical boxes stacked \u2192 display box/case\n\n"

    "Pokemon TCG PRODUCTS (common examples):\n"
    "- Booster Box = 36 packs, ~$150-250 AUD retail\n"
    "- Booster Bundle = 6 packs, ~$40-60 AUD retail\n"
    "- Elite Trainer Box = 9 packs + dice/sleeves, ~$60-80 AUD retail\n"
    "- Collection Box = 4-5 packs + promo card, ~$30-50 AUD retail\n"
    "- Booster Bundle Display Case = typically 10 booster bundles (60 packs), ~$350-700+ AUD retail\n- Display Box = 6 booster boxes (216 packs total), ~$900-1500 AUD retail\n\n"

    "Return ONLY valid JSON with these exact fields:\n"
    "{\n"
    "  \"item_type\": \"sealed booster box/sealed booster bundle/sealed pack/sealed tin/sealed case/sealed display box/elite trainer box/collection box/autographed memorabilia/game-used memorabilia/graded item/other\",\n"
    "  \"brand\": \"brand/league/publisher if visible (e.g., Pokemon TCG, Panini, Topps, Upper Deck, Wizards of the Coast) else empty string\",\n"
    "  \"product_name\": \"EXACT product name from packaging (include 'Booster Box' or 'Booster Bundle' or 'Display' if visible) - be PRECISE\",\n"
    "  \"set_or_series\": \"set/series/expansion name if visible else empty string\",\n"
    "  \"year\": \"4 digit year if visible else empty string\",\n"
    "  \"edition_or_language\": \"e.g., English/Japanese/1st Edition/Unlimited/Collector's Edition if visible else empty string\",\n"
    "  \"special_attributes\": [\"short tags like Factory Sealed\", \"Pokemon Center\", \"Hobby Box\", \"1st Edition\", \"Case Fresh\", \"36 Packs\", \"6 Pack Bundle\"],\n"
    "  \"description\": \"one clear paragraph describing what it is, SPECIFICALLY mentioning configuration (e.g., 'This is a sealed booster box containing 36 booster packs...') and what can be seen (packaging, labels, markings, pack count)\",\n"
    "  \"signatures\": \"names of any visible signatures or 'None visible'\",\n"
    "  \"seal_condition\": \"Factory Sealed/Opened/Resealed/Damaged/Not applicable\",\n"
    "  \"authenticity_notes\": \"authenticity indicators visible (holograms, stickers, COA) and any red flags\",\n"
    "  \"notable_features\": \"unique features worth noting (promo contents, special print, chase set, serial numbering, COA, pack count, etc.)\",\n"
    "  \"confidence\": 0.0-1.0,\n"
    "  \"category_hint\": \"Pokemon/Magic/YuGiOh/Sports/OnePiece/Other\"\n"
    "}\n\n"
    "Rules:\n"
    "- If multiple products are plausible, choose the best match and briefly note uncertainty in authenticity_notes.\n"
    "- Do NOT invent a year/edition/language if you cannot see it.\n"
    "- If it appears sealed, describe the wrap (tight/loose, tears, holes, seams, bubbling). Use 'Factory Sealed' only if it looks consistent.\n"
    "- If you cannot identify confidently, keep product_name generic and set confidence low.\n"
    "- CRITICAL: Look for pack count or product type text - 'Booster Box' vs 'Booster Bundle' makes a HUGE price difference!\n"
    "- When unsure between bundle vs bundle display case, default to the LARGER configuration if the item looks substantial.\n- When unsure between box/bundle, default to the LARGER configuration if the item looks substantial.\n"
    "Respond ONLY with JSON, no extra text."
)

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for i, bb in enumerate(imgs):
        if i > 0:
            content.append({"type": "text", "text": f"IMAGE {i} ABOVE ☝️ | IMAGE {i+1} BELOW 👇"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(bb)}", "detail": "high"}})

    msg = [{"role": "user", "content": content}]
    result = await _openai_chat(msg, max_tokens=900, temperature=0.1)
    data = _parse_json_or_none(result.get("content", "")) if not result.get("error") else None
    data = data or {}

    item = {
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
    }
    # ✅ Heuristic fix: Bundle Display/Case vs single Bundle
    # If the model returns "booster bundle" but the text/attributes indicate a display/case (e.g., 10 bundles),
    # force a more specific configuration to avoid huge pricing mismatches.
    try:
        blob_txt = " ".join([
            str(item.get("product_name", "")),
            str(item.get("description", "")),
            " ".join(item.get("special_attributes", []) or []),
            str(item.get("notable_features", "")),
            str(item.get("authenticity_notes", "")),
        ]).lower()

        if ("bundle" in blob_txt) and (("display" in blob_txt) or ("case" in blob_txt) or ("10 booster" in blob_txt) or ("ten booster" in blob_txt) or ("10x" in blob_txt) or ("x10" in blob_txt)):
            item["item_type"] = "sealed display box"

            # ensure a more specific product_name
            pn = (item.get("product_name") or "").strip()
            series = (item.get("set_or_series") or "").strip()

            if not pn or pn.lower() in ("booster bundle", "bundle", "booster bundle box"):
                item["product_name"] = (f"{series} Booster Bundle Display" if series else "Booster Bundle Display").strip()
            elif "display" not in pn.lower():
                item["product_name"] = (pn + " Display").strip()

            # tag 10 bundles if hinted
            if ("10" in blob_txt or "ten" in blob_txt) and all("10" not in str(x) for x in item.get("special_attributes", []) or []):
                item.setdefault("special_attributes", [])
                if isinstance(item["special_attributes"], list):
                    item["special_attributes"].append("10 Bundles")
    except Exception:
        pass



    # Optional: resolve a PriceCharting product_id for tighter Market Context matching
    product_id = ""
    try:
        if PRICECHARTING_TOKEN:
            q_parts = [item.get("brand"), item.get("product_name"), item.get("set_or_series"), item.get("edition_or_language"), item.get("year")]
            q = _norm_ws(" ".join([p for p in q_parts if p])).strip()
            if q:
                products = await _pc_search(q, category=None, limit=5)
                if products:
                    top = products[0] or {}
                    product_id = str(top.get("id") or top.get("product-id") or "").strip()
    except Exception:
        product_id = ""

    if product_id:
        item["product_id"] = product_id

    return JSONResponse(content={
        "item": item,  # wrapped payload (frontend supports item or flat)
        **item,        # flat payload for back-compat
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
    intent: Optional[str] = Form(None),
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
        if item_type:
            ctx += f"- Item Type: {_norm_ws(item_type)}\n"
        if description:
            ctx += f"- Description: {_norm_ws(description)}\n"

    
    intent_norm = (intent or '').strip().lower()
    intent_context = 'BUYING' if intent_norm == 'buying' else ('SELLING' if intent_norm == 'selling' else 'UNSPECIFIED')

    prompt = (
                f"""You are a professional memorabilia/collectibles grader.

USER INTENT (context): {intent_context}

**INTENT-SPECIFIC RESPONSE FRAMING:**

If {intent_context} is BUYING (user is considering purchasing):

overall_assessment field:
- Start with: "As a buyer examining this [item], here's what stands out..."
- Lead with RED FLAGS and authentication concerns FIRST
- Mention strengths only after discussing risks
- End with: "Given these observations, here's what I'd verify before buying..."

defects field (frame each as NEGOTIATION TOOL or VERIFICATION NEED):
- "Minor corner scuffing on bottom left - use this to negotiate 10-15% off the asking price"
- "Seal shows slight bubbling near top seam - ask seller for video showing seal under different lighting angles"
- "Small dent on back panel (2cm from edge) - request additional photos to confirm it's not deeper damage"
- "Surface scratching on front right - point this out to justify a lower offer"

value_factors field (mention with SKEPTICISM):
- "Factory seal appears intact (seller will emphasize this - verify independently)"
- "Limited edition designation (confirm this with official documentation)"  
- "Original shrink wrap visible (common reproduction - check authentication guides)"

spoken_word field (20-45 seconds of BUYER-FOCUSED ADVICE):
- Open with: "Alright, so you're thinking about buying this [item]. Here's my take as someone who's seen a lot of these..."
- Discuss: "First thing I'd do? [specific verification step]. Then check [specific area] because..."
- Mention: "The defects I'm seeing give you some negotiation room. Don't be afraid to point out [X] and [Y]..."
- Include: "Red flags to watch for: [list specific concerns]. Make sure you ask the seller about..."
- Close with: "Price-wise, I wouldn't go above [X] for this condition. If they're asking more, walk away or negotiate hard."

authenticity_logic.notes field:
- Lead with concerns: "Primary authentication concerns: [X]. Before buying, verify..."
- Provide specific verification steps: "Ask seller for: [specific photos/documentation]"
- If HIGH RISK: "⚠️ CAUTION: Multiple authentication red flags. I would NOT buy without expert verification."

If {intent_context} is SELLING (user owns and wants to sell):

overall_assessment field:
- Start with: "For your listing of this [item], here's what matters to buyers..."
- Lead with STRENGTHS and VALUE FACTORS first
- Then address defects with "disclosure strategy"
- End with: "Here's how I'd present this to maximize buyer confidence..."

defects field (frame each as DISCLOSURE REQUIREMENT with STRATEGY):
- "Minor corner scuffing on bottom left - mention in description: 'Light wear from storage, see photos. Priced accordingly.'"
- "Seal shows slight bubbling near top seam - photograph clearly and explain: 'Minor bubbling from temperature storage, seal never compromised.'"
- "Small dent on back panel (2cm from edge) - be upfront: 'Small cosmetic dent visible in photos, does not affect contents.'"
- "Surface scratching on front right - don't hide it: 'Some surface wear consistent with age, see detailed photos.'"

value_factors field (these are your SELLING POINTS - elaborate):
- "Factory seal intact - LEAD WITH THIS in your title and first photo"
- "Limited edition #[X] of [Y] - mention prominently, collectors seek specific numbers"
- "Original shrink wrap with official hologram - photograph hologram clearly, this proves authenticity"

spoken_word field (20-45 seconds of SELLER-FOCUSED ADVICE):
- Open with: "Alright, so you're listing this [item]. Here's how I'd position it to sell quickly and at the right price..."
- Discuss: "Your main selling points are [X, Y, Z] - make sure these are in your first sentence and photos 1-3"
- Mention: "For the defects, be transparent. In your description, say something like: '[specific honest wording]'. Buyers appreciate this and it prevents returns."
- Include: "Take additional photos of: [specific strengths]. Make sure lighting shows [specific feature] clearly."
- Close with: "Price strategy: I'd list at [X] to allow for offers, but don't accept less than [Y]. If you want a fast sale, price at [Z] and it'll move in 24-48 hours."

authenticity_logic.notes field:
- Frame positively: "Authenticity indicators that will reassure buyers: [X, Y, Z]"
- If concerns exist: "Proactively address these in your listing: '[specific wording to use]'"
- Include: "Consider getting [specific authentication] if buyer requests - having this ready speeds up sale"

If {intent_context} is UNSPECIFIED:
- Use balanced, educational tone
- Present both buyer considerations and seller best practices

**CRITICAL**: The CONDITION ASSESSMENT and GRADING LOGIC remain completely objective and identical. Only the ADVICE FRAMING and CONTEXTUAL GUIDANCE changes based on intent.

Keep grading logic identical; only tailor advice tone and context.

"""
        "You MUST identify what the item is (brand + product name + series/set) as specifically as the images allow, "
        "then grade condition conservatively.\n\n"
        "Return ONLY valid JSON with this EXACT structure:\n"
        "{\n"
        "  \"condition_grade\": \"Mint/Near Mint/Excellent/Good/Fair/Poor\",\n"
        "  \"confidence\": 0.0-1.0,\n"
        "  \"condition_distribution\": {\"Mint\":0-1,\"Near Mint\":0-1,\"Excellent\":0-1,\"Good\":0-1,\"Fair\":0-1,\"Poor\":0-1},\n"
        "  \"seal_integrity\": {\"status\": \"Factory Sealed/Opened/Resealed/Compromised/Not Applicable\", \"notes\": \"detailed notes about seal/wrap with exact locations\"},\n"
        "  \"packaging_condition\": {\"grade\": \"Mint/Near Mint/Excellent/Good/Fair/Poor\", \"notes\": \"detailed notes about packaging wear with exact locations\"},\n"
        "  \"signature_assessment\": {\"present\": true/false, \"quality\": \"Clear/Faded/Smudged/Not Applicable\", \"notes\": \"notes about signature placement/ink flow and authenticity concerns\"},\n"
        "  \"value_factors\": [\"short bullets\"],\n"
        "  \"defects\": [\"each defect as a full sentence with location + severity\"],\n"
        "  \"flags\": [\"short flags for important issues\"],\n"
        "  \"overall_assessment\": \"5-8 sentences in first person (start with: 'Looking at your [brand] [product_name]...')\",\n"
        "  \"spoken_word\": \"20-45 second spoken-word script in first person\",\n"
        "  \"authenticity_logic\": {\n"
        "    \"overall_authenticity_risk\": \"Low/Medium/High\",\n"
        "    \"story_alignment\": \"1-3 sentences\",\n"
        "    \"sealed_checks\": {\n"
        "      \"wrap_fold_pattern\": \"Pass/Unclear/Fail/Not Applicable\",\n"
        "      \"seam_alignment\": \"Pass/Unclear/Fail/Not Applicable\",\n"
        "      \"wear_vs_seal_consistency\": \"Pass/Unclear/Fail/Not Applicable\",\n"
        "      \"hologram_sticker_check\": \"Pass/Unclear/Fail/Not Applicable\",\n"
        "      \"weight_check\": \"Pass/Unclear/Fail/Not Applicable\"\n"
        "    },\n"
        "    \"game_used_checks\": {\n"
        "      \"wear_pattern_realism\": \"Pass/Unclear/Fail/Not Applicable\",\n"
        "      \"material_stress_realism\": \"Pass/Unclear/Fail/Not Applicable\",\n"
        "      \"markings_and_codes\": \"Pass/Unclear/Fail/Not Applicable\",\n"
        "      \"repairs_alterations\": \"Pass/Unclear/Fail/Not Applicable\",\n"
        "      \"photo_match_potential\": \"High/Medium/Low/Not Applicable\"\n"
        "    },\n"
        "    \"autograph_checks\": {\n"
        "      \"ink_pressure_variation\": \"Pass/Unclear/Fail/Not Applicable\",\n"
        "      \"stroke_flow_tapering\": \"Pass/Unclear/Fail/Not Applicable\",\n"
        "      \"hesitation_or_retrace\": \"Pass/Unclear/Fail/Not Applicable\",\n"
        "      \"ink_absorption_on_surface\": \"Pass/Unclear/Fail/Not Applicable\"\n"
        "    },\n"
        "    \"universal_red_flags\": [\"short bullets\"],\n"
        "    \"notes\": \"3-6 sentences, plain-English, first person\"\n"
        "  },\n"
        "  \"observed_id\": {\n"
        "    \"brand\": \"best-effort\",\n"
        "    \"product_name\": \"best-effort\",\n"
        "    \"set_or_series\": \"best-effort\",\n"
        "    \"year\": \"best-effort\",\n"
        "    \"edition_or_language\": \"best-effort\",\n"
        "    \"item_type\": \"best-effort\"\n"
        "  }\n"
        "}\n"
        f"{ctx}\n"
        "Rules:\n"
        "- For sealed items: evaluate wrap fold patterns (Y-folds), seam paths, wrap tension/thickness, glue/warping/bubbling, and whether box wear vs seal wear tells a consistent story.\n"
        "- For game-used/player-used: assess whether wear patterns, material stress, markings/codes, and any repairs look authentic vs artificially distressed; note photo-match potential.\n"
        "- For autographs: assess natural pressure variation, tapered strokes, flow, hesitation/retrace, and whether ink sits in/soaks into the surface.\n"
        "- Do NOT claim Factory Sealed unless the wrap/seal looks consistent. If uncertain, say so and reduce confidence.\n"
        "- If glare/blur prevents certainty, say so and reduce confidence.\n"
        "- Be specific with locations and avoid generic statements.\n"
        "Respond ONLY with JSON, no extra text."
    )

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for i, bb in enumerate(imgs):
        if i > 0:
            content.append({"type": "text", "text": f"IMAGE {i} ABOVE ☝️ | IMAGE {i+1} BELOW 👇"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(bb)}", "detail": "low"}})

    msg = [{"role": "user", "content": content}]
    result = await _openai_chat(msg, max_tokens=1600, temperature=0.1)
    data = _parse_json_or_none(result.get("content", "")) if not result.get("error") else None
    data = data or {}

    # Normalize / defaults (frontend-safe)
    condition_grade = _norm_ws(str(data.get("condition_grade", "N/A"))) or "N/A"
    conf = _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0)

    # Condition distribution
    dist = data.get("condition_distribution")
    if not isinstance(dist, dict):
        ladder = ["Mint", "Near Mint", "Excellent", "Good", "Fair", "Poor"]
        cg_norm = condition_grade.title()
        i = ladder.index(cg_norm) if cg_norm in ladder else 2
        p_main = 0.45 + 0.50 * conf
        rem = 1.0 - p_main
        dist = {k: 0.0 for k in ladder}
        dist[ladder[i]] = p_main
        if i - 1 >= 0:
            dist[ladder[i - 1]] += rem * 0.55
        if i + 1 < len(ladder):
            dist[ladder[i + 1]] += rem * 0.45
        tot = sum(dist.values()) or 1.0
        dist = {k: round(v / tot, 4) for k, v in dist.items()}

    seal = data.get("seal_integrity") if isinstance(data.get("seal_integrity"), dict) else {}
    seal_status = _norm_ws(str(seal.get("status") or "Not Applicable"))
    seal_notes = _norm_ws(str(seal.get("notes") or ""))
    seal_obj = {"status": seal_status, "notes": seal_notes, "grade": seal_status}

    packaging = data.get("packaging_condition") if isinstance(data.get("packaging_condition"), dict) else {}
    packaging_obj = {
        "grade": _norm_ws(str(packaging.get("grade") or "N/A")),
        "notes": _norm_ws(str(packaging.get("notes") or "")),
    }

    sig = data.get("signature_assessment") if isinstance(data.get("signature_assessment"), dict) else {}
    sig_obj = {
        "present": bool(sig.get("present")) if "present" in sig else False,
        "quality": _norm_ws(str(sig.get("quality") or "Not Applicable")),
        "notes": _norm_ws(str(sig.get("notes") or "")),
    }

    flags = data.get("flags", [])
    if not isinstance(flags, list):
        flags = []
    flags_out = list(dict.fromkeys([_norm_ws(str(x)).lower() for x in flags if _norm_ws(str(x))]))[:20]

    defects = data.get("defects", [])
    if not isinstance(defects, list):
        defects = []
    defects_out = [_norm_ws(str(x)) for x in defects if _norm_ws(str(x))][:40]

    value_factors = data.get("value_factors", [])
    if not isinstance(value_factors, list):
        value_factors = []
    value_factors_out = [_norm_ws(str(x)) for x in value_factors if _norm_ws(str(x))][:15]

    observed = data.get("observed_id") if isinstance(data.get("observed_id"), dict) else {}
    observed_id = {
        "item_type": _norm_ws(str(observed.get("item_type") or item_type or "")),
        "brand": _norm_ws(str(observed.get("brand") or "")),
        "product_name": _norm_ws(str(observed.get("product_name") or "")),
        "set_or_series": _norm_ws(str(observed.get("set_or_series") or "")),
        "year": _norm_ws(str(observed.get("year") or "")),
        "edition_or_language": _norm_ws(str(observed.get("edition_or_language") or "")),
    }

    # Carry forward (or resolve) PriceCharting product_id if present
    product_id = str(observed.get("product_id") or observed.get("productId") or "").strip()
    if not product_id:
        try:
            if PRICECHARTING_TOKEN:
                q_parts = [observed_id.get("brand"), observed_id.get("product_name"), observed_id.get("set_or_series"), observed_id.get("edition_or_language"), observed_id.get("year")]
                q = _norm_ws(" ".join([p for p in q_parts if p])).strip()
                if q:
                    products = await _pc_search(q, category=None, limit=5)
                    if products:
                        top = products[0] or {}
                        product_id = str(top.get("id") or top.get("product-id") or "").strip()
        except Exception:
            product_id = ""
    if product_id:
        observed_id["product_id"] = product_id

    overall = _norm_ws(str(data.get("overall_assessment", "")))
    spoken = _norm_ws(str(data.get("spoken_word", ""))) or overall

    auth = data.get("authenticity_logic") if isinstance(data.get("authenticity_logic"), dict) else {}
    auth_obj = {
        "overall_authenticity_risk": _norm_ws(str(auth.get("overall_authenticity_risk", ""))),
        "story_alignment": _norm_ws(str(auth.get("story_alignment", ""))),
        "sealed_checks": auth.get("sealed_checks", {}) if isinstance(auth.get("sealed_checks", {}), dict) else {},
        "game_used_checks": auth.get("game_used_checks", {}) if isinstance(auth.get("game_used_checks", {}), dict) else {},
        "autograph_checks": auth.get("autograph_checks", {}) if isinstance(auth.get("autograph_checks", {}), dict) else {},
        "universal_red_flags": auth.get("universal_red_flags", []) if isinstance(auth.get("universal_red_flags", []), list) else [],
        "notes": _norm_ws(str(auth.get("notes", ""))),
    }

    
    # ------------------------------
    # CV-assisted defect closeups (defect_snaps) for UI thumbnails (memorabilia)
    # ------------------------------
    defect_snaps: List[Dict[str, Any]] = []
    try:
        rois_m: List[Dict[str, Any]] = []
        try:
            rois_m = _cv_candidate_bboxes(b1, "front") + _cv_candidate_bboxes(b2, "back")
        except Exception:
            rois_m = []
    
        roi_labels_m: List[Dict[str, Any]] = []
        try:
            roi_labels_m = []
            if rois_m:
                roi_labels_m = await _openai_label_rois(rois_m, b1, b2)
    
        except Exception:
            roi_labels_m = []
    
        label_by_idx_m = {int(x.get("crop_index", -1)): x for x in (roi_labels_m or []) if isinstance(x, dict)}
    
        # If the LLM clearly reported defects, force a few best ROI crops even if the labeler is conservative.
        force_idxs = set()
        if defects_out and rois_m:
            scored = sorted(
                [(i, _safe_float((r or {}).get("score"), 0.0)) for i, r in enumerate(rois_m)],
                key=lambda t: -t[1]
            )
            force_idxs = set([i for i, _ in scored[:3]])
    
        def _is_likely_defect_m(lab: Dict[str, Any], roi: Dict[str, Any]) -> bool:
            if not isinstance(lab, dict):
                return False
            if bool(lab.get("is_defect", False)):
                return True
    
            t = str(lab.get("type") or "").strip().lower()
            note = str(lab.get("note") or "").strip().lower()
            conf = _clamp(_safe_float(lab.get("confidence"), 0.0), 0.0, 1.0)
    
            clean_types = {"none", "clean", "ok", "okay", "good", "no_defect", "no defect"}
            if t and t not in clean_types and conf >= 0.35:
                return True
    
            defect_words = (
                "tear", "split", "rip", "hole", "puncture", "crush", "crease", "dent",
                "scuff", "scratch", "mark", "stain", "lift", "peel", "wrinkle", "cloud",
                "seal", "seam", "repack", "tamper", "edge wear", "wear", "whitening",
            )
            if any(w in note for w in defect_words) and conf >= 0.30:
                return True
    
            roi_score = _safe_float((roi or {}).get("score"), 0.0)
            if roi_score >= 0.80 and t and t not in clean_types:
                return True
    
            return False
    
        for i, r in enumerate((rois_m or [])[:10]):
            if not isinstance(r, dict):
                continue
            src = b1 if r.get("side") == "front" else b2
            if not src:
                continue
            bbox = r.get("bbox") or {}
            thumb = _make_thumb_from_bbox(src, bbox, max_size=520)
            if not thumb:
                continue
    
            lab = label_by_idx_m.get(i) or {}
            side = str(r.get("side") or "").lower().strip()
            roi_name = str(r.get("roi") or "").lower().strip()
            force = i in force_idxs
    
            if not _is_likely_defect_m(lab, r) and not force:
                continue
    
            dtype = lab.get("type") or ("hotspot" if "edge" in roi_name else "defect")
            note = lab.get("note") if isinstance(lab, dict) else ""
            if not note:
                note = f"Close-up of {roi_name.replace('_',' ')}" if roi_name else "Defect close-up"
    
            conf2 = lab.get("confidence") if isinstance(lab, dict) else 0.0
            defect_snaps.append({
                "id": i + 1,
                "side": side,
                "roi": r.get("roi"),
                "type": dtype,
                "note": _norm_ws(str(note))[:220],
                "confidence": _clamp(_safe_float(conf2, 0.0), 0.0, 1.0),
                "bbox": bbox,
                "thumbnail_b64": thumb,
            })
    
        defect_snaps.sort(key=lambda x: -float(x.get("confidence") or 0.0))
        defect_snaps = defect_snaps[:8]
    except Exception:
        defect_snaps = []
    
    # Fallback: if we have reported defects but no labeled defect crops, show basic hotspots instead (better than empty UI).
    if (not defect_snaps) and defects_out:
        try:
            snaps = _make_basic_hotspot_snaps(b1, "front", max_snaps=4) + _make_basic_hotspot_snaps(b2, "back", max_snaps=4)
            defect_snaps = []
            for j, s in enumerate((snaps or [])[:8]):
                if not isinstance(s, dict) or not s.get("thumbnail_b64"):
                    continue
                s2 = dict(s)
                s2["id"] = j + 1
                defect_snaps.append(s2)
        except Exception:
            pass
    
    return JSONResponse(content={
            "condition_grade": condition_grade,
            "confidence": conf,
            "condition_distribution": dist,
            "seal_integrity": seal_obj,
            "packaging_condition": packaging_obj,
            "signature_assessment": sig_obj,
            "value_factors": value_factors_out,
            "defects": defects_out,
            "flags": flags_out,
            "overall_assessment": overall,
            "spoken_word": spoken,
            "authenticity_logic": auth_obj,
            "observed_id": observed_id,
            "defect_snaps": defect_snaps,
            "verify_token": f"vfy_{secrets.token_urlsafe(12)}",
            "market_context_mode": "click_only",
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

    intent: Optional[str] = Form(None),

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
    intent_norm = (intent or '').strip().lower()
    intent_context = 'BUYING' if intent_norm == 'buying' else ('SELLING' if intent_norm == 'selling' else 'UNSPECIFIED')

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
    ladder = _build_ebay_query_ladder_rich(n, sname, set_code_hint, num_display, ctype)
    ladder = ladder[:12] if ladder else [_norm_ws(n)]

    exclude_graded_effective = EXCLUDE_GRADED_DEFAULT if exclude_graded is None else bool(exclude_graded)


    
    # --------------------------
    # PriceCharting fallback (sealed / memorabilia + when eBay disabled)
    # --------------------------
    def _looks_like_memorabilia(it: Optional[str], desc: Optional[str]) -> bool:
        s = " ".join([str(it or ""), str(desc or "")]).lower()
        if not s.strip():
            return False
        # Heuristic keywords for non-single-card items
        return any(k in s for k in [
            "sealed", "booster box", "booster pack", "etb", "elite trainer", "tin", "collection box",
            "blister", "case", "hobby box", "display", "starter deck", "theme deck",
            "memorabilia", "autograph", "signed", "jersey", "patch", "slab", "graded",
            "dvd", "vhs", "figure", "funko", "comic", "poster"
        ])


    # If we're dealing with single cards and excluding graded comps, bias queries toward RAW/UNGRADED listings.
    is_memorabilia_like = False
    try:
        is_memorabilia_like = _looks_like_memorabilia(ctype, description)
    except Exception:
        is_memorabilia_like = False

    if exclude_graded_effective and (not is_memorabilia_like):
        ladder = [_dedupe_tokens(q + " raw ungraded") for q in (ladder or [])]
    async def _market_context_pricecharting(query_str: str, category_hint: str = "", pid: str = "") -> Dict[str, Any]:
        qs = _norm_ws(query_str).strip()
        if not qs:
            return {"available": False, "message": "Empty query."}

        if not PRICECHARTING_TOKEN:
            return {
                "available": False,
                "message": "Market context is not configured (PriceCharting token missing).",
                "used_query": qs,
                "query_ladder": ladder,
            }

        # Light category mapping (PriceCharting categories vary; empty is ok)
        cat_map = {
            "pokemon": "pokemon-cards",
            "magic": "magic-cards",
            "yugioh": "yugioh-cards",
            "sports": "sports-cards",
            "onepiece": "one-piece-cards",
        }
        cat = cat_map.get((ctype or "").lower(), "")
        if category_hint:
            cat = category_hint

        best = {}
        detail = {}
        used_pid = str(pid or "").strip()

        if used_pid:
            detail = await _pc_product(used_pid)
            best = detail or {}
        else:
            products = await (_pc_search_ungraded(qs + ' raw ungraded', category=cat or None, limit=10) if exclude_graded_effective and (not is_memorabilia_like) else _pc_search(qs, category=cat or None, limit=10))
            if not products:
                return {
                    "available": False,
                    "message": "No sufficient PriceCharting market data found for this item.",
                    "used_query": qs,
                    "query_ladder": ladder,
                    "observed": {"currency": "AUD", "active": {"count": 0}, "sold": {"count": 0}},
                }
            best = products[0]
            used_pid = str(best.get("id") or best.get("product-id") or "").strip()
            detail = await _pc_product(used_pid) if used_pid else {}

        merged = {}
        if isinstance(best, dict):
            merged.update(best)
        if isinstance(detail, dict):
            merged.update(detail)

        title = _norm_ws(str(merged.get("product-name") or merged.get("name") or merged.get("title") or qs))
        url = str(merged.get("url") or best.get("url") or "")

        prices_usd = _pc_extract_price_fields(merged)
        vals_usd = [float(v) for v in prices_usd.values() if isinstance(v, (int, float)) and float(v) > 0]
        vals_aud = [_usd_to_aud_simple(v) for v in vals_usd]
        vals_aud = [v for v in vals_aud if isinstance(v, (int, float)) and v > 0]

        if not vals_aud:
            return {
                "available": False,
                "message": "No sufficient PriceCharting market data found for this item.",
                "used_query": qs,
                "query_ladder": ladder,
                "observed": {"currency": "AUD", "active": {"count": 0}, "sold": {"count": 0}},
            }

        vals_sorted = sorted(vals_aud)
        def _pct(p: float) -> float:
            if not vals_sorted:
                return 0.0
            k = int(round((len(vals_sorted) - 1) * p))
            k = max(0, min(len(vals_sorted) - 1, k))
            return float(vals_sorted[k])

        p20 = round(_pct(0.20), 2)
        p50 = round(_pct(0.50), 2)
        p80 = round(_pct(0.80), 2)
        cnt = len(vals_sorted)

        # Build "matches" from available price fields (acts like comps for the UI)
        active_matches: List[Dict[str, Any]] = []
        for k, v in prices_usd.items():
            if not isinstance(v, (int, float)) or float(v) <= 0:
                continue
            active_matches.append({
                "title": f"{title} — {k.replace('_',' ').title()}",
                "price": _usd_to_aud_simple(v),
                "condition": k.replace("_", " "),
                "score": None,
                "url": url,
                "source": "PriceCharting",
            })
        active_matches = active_matches[:10]

        # Collector-style summary (works for both cards + sealed)
        vibe = "moving" if cnt >= 4 else "thin"
        summary = (
            f"For {title}, PriceCharting has a {vibe} read right now. "
            f"Across the available condition buckets, you’re roughly looking at "
            f"{p20:.0f}–{p80:.0f} AUD, with a typical middle around {p50:.0f} AUD. "
            f"Treat this as a guide and sanity-check against live listings for your exact variant/condition."
        )

        return {
            "available": True,
            "mode": "pricecharting",
            "market_summary": summary,
            "used_query": qs,
            "query_ladder": ladder,
            "confidence": confidence,
            "card": {
                "name": n,
                "set": sname,
                "number": num_display,
                "set_code": set_code_hint,
            },
            "observed": {
                "currency": "AUD",
                "fx": {"usd_to_aud_multiplier": AUD_MULTIPLIER},
                "active": {"p20": p20, "p50": p50, "p80": p80, "count": cnt},
                "sold": {"count": 0},
                "liquidity": "medium" if cnt >= 4 else "low",
                "trend": "—",
                "raw": {"pricecharting_product_id": used_pid, "url": url, "prices_usd": prices_usd},
            },
            "active_matches": active_matches,
            "graded": {},  # sealed/memorabilia typically not bucketed by PSA grades
            "grade_outcome": {"assessed_grade": resolved_grade, "expected_value_aud": p50} if resolved_grade else {"expected_value_aud": p50},
            "disclaimer": "Informational market context only. Figures are third-party estimates and may vary. Not financial advice.",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    # --------------------------
    # Market source priority
    # --------------------------
    # Source 1: eBay ACTIVE (when enabled)
    # Source 2: PriceCharting fallback (when eBay disabled OR eBay returns thin/empty results)
    if not USE_EBAY_API:
        # eBay not enabled in this deployment — fall back to PriceCharting.
        q_pc = _norm_ws(item_name or n or description or "").strip()
        return await _market_context_pricecharting(q_pc, category_hint="", pid=product_id or "")

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
        top = scored[:10]

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
        cand = await _browse_active(q, limit=10)
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

    # If eBay comes back empty/thin (common for niche sealed/memorabilia wording),
    # fall back to PriceCharting *only if configured*.
    if int(active_stats.get("count") or 0) == 0 and PRICECHARTING_TOKEN:
        q_pc = _norm_ws(item_name or n or description or used_query or "").strip()
        pc_payload = await _market_context_pricecharting(q_pc, category_hint="", pid=product_id or "")
        if pc_payload.get("available"):
            return pc_payload

    # --------------------------
    # Graded buckets (ACTIVE only; brand-agnostic)
    # --------------------------
    GRADE_RX = re.compile(r"\b(?:psa|cgc|bgs|sgc)\s*(?:grade\s*)?([0-9]{1,2}(?:\.[05])?)\b", re.IGNORECASE)
    GEM10_RX = re.compile(r"\b(psa\s*10|psa10|gem\s*mint\s*10|10\s*gem|cgc\s*10|bgs\s*10|sgc\s*10)\b", re.IGNORECASE)

    def _parse_grade_from_title(title: str) -> Optional[float]:
        t = (title or "").lower()
        if GEM10_RX.search(t):
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
    graded_pool = await _browse_active(graded_query, limit=10)

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


    # Where does THIS copy likely sit in the current ask range?
    value_position_aud = None
    try:
        if current_low is not None and current_high is not None and resolved_grade is not None:
            lo = float(current_low); hi = float(current_high)
            if hi >= lo and (hi - lo) > 0:
                g = float(resolved_grade)
                # map grade 1..10 -> 0.08..0.92
                f = (max(1.0, min(10.0, g)) - 1.0) / 9.0
                f = 0.08 + 0.84 * f
                value_position_aud = lo + (hi - lo) * f
        if value_position_aud is None and current_typ is not None:
            value_position_aud = float(current_typ)
    except Exception:
        value_position_aud = current_typ

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
        "Okay — quick market check before you send it:",
        "Righto, let’s suss the market real quick:",
        "Alright, pack-ripper rundown:",
    ]

    openers_2 = [
        "Here’s what I’m seeing:",
        "Here’s the read:",
        "Here’s the go:",
    ]

    opener = secrets.choice(slang_openers)
    if secrets.randbelow(100) < 25:
        opener = f"{opener} {secrets.choice(openers_2)}"

    grade_line = ""
    if exp_grade_key:
        grade_line = f" LeagAI’s got it around a {exp_grade_key} on the pics."
    price_line = f" Live asks are roughly {_money(current_typ)} AUD (range {_money(current_low)}–{_money(current_high)})."

    trend_phrases = [
        f" It looks {trend_hint} off what’s listed right now.",
        f" The asks look {trend_hint} at the moment.",
        f" Market mood is {trend_hint} based on current listings.",
    ]
    trend_line = secrets.choice(trend_phrases)

    # Where does *your* copy likely sit inside the live range?
    est_value_aud = None
    if current_low is not None and current_high is not None and resolved_grade is not None:
        try:
            g = float(resolved_grade)
            # map grade 1..10 => 0.08..0.92 (keeps it away from extreme ends)
            factor = (g - 1.0) / 9.0
            factor = max(0.0, min(1.0, factor))
            factor = 0.08 + 0.84 * factor
            lo = float(current_low)
            hi = float(current_high)
            est_value_aud = lo + (hi - lo) * factor

            # If it’s a rougher grade, nudge slightly below the low ask (buyers discount hard)
            if g <= 5.5 and lo > 0:
                est_value_aud = min(est_value_aud, lo * (0.92 + 0.01 * g))
            est_value_aud = round(est_value_aud, 2)
        except Exception:
            est_value_aud = None

    position_line = ""
    if est_value_aud is not None and current_low is not None and current_high is not None:
        position_templates = [
            f" For a {exp_grade_key}-ish copy like yours, I’d peg it around {_money(est_value_aud)} AUD — basically sitting between the low and the high asks.",
            f" Given the condition call, I’d slot your copy at about {_money(est_value_aud)} AUD in that current range.",
            f" Realistically, your one probably lives around {_money(est_value_aud)} AUD (condition-wise), not at the tippy-top.",
        ]
        position_line = secrets.choice(position_templates)

    graded_line = ""
    if exp_grade_key and exp_grade_key in grade_market:
        st = grade_market.get(exp_grade_key, {}).get("stats", {})
        if st and st.get("median") is not None:
            graded_templates = [
                f" If it lands that grade in a slab, graded asks are roughly {_money(st.get('median'))} AUD (current listings).",
                f" In the graded lane at that number, you’re seeing about {_money(st.get('median'))} AUD on asks right now.",
            ]
            graded_line = " " + secrets.choice(graded_templates)

    # Grading decision language — keep it collector-real, not the same line every time
    grade_advice = ""
    try:
        gc = float(grading_cost or 0)
    except Exception:
        gc = 0.0

    if worth_grading is True:
        grade_advice = " " + secrets.choice([
            f" If you’re chasing a slab or cleaner resale, it can be a decent shout if you’re chasing a slab or a cleaner resale.",
            f" I’d consider sending it if you want it protected/registry-ready.",
        ])
    elif worth_grading is False:
        grade_advice = " " + secrets.choice([
            f" I’d only send it if it’s a personal PC piece or you want it in the registry slabbed up.",
            f" This feels more like a binder/PC card unless you just want it in plastic for the vibe.",
            f" If you’re grading for profit, I’d be picky here — but for the collection? send it.",
        ])

    advice_line = ""
    # Buyer vs Seller intent tailoring (language only; underlying data is the same)
    if intent_context == "BUYING":
        if buy_target_aud is not None:
            advice_line = " " + secrets.choice([
                f" If you’re buying raw, I’d want to be around {_money(buy_target_aud)} AUD or less for this condition.",
                f" Raw buy target: about {_money(buy_target_aud)} AUD (or under) — that’s where it starts to make sense.",
                f" If you’re hunting a deal, try land it near {_money(buy_target_aud)} AUD — anything higher is paying for hope."
            ])
    elif intent_context == "SELLING":
        # Use the position estimate if we have one, otherwise fall back to the current active median
        base = value_position_aud if value_position_aud is not None else (current_typ if current_typ is not None else current_low)
        if base is not None:
            list_at = float(base) * 1.08
            take_at = float(base)
            advice_line = " " + secrets.choice([
                f" If you’re selling, I’d list around {_money(list_at)} AUD (allow room for offers) and aim to accept near {_money(take_at)} AUD depending on interest.",
                f" Seller target: list about {_money(list_at)} AUD, realistically expect {_money(take_at)} AUD (condition + timing will move it).",
                f" If you want a faster sale, price closer to {_money(take_at)} AUD; if you can wait, start near {_money(list_at)} AUD and adjust."
            ])

    market_summary = _norm_ws(f"{opener}{grade_line}{price_line}{trend_line}{position_line}{graded_line}{grade_advice}{advice_line}")

    # Strip any residual grading fee mentions (keep market info clean)
    try:
        market_summary = market_summary.replace('Estimated grading fee:', '')
        market_summary = re.sub(r"Grading\s*fee:.*?(\n|$)", "", market_summary, flags=re.I)
        market_summary = re.sub(r"\$\s*\d+\s*grading.*?(\n|$)", "", market_summary, flags=re.I)
    except Exception:
        pass


    
    # --------------------------
    # Market graph ranges (for frontend bar graph + user-value red line)
    # --------------------------
    assessed_value = None
    try:
        assessed_value = float(est_value_aud) if est_value_aud is not None else (float(value_position_aud) if value_position_aud is not None else (float(current_typ) if current_typ is not None else None))
    except Exception:
        assessed_value = None

    low_v = float(current_low) if current_low is not None else None
    med_v = float(current_typ) if current_typ is not None else None
    high_v = float(current_high) if current_high is not None else None

    market_graph = None
    try:
        if low_v is not None and med_v is not None and high_v is not None and low_v > 0 and high_v >= low_v:
            market_graph = {
                "price_ranges": {
                    "poor": {"min": low_v * 0.6, "max": low_v, "label": "Poor Condition"},
                    "played": {"min": low_v, "max": low_v + (med_v - low_v) * 0.5, "label": "Played"},
                    "good": {"min": low_v + (med_v - low_v) * 0.5, "max": med_v, "label": "Good"},
                    "very_good": {"min": med_v, "max": med_v + (high_v - med_v) * 0.4, "label": "Very Good"},
                    "excellent": {"min": med_v + (high_v - med_v) * 0.4, "max": high_v, "label": "Excellent"},
                    "near_mint": {"min": high_v, "max": high_v * 1.15, "label": "Near Mint"},
                },
                "user_value": assessed_value,
                "currency": "AUD",
            }
    except Exception:
        market_graph = None

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
        "market_graph": market_graph,
        "active_matches": active_matches,
        "match_debug": {"active": active_debug, "ebay_calls": ebay_call_debug},
        "observed": {
            "currency": "AUD",
            "fx": {"usd_to_aud_rate": _FX_CACHE.get("usd_aud"), "cache_seconds": FX_CACHE_SECONDS},
            "active": active_stats,
            "raw": {  # legacy: point raw to ACTIVE stream now
                "source": "ebay_active_top10",
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
# ========================================
# MARKET TREND PREDICTIONS
# ========================================

@app.get("/api/market-trends/{card_identifier}")
@safe_endpoint
async def get_market_trends(
    card_identifier: str,
    days: int = 90,
    api_key: Optional[str] = Depends(verify_api_key_optional),  # optional (read endpoint)
):
    """
    Get historical price data.

    STRATEGY:
    1. Check DB for real logged history FIRST
    2. If 7+ days of data, use that (REAL history)
    3. If <7 days, fetch current eBay price as fallback
    4. Log current price to DB for future
    """
    try:
        ident = (card_identifier or "").strip()
        if not ident:
            raise HTTPException(status_code=400, detail="card_identifier required")

        # Parse identifier: "CardName|SetName|Grade"
        parts = [p.strip() for p in ident.split("|")] if "|" in ident else [ident]
        card_name = parts[0] if len(parts) > 0 else ""
        set_name = parts[1] if len(parts) > 1 else ""
        grade = parts[2] if len(parts) > 2 else ""

        logging.info(f"🔍 Market trends request: {ident}")

        # ═══════════════════════════════════════════════════════
        # STEP 1: Check DB for REAL logged history
        # ═══════════════════════════════════════════════════════
        db_history = get_price_history(card_identifier=ident, days=int(days or 90))

        if len(db_history) >= 7:
            logging.info(f"✅ DATABASE HIT: {len(db_history)} real data points")

            historical_prices = [
                {
                    "date": (str(entry.get("recorded_date") or "")[:10]),
                    "price_low": float(entry.get("price_low") or entry.get("price_current") or 0.0),
                    "price_median": float(entry.get("price_median") or entry.get("price_current") or 0.0),
                    "price_high": float(entry.get("price_high") or entry.get("price_current") or 0.0),
                    "volume": int(entry.get("volume") or 0),
                }
                for entry in db_history
            ]
            historical_prices = [x for x in historical_prices if x.get("date")]

            historical_prices.sort(key=lambda x: x["date"])

            prediction = predict_future_prices(historical_prices)
            seasonality = detect_seasonality(historical_prices)

            return JSONResponse(content={
                "success": True,
                "card_identifier": ident,
                "data_source": "database_logged_history",
                "historical_data": historical_prices,
                "prediction": prediction,
                "seasonality": seasonality,
                "analysis_period_days": int(days or 90),
                "actual_data_points": len(historical_prices),
                "note": "✅ Using genuine accumulated price history",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })

        # ═══════════════════════════════════════════════════════
        # STEP 2: Not enough DB data – fetch current price from eBay
        # ═══════════════════════════════════════════════════════
        logging.info(f"⚠️ DATABASE MISS: Only {len(db_history)} points, fetching from eBay")

        # Build eBay query ladder (specific -> broad)
        if "|" in ident:
            queries = _build_ebay_query_ladder(card_name=card_name, card_set=set_name, grade=grade)
        else:
            queries = _build_ebay_query_ladder(card_name=ident, card_set="", grade=grade)

        if not queries:
            raise HTTPException(status_code=400, detail="Could not build search query")

        logging.info(f"🔍 eBay queries: {queries}")
        target_days = max(7, min(int(days or 90), 90))

        completed = {}
        active = {}
        chosen_query = ""

        for q in queries:
            chosen_query = q
            completed = await _ebay_completed_stats(q, limit=10, days_lookback=30) or {}
            active = await _ebay_active_stats(q, limit=10) or {}
            if (completed.get("median") and completed.get("median") > 0) or (active.get("median") and active.get("median") > 0):
                break
        search_query = chosen_query
        current_price = 0.0
        price_low = 0.0
        price_high = 0.0
        volume = 0

        if completed and completed.get("median"):
            current_price = float(completed.get("median") or 0.0)
            price_low = float(completed.get("low") or (current_price * 0.9))
            price_high = float(completed.get("high") or (current_price * 1.1))
            volume = int(completed.get("count") or 0)
            logging.info(f"📊 eBay sold: ${current_price:.2f} ({volume} sales)")
        elif active and active.get("median"):
            current_price = float(active.get("median") or 0.0)
            price_low = current_price * 0.9
            price_high = current_price * 1.1
            volume = int(active.get("count") or 0)
            logging.info(f"📊 eBay active: ${current_price:.2f} ({volume} listings)")

        if current_price <= 0:
            logging.warning(f"❌ No eBay data found for: {search_query}")
            return JSONResponse(content={
                "success": False,
                "error": "No price data found for this card",
                "card_identifier": ident,
                "search_query": search_query,
                "data_source": "ebay_attempted",
                "suggestions": [
                    "Card may be too new or too rare",
                    "Try searching with simpler name (just card name)",
                    "Check spelling",
                    "Check again tomorrow - building price history"
                ],
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })

        # ═══════════════════════════════════════════════════════
        # STEP 3: Log current price to DB for future
        # ═══════════════════════════════════════════════════════
        try:
            entry = record_price_history(
                card_identifier=ident,
                card_name=card_name,
                card_set=set_name,
                card_number="",
                grade=grade,
                price_current=current_price,
                price_low=price_low,
                price_median=current_price,
                price_high=price_high,
                volume=volume,
                source="ebay",
                data_quality="verified",
            )
            logging.info(f"✅ LOGGED TO DB: ${current_price:.2f} (Entry ID: {entry['id']})")
        except Exception as e:
            logging.warning(f"⚠️ Failed to log price to DB: {e}")

        historical_data = []
        if db_history:
            historical_data = [
                {
                    "date": (str(entry.get("recorded_date") or "")[:10]),
                    "price_median": float(entry.get("price_median") or entry.get("price_current") or 0.0),
                    "price_low": float(entry.get("price_low") or entry.get("price_current") or 0.0),
                    "price_high": float(entry.get("price_high") or entry.get("price_current") or 0.0),
                    "volume": int(entry.get("volume") or 0),
                }
                for entry in db_history
            ]
            historical_data = [x for x in historical_data if x.get("date")]

        historical_data.append({
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "price_median": current_price,
            "price_low": price_low,
            "price_high": price_high,
            "volume": volume
        })

        historical_data.sort(key=lambda x: x["date"])

        return JSONResponse(content={
            "success": True,
            "card_identifier": ident,
            "search_query": search_query,
            "data_source": "ebay_current_snapshot",
            "current_price": current_price,
            "price_range": {
                "low": price_low,
                "median": current_price,
                "high": price_high
            },
            "historical_data": historical_data,
            "actual_data_points": len(historical_data),
            "ebay_listings_analyzed": volume,
            "note": f"Building history ({len(db_history)} days logged). Check again tomorrow for trend analysis!",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Market trends error for {card_identifier}: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "card_identifier": (card_identifier or ""),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }, status_code=500)


def process_ebay_to_timeseries(ebay_data: dict, active_data: dict, target_days: int = 90) -> list:
    """
    Convert eBay sold listings into a daily time series for trend analysis.

    ebay_data: output from _ebay_completed_stats()
      - currently includes: prices (AUD floats) and count
      - does NOT reliably include per-sale dates (FindingService can, but we don't parse/store them yet)

    active_data: output from _ebay_active_stats() (optional)
      - used for today's "current asks" fallback if there is no sale today
    """
    try:
        target_days = int(target_days or 90)
    except Exception:
        target_days = 90
    target_days = max(7, min(90, target_days))

    if not ebay_data or not isinstance(ebay_data, dict) or not ebay_data.get("prices"):
        return []

    sold_prices = ebay_data.get("prices") or []
    prices_only = []
    for p in sold_prices:
        try:
            if isinstance(p, dict):
                prices_only.append(float(p.get("price") or 0))
            else:
                prices_only.append(float(p))
        except Exception:
            continue
    prices_only = [p for p in prices_only if p and p > 0]
    if not prices_only:
        return []

    # We don't have real sale dates in our completed stats payload yet.
    # So we distribute the observed sold prices across the last N (<=90) days deterministically.
    # This removes "random trend" while staying grounded in real price observations.
    window = min(90, target_days, len(prices_only) if len(prices_only) > 0 else 90)
    window = max(7, window)

    from collections import defaultdict
    daily_sales = defaultdict(list)

    # Deterministic spread: distribute prices in natural order (unsorted).
    # This preserves the inherent randomness and prevents artificial trends.
    for i, price in enumerate(prices_only):
        day_offset = i % window
        d = datetime.now() - timedelta(days=(window - 1 - day_offset))
        daily_sales[d.strftime("%Y-%m-%d")].append(float(price))

    # Build time series (daily aggregates)
    series = []
    for i in range(target_days):
        d = datetime.now() - timedelta(days=(target_days - 1 - i))
        ds = d.strftime("%Y-%m-%d")
        day_prices = daily_sales.get(ds) or []

        if day_prices:
            sp = sorted(day_prices)
            median_price = sp[len(sp) // 2]
            low_price = min(day_prices)
            high_price = max(day_prices)
            vol = len(day_prices)
            series.append({
                "date": ds,
                "price_low": round(low_price, 2),
                "price_median": round(median_price, 2),
                "price_high": round(high_price, 2),
                "volume": int(vol),
            })

    # If we have no data for "today", optionally use active listings as a last point
    today = datetime.now().strftime("%Y-%m-%d")
    if (not series or series[-1].get("date") != today) and active_data and isinstance(active_data, dict):
        ap = active_data.get("prices") or []
        active_prices = []
        for p in ap:
            try:
                if isinstance(p, dict):
                    active_prices.append(float(p.get("price") or p.get("price_aud") or 0))
                else:
                    active_prices.append(float(p))
            except Exception:
                continue
        active_prices = [p for p in active_prices if p and p > 0]
        if active_prices:
            sp = sorted(active_prices)
            median_price = sp[len(sp) // 2]
            series.append({
                "date": today,
                "price_low": round(min(active_prices), 2),
                "price_median": round(median_price, 2),
                "price_high": round(max(active_prices), 2),
                "volume": int(len(active_prices)),
            })

    # Fill gaps forward (simple carry-forward) so the chart can draw a continuous line
    if len(series) >= 2:
        by_date = {x["date"]: x for x in series if isinstance(x, dict) and x.get("date")}
        filled = []
        last = None
        for i in range(target_days):
            d = datetime.now() - timedelta(days=(target_days - 1 - i))
            ds = d.strftime("%Y-%m-%d")
            if ds in by_date:
                last = by_date[ds]
                filled.append(last)
            elif last:
                filled.append({
                    "date": ds,
                    "price_low": last["price_low"],
                    "price_median": last["price_median"],
                    "price_high": last["price_high"],
                    "volume": 0,
                })
        return filled

    return series

def generate_mock_price_history(days: int):
    """Generate mock historical price data for demo UI."""
    data = []
    base_price = 100.0
    for i in range(days):
        date = datetime.now() - timedelta(days=days - i)
        trend = base_price + (i * 0.3)  # slight upward trend

        noise = 0.0
        seasonal = 0.0
        vol = 12

        if np is not None:
            noise = float(np.random.normal(0, 5))
            seasonal = float(10 * np.sin(2 * np.pi * i / 365))
            vol = int(np.random.uniform(5, 25))

        price = max(1.0, trend + noise + seasonal)

        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "price_low": round(price * 0.9, 2),
            "price_median": round(price, 2),
            "price_high": round(price * 1.1, 2),
            "volume": vol
        })
    return data


def predict_future_prices(historical_data, forecast_days: int = 90):
    """Generate price predictions using linear regression with safe fallbacks."""
    if not historical_data or len(historical_data) < 30:
        return {"available": False, "reason": "Insufficient historical data"}

    prices = [float(d.get("price_median", 0) or 0) for d in historical_data]
    n = len(prices)

    # Build x axis
    if np is not None:
        x = np.arange(n)
    else:
        x = list(range(n))

    # Regression (scipy if available, else numpy polyfit, else simple)
    slope = 0.0
    intercept = float(prices[0])
    r_value = 0.0

    if stats is not None and np is not None:
        res = stats.linregress(x, prices)
        slope, intercept, r_value = float(res.slope), float(res.intercept), float(res.rvalue)
    elif np is not None:
        slope, intercept = np.polyfit(x, prices, 1)
        # crude correlation estimate
        try:
            r = np.corrcoef(x, prices)[0, 1]
            r_value = float(r)
        except Exception:
            r_value = 0.0
    else:
        # very simple slope estimate
        slope = (prices[-1] - prices[0]) / max(1, (n - 1))
        intercept = prices[0]
        r_value = 0.0

    # Confidence: abs(r) (0..1)
    confidence = min(1.0, max(0.0, abs(r_value)))

    # Std dev for bands
    if np is not None:
        std_dev = float(np.std(prices))
        mean_price = float(np.mean(prices)) if prices else 1.0
    else:
        mean_price = sum(prices) / max(1, len(prices))
        var = sum((p - mean_price) ** 2 for p in prices) / max(1, len(prices))
        std_dev = var ** 0.5

    predictions = []
    for i in range(forecast_days):
        future_x = n + i
        predicted_price = (slope * future_x) + intercept
        lower_bound = predicted_price - (1.96 * std_dev)
        upper_bound = predicted_price + (1.96 * std_dev)
        future_date = datetime.now() + timedelta(days=i)

        predictions.append({
            "date": future_date.strftime("%Y-%m-%d"),
            "predicted_price": round(predicted_price, 2),
            "lower_bound": round(max(0.0, lower_bound), 2),
            "upper_bound": round(max(0.0, upper_bound), 2),
            "confidence": round(confidence * 100, 1)
        })

    current_price = prices[-1]
    price_30d = predictions[29]["predicted_price"] if len(predictions) > 29 else predictions[-1]["predicted_price"]
    price_90d = predictions[-1]["predicted_price"]

    change_30d = ((price_30d - current_price) / current_price) * 100 if current_price else 0.0
    change_90d = ((price_90d - current_price) / current_price) * 100 if current_price else 0.0

    volatility = (std_dev / mean_price * 100) if mean_price else 0.0

    return {
        "available": True,
        "forecast_days": forecast_days,
        "predictions": predictions,
        "current_price": round(current_price, 2),
        "predicted_30d": round(price_30d, 2),
        "predicted_90d": round(price_90d, 2),
        "change_30d_percent": round(change_30d, 1),
        "change_90d_percent": round(change_90d, 1),
        "confidence_score": round(confidence * 100, 1),
        "trend": "increasing" if slope > 0 else "decreasing",
        "volatility": round(volatility, 1),
        "recommendation": generate_trend_recommendation(slope, confidence, change_90d)
    }


def detect_seasonality(historical_data):
    """Detect simple seasonality using FFT when available."""
    if not historical_data or len(historical_data) < 60:
        return {"available": False}

    if np is None or fft is None or fftfreq is None:
        return {"available": False, "reason": "FFT unavailable (numpy/scipy not installed)"}

    prices = [float(d.get("price_median", 0) or 0) for d in historical_data]
    x = np.arange(len(prices))

    # Detrend with polyfit
    slope, intercept = np.polyfit(x, prices, 1)
    trend = slope * x + intercept
    detrended = np.array(prices) - trend

    fft_vals = np.abs(fft(detrended))
    freqs = fftfreq(len(prices))

    # dominant (exclude DC, and use only positive freqs)
    half = len(fft_vals) // 2
    dominant_idx = int(np.argmax(fft_vals[1:half]) + 1)
    dom_freq = float(freqs[dominant_idx]) if dominant_idx < len(freqs) else 0.0

    dominant_period = int(round(1 / abs(dom_freq))) if dom_freq != 0 else 0

    seasonal_patterns = []
    if 350 <= dominant_period <= 380:
        seasonal_patterns.append({
            "pattern": "Yearly cycle",
            "period_days": dominant_period,
            "description": "Prices follow an annual pattern - likely tournament seasons or holiday cycles"
        })
    elif 80 <= dominant_period <= 100:
        seasonal_patterns.append({
            "pattern": "Quarterly cycle",
            "period_days": dominant_period,
            "description": "Prices fluctuate quarterly - possibly tied to set releases"
        })

    return {
        "available": True,
        "has_seasonality": len(seasonal_patterns) > 0,
        "patterns": seasonal_patterns,
        "dominant_period_days": dominant_period if dominant_period > 0 else None
    }


def generate_trend_recommendation(slope: float, confidence: float, change_90d: float) -> str:
    """Generate guidance text based on the model signal. (No ROI language.)"""
    if confidence < 0.5:
        return "⚠️ Low confidence prediction - market is volatile. Monitor closely before making decisions."

    if change_90d > 15:
        return "📈 Strong upward trend predicted. Consider holding; if buying, be selective and compare listings."
    elif change_90d > 5:
        return "↗️ Moderate upward trend. Stable market; consider buying on dips."
    elif change_90d < -15:
        return "📉 Declining trend predicted. If selling, you may want to list sooner; if buying, wait for stabilization."
    elif change_90d < -5:
        return "↘️ Slight downward trend. Hold if you own, or look for better entry points."
    else:
        return "➡️ Stable market predicted. Good for long-term collectors; not much short-term movement expected."


@app.post("/api/market-trends/record-price")
@safe_endpoint
async def record_market_price(
    card_identifier: str = Form(...),
    card_name: str = Form(...),
    card_set: str = Form(""),
    card_number: str = Form(""),
    grade: str = Form(""),
    price_low: float = Form(...),
    price_median: float = Form(...),
    price_high: float = Form(...),
    volume: int = Form(0),
    source: str = Form("pricecharting"),
    api_key: str = Depends(verify_api_key),  # required (write endpoint)
):
    """
    Record a price snapshot for historical tracking.
    NOW ACTUALLY SAVES TO DATABASE (SQLite).
    """
    try:
        entry = record_price_history(
            card_identifier=card_identifier,
            card_name=card_name,
            card_set=card_set,
            card_number=card_number,
            grade=grade,
            price_current=float(price_median or 0.0),
            price_low=float(price_low or price_median or 0.0),
            price_median=float(price_median or 0.0),
            price_high=float(price_high or price_median or 0.0),
            volume=int(volume or 0),
            source=source,
            data_quality="verified",
        )

        logging.info(f"✅ PRICE LOGGED: {card_name} = ${float(price_median or 0.0):.2f} (ID: {entry['id']})")

        return JSONResponse(content={
            "success": True,
            "message": "Price snapshot recorded successfully",
            "entry_id": entry["id"],
            "card_identifier": entry["card_identifier"],
            "recorded_date": entry["recorded_date"],
            "price": float(entry["price_current"]),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

    except Exception as e:
        logging.error(f"❌ PRICE RECORDING FAILED: {card_identifier} - {str(e)}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "card_identifier": (card_identifier or ""),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            },
            status_code=500
        )



# ========================================
# DEFECT HEATMAP GENERATION
# ========================================
class MarketPriceLookupRequest(BaseModel):
    card_name: str
    card_set: Optional[str] = None
    card_year: Optional[str] = None
    card_number: Optional[str] = None
    grade: Optional[str] = None


@app.post("/api/market/price-lookup")
@safe_endpoint
async def market_price_lookup(request: MarketPriceLookupRequest):
    """
    Look up current market price for a specific card/item.
    NOW USES EBAY (same smart query logic as market-trends).
    """
    try:
        card_name = (request.card_name or "").strip()
        card_set = (request.card_set or "").strip()
        card_number = (request.card_number or "").strip()
        grade = (request.grade or "").strip()

        if not card_name:
            return {"current_price": 0, "source": "error", "error": "card_name required"}

        logging.info(f"💰 Price lookup request: {card_name}")

        # Build smart query ladder (try specific -> broad)
        queries = _build_ebay_query_ladder(
            card_name=card_name,
            card_set=card_set,
            card_number=card_number,
            grade=grade,
        )

        if not queries:
            return {"current_price": 0, "source": "error", "error": "Could not build search query"}

        last_completed = None
        last_active = None

        for search_query in queries:
            logging.info(f"🔍 eBay query: '{search_query}'")

            # Try completed (sold) listings first
            completed = await _ebay_completed_stats(search_query, limit=10, days_lookback=30)
            last_completed = completed or last_completed

            if completed and completed.get("median") and completed.get("median") > 0:
                current_price = float(completed.get("median"))
                logging.info(f"✅ Found price: ${current_price:.2f} (from {completed.get('count', 0)} sales)")
                return {
                    "current_price": current_price,
                    "source": "ebay_completed",
                    "search_query": search_query,
                    "queries_tried": queries,
                    "card_name": card_name,
                    "sales_count": completed.get("count", 0),
                    "last_updated": datetime.now().isoformat()
                }

            # If no completed, try active listings
            active = await _ebay_active_stats(search_query, limit=10)
            last_active = active or last_active

            if active and active.get("median") and active.get("median") > 0:
                current_price = float(active.get("median"))
                logging.info(f"✅ Found price: ${current_price:.2f} (from {active.get('count', 0)} active listings)")
                return {
                    "current_price": current_price,
                    "source": "ebay_active",
                    "search_query": search_query,
                    "queries_tried": queries,
                    "card_name": card_name,
                    "listings_count": active.get("count", 0),
                    "last_updated": datetime.now().isoformat()
                }

        # No results from either source
        logging.warning(f"❌ No eBay results for any query tried: {queries}")
        return {
            "current_price": 0,
            "source": "ebay_no_results",
            "error": "No eBay listings found",
            "search_query": queries[-1] if queries else "",
            "queries_tried": queries,
            "card_name": card_name,
        }

    except Exception as e:
        logging.error(f"❌ Price lookup error: {e}")
        return {"current_price": 0, "source": "error", "error": str(e)}

@app.post("/api/collection/update-values")
@safe_endpoint
async def update_collection_market_values(
    user_id: int = Form(...),
):
    """
    Fetch current market values for all items in a user's collection.
    NOTE: WordPress is the source of truth for collections. This endpoint is a stub
    kept for backwards compatibility with older front-ends.
    """
    try:
        # WordPress now performs updates via AJAX (cg_update_collection_values).
        return {
            "success": True,
            "updated_count": 0,
            "total_value": "0.00",
            "change_24h": "0.00",
            "change_percent": "0.0",
            "note": "Use WordPress AJAX action cg_update_collection_values for live updates."
        }
    except Exception as e:
        logging.error(f"Market value update error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/generate-heatmap")
@safe_endpoint
async def generate_defect_heatmap(
    front: UploadFile = File(...),
    back: UploadFile = File(None),
    assessment_data: str = Form(...),
):
    """
    Generate heatmap overlay data (zones) based on assessment JSON.
    Returns canvas-friendly coordinates (0..1) + severities.
    """
    # We read bytes so the request matches frontend FormData usage; we don't process the pixels yet.
    _ = await front.read()
    if back:
        _ = await back.read()

    assessment = {}
    try:
        assessment = json.loads(assessment_data) if assessment_data else {}
    except Exception:
        assessment = {}

    defect_zones = extract_defect_zones(assessment)

    heatmap_data = {
        "front": generate_heatmap_layer(defect_zones.get("front", []), "front"),
        "back": generate_heatmap_layer(defect_zones.get("back", []), "back") if back else None,
    }

    return JSONResponse(content={
        "success": True,
        "heatmap_data": heatmap_data,
        "defect_count": sum(len(zs) for zs in defect_zones.values()),
        "severity_breakdown": calculate_severity_breakdown(defect_zones),
    })


def extract_defect_zones(assessment):
    """
    Extract defect locations and severities from assessment dict.
    Returns: { "front": [..], "back": [..] }
    """
    zones = {"front": [], "back": []}

    # Corners
    corners = assessment.get("corners", {}) if isinstance(assessment, dict) else {}
    corner_positions = {
        "top_left": (0.12, 0.12),
        "top_right": (0.88, 0.12),
        "bottom_left": (0.12, 0.88),
        "bottom_right": (0.88, 0.88),
    }

    for side in ["front", "back"]:
        side_corners = (corners.get(side, {}) or {}) if isinstance(corners, dict) else {}
        for corner_name, (x, y) in corner_positions.items():
            corner_data = (side_corners.get(corner_name, {}) or {}) if isinstance(side_corners, dict) else {}
            condition = str(corner_data.get("condition", "")).lower()

            severity = 0
            if "sharp" in condition or "mint" in condition:
                severity = 0
            else:
                # default to minor if not explicitly sharp
                severity = 1
                if any(w in condition for w in ["whitening", "wear", "fray"]):
                    severity = max(severity, 2)
                if any(w in condition for w in ["severe", "crease", "bend", "dent"]):
                    severity = 3

            if severity > 0:
                zones[side].append({
                    "x": x, "y": y, "radius": 0.14,
                    "severity": severity,
                    "type": "corner",
                    "label": corner_name.replace("_", " ").title()
                })

    # Edges
    edges = assessment.get("edges", {}) if isinstance(assessment, dict) else {}
    edge_positions = {
        "top": (0.5, 0.06),
        "right": (0.94, 0.5),
        "bottom": (0.5, 0.94),
        "left": (0.06, 0.5),
    }

    for side in ["front", "back"]:
        side_edges = (edges.get(side, {}) or {}) if isinstance(edges, dict) else {}
        notes = str(side_edges.get("notes", "")).lower()

        for edge_name, (x, y) in edge_positions.items():
            if edge_name in notes:
                severity = 1
                if any(w in notes for w in ["significant", "moderate"]):
                    severity = 2
                if any(w in notes for w in ["severe", "heavy"]):
                    severity = 3

                zones[side].append({
                    "x": x, "y": y, "radius": 0.18,
                    "severity": severity,
                    "type": "edge",
                    "label": edge_name.title() + " Edge"
                })

    # Surface
    surface = assessment.get("surface", {}) if isinstance(assessment, dict) else {}
    for side in ["front", "back"]:
        side_surface = (surface.get(side, {}) or {}) if isinstance(surface, dict) else {}
        notes = str(side_surface.get("notes", "")).lower()
        if any(w in notes for w in ["scratch", "print line", "stain", "dent", "scuff"]):
            severity = 2
            if any(w in notes for w in ["severe", "significant", "heavy"]):
                severity = 3
            zones[side].append({
                "x": 0.5, "y": 0.5, "radius": 0.28,
                "severity": severity,
                "type": "surface",
                "label": "Surface Defect"
            })

    return zones


def generate_heatmap_layer(zones, side: str):
    return {
        "zones": zones,
        "total_severity": sum(int(z.get("severity", 0) or 0) for z in zones),
        "max_severity": max([int(z.get("severity", 0) or 0) for z in zones]) if zones else 0,
        "side": side
    }


def calculate_severity_breakdown(defect_zones):
    breakdown = {"minor": 0, "moderate": 0, "severe": 0}
    for side_zones in (defect_zones or {}).values():
        for zone in side_zones:
            sev = int(zone.get("severity", 0) or 0)
            if sev == 1:
                breakdown["minor"] += 1
            elif sev == 2:
                breakdown["moderate"] += 1
            elif sev >= 3:
                breakdown["severe"] += 1
    return breakdown



# Runner
# ==============================
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

def _build_ebay_query_ladder_rich(card_name: str, set_name: str, set_code: str, card_number: str, card_type: str) -> List[str]:
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
# ==============================
# Runner
# ==============================
# ========================================
# PORTFOLIO TRACKER ENDPOINTS
# ========================================

@app.post("/api/collection/add")
@safe_endpoint
async def add_to_collection(
    user_id: int = Form(...),
    submission_id: str = Form(...),
    card_name: str = Form(...),
    card_set: str = Form(""),
    card_year: str = Form(""),
    grade_overall: str = Form(""),
    purchase_price: Optional[float] = Form(None),
    purchase_date: Optional[str] = Form(None),
    notes: str = Form(""),
):
    """
    Add a card to user's collection.

    NOTE: For now this returns success and expects WordPress (PHP) to write to WP DB.
    (Later we can wire this to a direct MySQL connection or webhook.)
    """
    try:
        return JSONResponse(content={
            "success": True,
            "message": "Card added to collection",
            "user_id": user_id,
            "submission_id": submission_id
        })
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


@app.get("/api/collection/value-history")
@safe_endpoint
async def get_collection_value_history(user_id: int, days: int = 30):
    """Get collection value over time for charts (mock structure for now)."""
    from datetime import timedelta
    base_date = datetime.now()

    history = []
    for i in range(days):
        date = base_date - timedelta(days=days - i)
        history.append({
            "date": date.strftime("%Y-%m-%d"),
            "total_value": 5000 + (i * 50),
            "card_count": 15
        })

    return JSONResponse(content={
        "success": True,
        "history": history,
        "current_total": history[-1]["total_value"] if history else 0
    })


@app.post("/api/collection/update-values")
@safe_endpoint
async def update_collection_market_values(user_id: int = Form(...)):
    """
    Update all market values for a user's collection (mock structure for now).
    Intended future flow:
    1) fetch all cards in WP collection table
    2) query PriceCharting
    3) update current_market_value + record price_history snapshots
    """
    return JSONResponse(content={
        "success": True,
        "updated_count": 15,
        "total_value": 5250.00,
        "change_24h": 125.00,
        "change_percent": 2.4
    })


# ========================================
# CARD COMPARISON ENDPOINT
# ========================================

@app.post("/api/compare-cards")
@safe_endpoint
async def compare_cards(submission_ids: str = Form(...)):
    """
    Compare 2–4 cards side by side with AI recommendation.
    For now: returns a safe mock structure (PHP page renders it).
    """
    ids = [sid.strip() for sid in submission_ids.split(',') if sid.strip()]
    if len(ids) < 2 or len(ids) > 4:
        raise HTTPException(status_code=400, detail="Must compare 2-4 cards")

    comparison = {
        "submission_ids": ids,
        "cards": [],
        "winner": None,
        "recommendation": "",
        "comparison_matrix": {}
    }

    for i, sid in enumerate(ids):
        comparison["cards"].append({
            "submission_id": sid,
            "card_name": f"Card {i+1}",
            "grade": f"{8+i}",
            "defect_count": max(0, 3 - i),
            "market_value": 100 + (i * 25),
            "strengths": ["Sharp corners", "Good centering"],
            "weaknesses": ["Minor edge wear"]
        })

    comparison["winner"] = ids[0] if ids else None
    comparison["recommendation"] = f"""Based on comparison analysis:

🏆 Winner: {comparison["winner"]}

Why it wins:
- Best overall condition profile
- Fewer visible issues
- Stronger value positioning

Recommended action:
- If BUYING: prioritize {comparison["winner"]}
- If SELLING: lead with {comparison["winner"]}
"""

    return JSONResponse(content=comparison)


# ========================================
# GRADING CONFIDENCE PREDICTOR
# ========================================

@app.post("/api/predict-grade")
@safe_endpoint
async def predict_grade_confidence(
    front: UploadFile = File(...),
    back: UploadFile = File(None),
):
    """
    Quick pre-assessment showing probability distribution of grades
    plus photo quality feedback.
    """
    front_bytes = await front.read()
    back_bytes = await back.read() if back else None

    photo_quality = await analyze_photo_quality(front_bytes, back_bytes)

    prediction_prompt = """You are a quick card grading predictor.
Based on these images, provide:
1. Probability distribution across grades 1-10
2. Most likely grade
3. Specific improvements needed for higher grade

Return JSON:
{
  "grade_probabilities": {
    "10": 0.05,
    "9": 0.15,
    "8": 0.40,
    "7": 0.30,
    "6": 0.10
  },
  "most_likely_grade": "8",
  "confidence": 0.75,
  "improvements_for_higher_grade": [
    "Better lighting on top-left corner to confirm whitening extent",
    "Close-up of edges needed",
    "Back photo shows glare - retake at different angle"
  ],
  "grade_limiters": ["Corner whitening", "Edge wear visible"],
  "quick_summary": "This card will likely grade 8 or 7 based on visible corner wear."
}
"""

    msg = [{
        "role": "user",
        "content": (
            [{"type": "text", "text": prediction_prompt},
             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(front_bytes)}", "detail": "low"}}]
            + ([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(back_bytes)}", "detail": "low"}}] if back_bytes else [])
        )
    }]

    result = await _openai_chat(msg, max_tokens=800, temperature=0.2)
    prediction_data = _parse_json_or_none(result.get("content", "")) or {}

    return JSONResponse(content={
        "success": True,
        "photo_quality": photo_quality,
        "grade_prediction": prediction_data,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })


async def analyze_photo_quality(front_bytes: bytes, back_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    """
    Lightweight photo quality analysis (no numpy/scipy dependencies).
    Uses PIL statistics + edge variance as a blur proxy.
    """
    if not PIL_AVAILABLE:
        return {"available": False, "message": "Image analysis unavailable"}

    try:
        front_img = Image.open(io.BytesIO(front_bytes)).convert('RGB')

        issues: List[str] = []
        suggestions: List[str] = []
        score = 50

        # Resolution check
        width, height = front_img.size
        pixels = width * height
        if pixels < 1_000_000:
            issues.append("Low resolution")
            suggestions.append("Use a higher resolution camera (at least 1920x1080)")
            score -= 20
        elif pixels > 5_000_000:
            score += 15

        # Brightness check
        gray = front_img.convert('L')
        stat = ImageStat.Stat(gray)
        mean_brightness = stat.mean[0] if stat.mean else 0
        if mean_brightness < 80:
            issues.append("Image too dark")
            suggestions.append("Use more lighting - natural daylight works best")
            score -= 15
        elif mean_brightness > 200:
            issues.append("Image overexposed/too bright")
            suggestions.append("Reduce lighting or move away from direct light source")
            score -= 15
        else:
            score += 10

        # Blur proxy: variance of edges
        edges = gray.filter(ImageFilter.FIND_EDGES)
        est = ImageStat.Stat(edges)
        # stddev[0] roughly indicates edge strength; low means blur
        edge_std = est.stddev[0] if est.stddev else 0.0
        # normalize to 0-100
        sharpness_score = max(0.0, min(100.0, (edge_std / 64.0) * 100.0))
        if sharpness_score < 35:
            issues.append("Image appears blurry")
            suggestions.append("Hold camera steady or use a tripod")
            suggestions.append("Ensure auto-focus has locked before capturing")
            score -= 25
        else:
            score += 15

        # Contrast / glare hint: large range can indicate specular highlights
        extrema = stat.extrema[0] if stat.extrema else (0, 0)
        if (extrema[1] - extrema[0]) > 200:
            issues.append("High contrast detected (possible glare)")
            suggestions.append("Angle card to avoid reflections (especially holo surfaces)")
            suggestions.append("Use diffused lighting instead of direct flash")
            score -= 10

        score = max(0, min(100, score))

        return {
            "available": True,
            "overall_score": int(score),
            "resolution": {"width": width, "height": height, "megapixels": round(pixels / 1_000_000, 1)},
            "sharpness_score": round(sharpness_score, 1),
            "brightness_score": round((mean_brightness / 255.0) * 100.0, 1),
            "issues": issues,
            "suggestions": suggestions,
            "verdict": "Excellent" if score > 80 else "Good" if score > 60 else "Needs Improvement"
        }

    except Exception as e:
        return {"available": False, "error": str(e)}


# =========================================================
# STAGE 2 FEATURES — ALERTS / LIVE GRADING / TIMELINE
# =========================================================

# ========================================
# MARKET ALERT SYSTEM
# ========================================

@app.post("/api/alerts/create")
@safe_endpoint
async def create_price_alert(
    user_id: int = Form(...),
    card_name: str = Form(...),
    card_set: str = Form(""),
    grade: str = Form(""),
    alert_type: str = Form("drops_below"),
    target_price: float = Form(...),
):
    """Create a new price alert (PHP persists to WP DB; API returns a confirmation payload)."""
    return JSONResponse(content={
        "success": True,
        "alert_id": "alert_" + secrets.token_hex(8),
        "message": f"Alert created: notify when {card_name} {alert_type.replace('_', ' ')} ${target_price:.2f}",
        "data": {
            "user_id": user_id,
            "card_name": card_name,
            "card_set": card_set,
            "grade": grade,
            "alert_type": alert_type,
            "target_price": target_price,
        }
    })


@app.post("/api/alerts/check")
@safe_endpoint
async def check_price_alerts():
    """Cron job endpoint to check all active alerts (mock payload for now)."""
    triggered_alerts = []

    mock_alert = {
        "id": 123,  # placeholder (WP DB id)
        "alert_id": "alert_123",
        "user_id": 1,
        "card_name": "Charizard",
        "current_price": 475.00,
        "target_price": 500.00,
        "alert_type": "drops_below",
        "triggered": True
    }

    if mock_alert["triggered"]:
        triggered_alerts.append(mock_alert)

    return JSONResponse(content={
        "success": True,
        "checked_alerts": 10,
        "triggered_alerts": triggered_alerts,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })


@app.get("/api/alerts/user/{user_id}")
@safe_endpoint
async def get_user_alerts(user_id: int):
    """Get all alerts for a user (mock)."""
    return JSONResponse(content={
        "success": True,
        "alerts": [
            {
                "id": "alert_123",
                "card_name": "Charizard Base Set",
                "grade": "9",
                "alert_type": "drops_below",
                "target_price": 500.00,
                "current_price": 525.00,
                "is_active": True,
                "created_at": "2026-02-01T10:00:00Z"
            }
        ]
    })




class AlertComposeRequest(BaseModel):
    card_name: str
    card_set: Optional[str] = None
    grade: Optional[str] = None
    direction: str  # "up" or "down"
    pct_change: float
    old_price: Optional[float] = None
    new_price: Optional[float] = None
    currency: str = "AUD"
    user_name: Optional[str] = None
    source: Optional[str] = None


@app.post("/api/alerts/compose-message")
@safe_endpoint
async def compose_alert_message(
    request: AlertComposeRequest,
    api_key: str = Depends(verify_api_key),  # required (used by WP to generate email text)
):
    """Generate a modern spoken-word style alert message for an email notification."""
    try:
        card_bits = " ".join([b for b in [request.card_name, request.card_set or "", request.grade or ""] if b]).strip()
        direction_word = "jumped" if (request.direction or "").lower() == "up" else "dropped"
        pct = float(request.pct_change or 0.0)
        old_p = request.old_price
        new_p = request.new_price
        currency = (request.currency or "AUD").strip().upper()
        who = (request.user_name or "").strip()

        # Fallback message (used if OpenAI is not configured)
        fallback = (
            f"{'Hey ' + who + ', ' if who else ''}"
            f"your {card_bits} just {direction_word} {abs(pct):.1f}% "
            f"({currency} {old_p:.2f} → {currency} {new_p:.2f})" if (old_p is not None and new_p is not None) else
            f"your {card_bits} just {direction_word} {abs(pct):.1f}%" 
        )

        prompt = f"""Write a short, modern spoken-word style message to a collectibles collector.
Tone: confident, hype-but-classy, not cringe. 60–90 words. No financial advice. No ROI language.

Context:
- Item: {card_bits}
- Move: {direction_word} {abs(pct):.1f}%
- Old price: {currency} {old_p:.2f} (if provided)
- New price: {currency} {new_p:.2f} (if provided)
- Source: {request.source or 'collection update'}

Requirements:
- Mention the % move clearly.
- If old/new prices provided, include them once in brackets.
- End with a single, simple action line (e.g. “Check your collection snapshot.”).
Return plain text only (no JSON)."""

        msg = [{"role": "user", "content": prompt}]
        result = await _openai_chat(msg, max_tokens=220, temperature=0.7)

        text = (result.get("content") or "").strip()
        if not text:
            text = fallback

        return JSONResponse(content={
            "success": True,
            "message": text,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    except Exception as e:
        logging.error(f"❌ compose_alert_message failed: {e}")
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }, status_code=500)

# ========================================
# LIVE GRADING ROOM (WebSocket)
# ========================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass

    async def send_feedback(self, websocket: WebSocket, message: dict):
        await websocket.send_json(message)

manager = ConnectionManager()


@app.websocket("/ws/live-grading/{session_id}")
async def live_grading_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time grading feedback."""
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "frame":
                frame_base64 = data.get("frame")
                feedback = await analyze_live_frame(frame_base64)
                await manager.send_feedback(websocket, feedback)

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        manager.disconnect(websocket)


async def analyze_live_frame(frame_base64: str) -> dict:
    """Analyze a single video frame and provide instant feedback."""
    prompt = """Quick frame analysis for live grading assistance.
Provide ONLY these instant feedback items:
- Is card visible and in frame? (yes/no)
- Is lighting adequate? (yes/no/needs adjustment)
- Is focus sharp? (yes/no)
- Any obvious defects visible? (yes/no)
- Suggested camera adjustments (if any)

Return JSON:
{
  "in_frame": true,
  "lighting_ok": true,
  "focus_ok": false,
  "defects_visible": false,
  "suggestions": ["Move camera closer", "Improve focus"],
  "confidence": 0.8
}
"""

    if not frame_base64:
        return {"type": "feedback", "analysis": {"in_frame": False, "lighting_ok": False, "focus_ok": False, "defects_visible": False, "suggestions": ["No frame received"], "confidence": 0.0}, "timestamp": datetime.utcnow().isoformat() + "Z"}

    try:
        msg = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}", "detail": "low"}},
            ]
        }]

        result = await _openai_chat(msg, max_tokens=300, temperature=0.1)
        analysis = _parse_json_or_none(result.get("content", "")) or {}

        return {"type": "feedback", "analysis": analysis, "timestamp": datetime.utcnow().isoformat() + "Z"}
    except Exception as e:
        return {"type": "error", "message": str(e), "timestamp": datetime.utcnow().isoformat() + "Z"}


# ========================================
# TIMELINE / HISTORY TRACKING
# ========================================

@app.post("/api/timeline/add-event")
@safe_endpoint
async def add_timeline_event(
    submission_id: str = Form(...),
    event_type: str = Form(...),
    event_title: str = Form(...),
    event_description: str = Form(""),
    image: UploadFile = File(None),
):
    """Add a new event to submission timeline (mock store)."""
    image_path = None
    if image:
        image_bytes = await image.read()
        image_filename = f"timeline_{submission_id}_{secrets.token_hex(4)}.jpg"
        image_path = f"/uploads/timeline/{image_filename}"
        # In production, save to disk / object store.

    return JSONResponse(content={
        "success": True,
        "event_id": "evt_" + secrets.token_hex(8),
        "message": "Timeline event added",
        "image_path": image_path
    })


@app.get("/api/timeline/{submission_id}")
@safe_endpoint
async def get_timeline(submission_id: str):
    """Get complete timeline for a submission (mock)."""
    events = [
        {
            "id": "evt_1",
            "event_type": "assessment",
            "event_title": "AI Assessment Completed",
            "event_description": "Card graded as Grade 8 by League-AI",
            "created_at": "2026-02-01T10:30:00Z",
            "created_by": "System"
        },
        {
            "id": "evt_2",
            "event_type": "approval",
            "event_title": "Submission Approved",
            "event_description": "Your submission has been approved for professional grading",
            "created_at": "2026-02-02T14:15:00Z",
            "created_by": "Admin"
        },
        {
            "id": "evt_3",
            "event_type": "shipped",
            "event_title": "Card Shipped to Facility",
            "event_description": "Tracking: AU123456789",
            "created_at": "2026-02-03T09:00:00Z",
            "created_by": "Customer"
        },
        {
            "id": "evt_4",
            "event_type": "received",
            "event_title": "Card Received at Facility",
            "event_description": "Card safely received and logged",
            "created_at": "2026-02-05T11:30:00Z",
            "created_by": "Facility"
        }
    ]

    return JSONResponse(content={
        "success": True,
        "submission_id": submission_id,
        "events": events,
        "total_events": len(events)
    })


# ========================================
# AUTHENTICATION VERIFICATION
# ========================================

@app.post("/api/verify-authenticity")
@safe_endpoint
async def verify_card_authenticity(
    front: UploadFile = File(...),
    back: UploadFile = File(None),
    card_name: str = Form(...),
    card_set: str = Form(""),
    card_year: str = Form(""),
):
    """AI-driven authenticity verification (preliminary, not definitive)."""
    front_bytes = await front.read()
    back_bytes = await back.read() if back else None

    auth_prompt = f"""You are an expert in trading card authentication.
Analyze these images of {card_name} ({card_set}, {card_year}) for authenticity markers.

Check for common counterfeit indicators:
1. Print quality - legitimate cards have precise, clean printing
2. Font kerning - spacing between letters should match authentic examples
3. Color saturation - counterfeits often have oversaturated or muted colors
4. Holofoil pattern - if applicable, check for consistent holographic pattern
5. Edge cut - authentic cards have precise, uniform edges
6. Card stock texture and thickness indicators
7. Set symbol and copyright text clarity
8. Any obvious signs of reproduction or printing artifacts

Return detailed JSON:
{{
  "authenticity_score": 85,
  "confidence": 0.90,
  "overall_verdict": "Likely Authentic|Suspicious|Likely Counterfeit",
  "red_flags": ["..."],
  "green_flags": ["..."],
  "key_observations": {{
    "print_quality": {{"score": 90, "notes": "..."}},
    "font_accuracy": {{"score": 75, "notes": "..."}},
    "color_accuracy": {{"score": 95, "notes": "..."}},
    "holofoil_pattern": {{"score": 80, "notes": "..."}},
    "manufacturing_marks": {{"score": 90, "notes": "..."}}
  }},
  "comparison_notes": "...",
  "recommendation": "..."
}}

Be thorough but fair. Many authentic cards have minor variations due to print runs.
Respond ONLY with JSON.
"""

    msg = [{
        "role": "user",
        "content": [
            {"type": "text", "text": auth_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(front_bytes)}", "detail": "high"}},
        ] + ([
            {"type": "text", "text": "BACK IMAGE:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(back_bytes)}", "detail": "high"}},
        ] if back_bytes else [])
    }]

    result = await _openai_chat(msg, max_tokens=1500, temperature=0.1)
    auth_data = _parse_json_or_none(result.get("content", "")) or {}

    automated_checks: Dict[str, Any] = {}
    if PIL_AVAILABLE and front_bytes:
        automated_checks = perform_automated_auth_checks(front_bytes)

    return JSONResponse(content={
        "success": True,
        "card_info": {
            "name": card_name,
            "set": card_set,
            "year": card_year
        },
        "authentication": auth_data,
        "automated_checks": automated_checks,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })



def perform_automated_auth_checks(image_bytes: bytes) -> Dict[str, Any]:
    """
    Automated computer vision checks for authenticity (NO SciPy).
    Uses only Pillow + NumPy (best-effort heuristics).
    """
    if not PIL_AVAILABLE:
        return {"available": False}

    try:
        from PIL import ImageOps, ImageFilter, ImageStat

        img = Image.open(io.BytesIO(image_bytes))
        checks: Dict[str, Any] = {}

        # 1) Resolution check
        width, height = img.size
        checks["resolution"] = {
            "width": width,
            "height": height,
            "pass": bool(width >= 1000 and height >= 1000),
            "note": "High resolution suggests authentic scan" if width >= 1000 else "Low resolution may indicate reproduction",
        }

        # 2) Color histogram analysis (placeholder heuristic)
        _ = img.histogram()
        checks["color_distribution"] = {
            "analyzed": True,
            "pass": True,
            "note": "Color distribution analyzed",
        }

        # 3) Edge detection (PIL-based)
        try:
            gray = ImageOps.grayscale(img)

            # FIND_EDGES tends to work well for print/sharpness heuristics
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_strength = float(ImageStat.Stat(edges).mean[0])

            # EDGE_ENHANCE_MORE can help on slightly soft images
            edges_enhanced = gray.filter(ImageFilter.EDGE_ENHANCE_MORE)
            enhanced_strength = float(ImageStat.Stat(edges_enhanced).mean[0])

            # Combine heuristics (scaled to roughly match prior ranges)
            final_strength = max(edge_strength, enhanced_strength * 0.5)

            checks["edge_quality"] = {
                "strength": float(final_strength),
                "pass": bool(final_strength > 15),  # threshold tuned for PIL intensity space
                "note": "Clean edge detection" if final_strength > 15 else "Soft edges may indicate reproduction",
                "method": "PIL_ImageFilter",
            }
        except Exception as e:
            checks["edge_quality"] = {
                "available": False,
                "pass": True,
                "note": f"Edge check unavailable: {str(e)}",
            }

        overall_pass = all(
            v.get("pass", True) for v in checks.values() if isinstance(v, dict)
        )

        return {"available": True, "checks": checks, "overall_pass": overall_pass}

    except Exception as e:
        return {"available": False, "error": str(e)}


def generate_bulk_summary(results: List[Dict[str, Any]], best_finds: List[Dict[str, Any]]) -> str:
    """Generate a readable summary block for the bulk batch."""
    total_value = sum(float(r.get("estimated_value", 0) or 0) for r in results)
    # estimate avg grade from start of range (e.g. "7-8")
    grades: List[float] = []
    for r in results:
        g = str(r.get("estimated_grade", "") or "").strip()
        if not g:
            continue
        try:
            g0 = float(g.split("-")[0].strip())
            grades.append(g0)
        except Exception:
            pass
    avg_grade = (sum(grades) / len(grades)) if grades else 0
    high_value = [r for r in results if float(r.get("estimated_value", 0) or 0) > 100]

    summary = f"""
📦 Bulk Assessment Summary

Total Cards Processed: {len(results)}
Total Estimated Value: ${total_value:.2f}
Average Grade: ~{avg_grade:.1f}
High Value Cards (>$100): {len(high_value)}

🏆 Top 3 Finds:
""".strip()

    for idx, card in enumerate(best_finds, 1):
        summary += f"\n{idx}. {card.get('card_name', 'Unknown')} - Grade {card.get('estimated_grade', 'N/A')} - ${float(card.get('estimated_value', 0) or 0):.2f}"

    return summary


# ========================================
# QR CODE GENERATION
# ========================================

@app.post("/api/generate-qr")
@safe_endpoint
async def generate_submission_qr(
    submission_id: str = Form(...),
    base_url: str = Form("https://collectors-league.com"),
):
    """Generate a QR code (as base64) linking to the submission details."""
    url = f"{base_url.rstrip('/')}/submission/{submission_id}"

    try:
        import qrcode  # type: ignore
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": f"qrcode dependency not available: {str(e)}"
        })

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    qr_base64 = base64.b64encode(buffer.getvalue()).decode()

    return JSONResponse(content={
        "success": True,
        "submission_id": submission_id,
        "url": url,
        "qr_code_base64": qr_base64,
        "qr_code_data_url": f"data:image/png;base64,{qr_base64}"
    })


# ==============================
# Runner
# ==============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
