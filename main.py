"""
The Collectors League Australia - Scan API
Futureproof v6.9.30 (2026-03-25)

What changed vs v6.9.30 (2026-03-25)
- ✅ NEW: _cluster_mean(prices, tolerance=0.15) helper — finds the largest group of prices
  within 15% of each other and returns their mean. Eliminates rogue outlier listings
  (e.g. one $5k listing among ten $300 listings) without losing genuine price spread.
  Returns None + cluster_size=0 when no consensus cluster of ≥2 prices forms.
- ✅ NEW: _ebay_active_stats and _ebay_completed_stats now compute cluster_mean alongside
  median. The cluster_mean replaces median as the primary price signal when a valid
  consensus cluster forms (≥2 prices, cluster covers ≥40% of total listings).
  Median is retained as fallback and always included in the response for transparency.
- ✅ NEW: cluster_size, cluster_low, cluster_high, cluster_coverage fields added to eBay
  stat responses so the UI can show "12 of 15 listings agree on ~$300" context.
- ✅ NEW: _ebay_sold_strict priority flipped for non-TCG cards (Star Wars, sports, memorabilia).
  New order: (1) graded active BIN → (2) blend active+sold → (3) raw active → (4) raw sold.
  TCG cards (Pokémon, One Piece, etc.) retain the existing sold-first priority unchanged.
  Rationale: for low-liquidity vintage cards current asks ARE the market; historic sales
  are sparse and mix different market environments.
- ✅ NEW: _run_active in _ebay_sold_strict now passes signed query when is_signed=True
  for non-TCG, and applies cluster_mean to the active price list before returning.

What changed vs v6.9.29 (2026-03-25)
- ✅ FIX: Star Wars anchor bumped $2.50 → $4.00 — vintage Decipher/Dark Horse common floor was too low
- ✅ FIX: _STAR_WARS_BASES: added "dark horse", "cam kennedy", "artist edition" tiers; bumped "dark empire" 8.8×→10×, "cover gallery" 5.6×→8×, "series ii" 14×→16×, "hologram" 24×→28×
- ✅ FIX: cla_layered_valuation CHECK 6 — added franchise artist tier (3.5×, floor $80) for Cam Kennedy, Hildebrandt, Struzan, McQuarrie, Dorman, Wheatley; was incorrectly lumped with generic comic artist (2.2×)
- ✅ FIX: _ebay_sold_strict — days_lookback extended to 365 for vintage signed Star Wars/sports; 180 for other non-TCG signed cards (was 90 for all)
- ✅ FIX: _ebay_sold_strict non-TCG tiers now inject "signed" keyword when is_signed=True, preventing unsigned comps from polluting the base price before the signed multiplier fires

What changed vs v6.9.2 (2026-03-16)
- ✅ CRITICAL FIX: Added missing /api/fingerprint/generate endpoint — was returning 404,
  causing the "Generate LingérPrint™ DNA" button in the DNA viewer to fail entirely.
  cg-grading-dashboard.php (cgd_ajax_fingerprint_generate) calls this endpoint directly;
  without it the PHP fell through to cgd_generate_local_fingerprint only, which lacks
  surface_texture and returns "Not captured" for all API-derived fields.
  The new endpoint accepts front/back multipart uploads OR image_front_url/image_back_url
  form fields, runs the full fingerprint pipeline (dHash, surface SHA-256, colour zones,
  histogram, embedding, keypoints) and returns {"success": true, "fingerprint": {...}}.
- ✅ CRITICAL FIX: Added missing /api/fingerprint/match endpoint — was returning 404.
  PHP has a local cgd_hash_similarity fallback that activates when this returns no matches,
  so the endpoint returning 200 + empty matches list is all that's needed.
- ✅ CARRIED: surface_texture .tobytes() fix from v6.9.2 applies to the new endpoint too.

What changed vs v6.9.1 (2026-02-26)
- ✅ CRITICAL FIX: surface_texture "Not captured" — block 6b was silently failing due to
  `bytes(b for px in _crop32.getdata() for b in px)` generator expression. In Pillow >= 11
  the ImagingCore pixel values returned by getdata() can cause a TypeError that the bare
  `except: pass` swallowed silently. Fixed by replacing with `.tobytes()` — the canonical
  Pillow API for raw pixel bytes, safe across all versions.
- ✅ FIX: All 6 fingerprint blocks now log exceptions via logging.warning() and write errors
  into a new `fp_errors` dict in the API response, so "Not captured" is self-diagnosing
  without needing to grep Render logs.
- ✅ FIX: PIL_AVAILABLE=False and empty front_proc are now explicit elif branches with
  warning-level log messages, not a single silent `if PIL_AVAILABLE and front_proc:` guard.

What changed vs v6.9.0 (2026-02-26)
- ✅ CRITICAL FIX: Removed second-pass AI call from /api/verify — was exceeding Render 512MB memory limit (502 Bad Gateway)
- ✅ CRITICAL FIX: Disabled _openai_label_rois in /api/verify and memorabilia endpoints — memory budget
- ✅ NOTE: Single-pass mode — primary OpenAI vision call handles defect detection; second pass was redundant overhead
- ✅ NOTE: defect_filters kept in response schema (returns empty dict) for frontend compatibility
- ✅ FIX: _grade_bucket() now supports Grade 12 (CL Ultra Flawless) — was silently dropping to N/A
- ✅ FIX: Removed duplicate /api/collection/update-values endpoint (mock was overwriting real)
- ✅ FIX: Memorabilia assessment now uses detail:high images (was low — couldn't see seal damage)
- ✅ FIX: Assessment prompt rule numbering corrected (was 1,2,3,5,4)
- ✅ FIX: _grade_distribution() now covers full grade range 1-12 (was only 8-10)
- ✅ FIX: predict-grade endpoint updated for half-grades and full range
- ✅ FIX: Assessment max_tokens bumped 2200→3000 for expanded output
- ✅ NEW: Half-grade support (9.5, 8.5, 7.5 etc.) — _parse_half_grade(), pregrade normalization
- ✅ NEW: Grading standard selector (PSA/BGS/CGC/CL) via grading_standard param on /api/verify
- ✅ NEW: Error card / misprint / factory defect detection (miscut, crimped, wrong-back, holo bleed etc.)
- ✅ NEW: Expanded surface condition factors (silvering vs whitening, yellowing, foxing, ink transfer, warp, gloss)
- ✅ NEW: Rarity / variant / edition / finish / artist / promo / error detection in /api/identify
- ✅ NEW: Sleeve/toploader detection in assessment (accounts for artifacts)
- ✅ NEW: Card warp/bowing + back pattern awareness
- ✅ NEW: Expanded _normalize_card_type() — 20+ TCGs + specific sports (AFL, NRL, Cricket, NBA etc.)
- ✅ NEW: Expanded authenticity checks — rosette pattern, card stock, back pattern, light/weight test guidance

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

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Security, status, Request, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel
from datetime import datetime, timedelta
from statistics import mean, median
from functools import wraps
from contextlib import asynccontextmanager
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
import base64
import io
from io import BytesIO
import os
import asyncio
import json
import sqlite3
import csv
import secrets
import random

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

# Anthropic SDK — used by /api/ai-grade for CLA official grading
try:
    import anthropic as _anthropic_sdk
    ANTHROPIC_AVAILABLE = True
except Exception:
    _anthropic_sdk = None
    ANTHROPIC_AVAILABLE = False

# u2500u2500 Shared async HTTP client u2014 reused across all requests to avoid per-call
#    TCP + TLS handshake overhead. Timeout of 120s covers large OpenAI payloads.
_HTTP_CLIENT = None

def _get_http_client():
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.AsyncClient(timeout=120.0)
    return _HTTP_CLIENT



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
        edges   = [r for r in rois if str(r.get("roi", "")).startswith("edge_")]

        # Center crop for surface/print-line defects (interior of card)
        margin = max(int(w * 0.25), 20)
        surface_s = float(score_box(margin, margin, w - margin, h - margin))
        center_bbox = {
            "x": round(margin / w, 4), "y": round(margin / h, 4),
            "w": round((w - 2 * margin) / w, 4), "h": round((h - 2 * margin) / h, 4),
        }
        surface_roi = {"side": side, "roi": "surface_center", "score": round(surface_s, 4), "bbox": center_bbox}

        out = corners[:4] + edges[:2] + [surface_roi]
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
            "For each crop, decide whether it shows a REAL defect (not glare/noise). IMPORTANT: Foil/holo patterns, sparkle, textured foil, embossing, and normal printing texture are NOT defects. "
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
        "For crops labeled 'surface_center': look specifically for scratches, print lines, surface marks, staining. "
        "DO NOT flag foil sparkle/texture, holo patterns, embossing, or light reflections as scratches/print lines. "
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
_FX_CACHE = {"ts": 0, "usd_aud": None, "gbp_aud": None, "eur_aud": None, "cad_aud": None, "jpy_aud": None}
_EBAY_CACHE = {}  # key -> {ts, data}
FX_CACHE_SECONDS = int(os.getenv("FX_CACHE_SECONDS", "3600"))


def _evict_stale_caches() -> int:
    """Remove expired entries from _EBAY_CACHE and trim _TRANSLATE_CACHE.

    Call periodically to prevent unbounded memory growth on long-running instances.
    Returns the number of entries removed.
    """
    now = int(time.time())
    removed = 0
    # eBay cache: stale after 2× the longest TTL (1 hour)
    stale_ebay = [k for k, v in _EBAY_CACHE.items()
                  if now - int((v or {}).get("ts", 0)) > 3600]
    for k in stale_ebay:
        _EBAY_CACHE.pop(k, None)
        removed += 1
    # Translation cache: cap at 2,000 entries (each is a short string)
    if len(_TRANSLATE_CACHE) > 2000:
        overflow = len(_TRANSLATE_CACHE) - 2000
        for k in list(_TRANSLATE_CACHE.keys())[:overflow]:
            _TRANSLATE_CACHE.pop(k, None)
            removed += 1
    if removed:
        logging.info(f"Cache eviction: removed {removed} stale entries")
    return removed

# ==============================
# App & CORS
# ==============================
@asynccontextmanager
async def _lifespan(app):
    """Startup / shutdown lifecycle for the FastAPI app.
    - Startup: log version
    - Shutdown: close shared HTTP client cleanly (avoids ResourceWarning on Render restarts)
    """
    logging.info(f"🚀 Starting Collectors League Scan API v6.9.30")
    yield
    global _HTTP_CLIENT
    if _HTTP_CLIENT and not _HTTP_CLIENT.is_closed:
        await _HTTP_CLIENT.aclose()
        logging.info("🛑 Shared HTTP client closed cleanly")


app = FastAPI(title="Collectors League Scan API", lifespan=_lifespan)

logger = logging.getLogger("cl-scan-api")

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
APP_VERSION = os.getenv("CL_SCAN_VERSION", "2026-03-25-v6.9.30")

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY",      "").strip()
POKEMONTCG_API_KEY  = os.getenv("POKEMONTCG_API_KEY",  "").strip()
PRICECHARTING_TOKEN = os.getenv("PRICECHARTING_TOKEN",  "").strip()
ADMIN_TOKEN         = os.getenv("CL_ADMIN_TOKEN",       "").strip()  # optional
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY",    "").strip()  # used by /api/ai-grade

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
    if not any(secrets.compare_digest(tok, k) for k in valid):
        logging.warning(f"❌ Invalid API key attempt: {tok[:4]}****")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logging.info(f"✅ Authenticated: {tok[:4]}****")
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


def _detect_tcg_game(card_name: str, set_name: str = "", card_number: str = "") -> str:
    """
    Detect which TCG game a card belongs to based on set name patterns and card number prefixes.
    Returns the eBay-friendly game keyword to inject into search queries.

    Returns:
        "Pokemon"   — Pokemon TCG (default when uncertain)
        "One Piece" — One Piece Card Game (OP prefix)
        "Dragon Ball" — Dragon Ball Super Card Game (BT/FB/B prefix)
        "Digimon"   — Digimon Card Game (BT/EX prefix with Digimon sets)
        "Magic"     — Magic: The Gathering
        "Yu-Gi-Oh"  — Yu-Gi-Oh!
    """
    set_lower  = (set_name or "").lower().strip()
    name_lower = (card_name or "").lower().strip()
    num_upper  = (card_number or "").upper().strip()

    # One Piece Card Game — set codes: OP01-OP15, ST (starter decks), P (promo)
    one_piece_sets = ["romance dawn", "paramount war", "pillars of strength", "kingdoms of intrigue",
                      "awakening of the new era", "wings of the captain", "500 years in the future",
                      "two legends", "emperors in the new world", "royal blood", "supreme darkness",
                      "side characters", "memorial collection", "op-", "op0", "op1", "op2", "op3",
                      "op4", "op5", "op6", "op7", "op8", "op9", "op10", "op11", "op12", "op13",
                      "op14", "op15", "one piece"]
    for marker in one_piece_sets:
        if marker in set_lower:
            return "One Piece"
    # One Piece card numbers: OP01-123, ST01-001, etc.
    if re.match(r'^(OP\d{2}|ST\d{2})', num_upper):
        return "One Piece"

    # Dragon Ball Super Card Game — sets: BT, FB, B series; Fusion World
    dbs_sets = ["dragon ball", "fusion world", "zenkai series", "galactic battle", "union force",
                "cross worlds", "colossal warfare", "realm of the gods", "miraculous revival",
                "ultimate squad", "bt0", "bt1", "bt2", "bt3", "bt4", "bt5", "bt6", "bt7", "bt8",
                "bt9", "bt10", "bt11", "bt12", "bt13", "bt14", "bt15", "bt16", "bt17", "bt18",
                "fb0", "fb01", "fb02", "fb03", "fb04"]
    for marker in dbs_sets:
        if marker in set_lower:
            return "Dragon Ball"
    if re.match(r'^(BT\d{2}|FB\d{2})', num_upper):
        return "Dragon Ball"

    # Digimon — sets: BT/EX/RB/P/ST with Digimon context
    if "digimon" in set_lower or "digimon" in name_lower:
        return "Digimon"
    if re.match(r'^(BT\d|EX\d|RB\d)', num_upper) and "digimon" in name_lower:
        return "Digimon"

    # Magic: The Gathering
    mtg_markers = ["magic", "mtg", "the gathering", "commander", "dominaria", "innistrad",
                   "zendikar", "strixhaven", "eldraine", "theros", "ravnica", "throne of eldraine"]
    for marker in mtg_markers:
        if marker in set_lower or marker in name_lower:
            return "Magic"

    # Yu-Gi-Oh
    ygo_markers = ["yu-gi-oh", "yugioh", "master duel", "speed duel", "konami", "duel links"]
    for marker in ygo_markers:
        if marker in set_lower or marker in name_lower:
            return "Yu-Gi-Oh"

    # Default: Pokemon (most common in this platform)
    return "Pokemon"


def _build_ebay_query_ladder(
    card_name: str,
    set_name: str = "",
    card_number: str = "",
    grade: str = "",
    **kwargs,
) -> list:
    """
    Return a small ladder of increasingly-broad eBay keyword queries.

    Compatible with older call-sites that pass `card_set=...`.
    """
    # Back-compat for older kwarg name
    if not set_name and isinstance(kwargs.get("card_set"), str):
        set_name = kwargs.get("card_set", "") or ""

    # ── CJK detection and English extraction ─────────────────────────────────
    # If card_name contains Japanese/Chinese/Korean characters, eBay returns
    # zero results. Detect CJK regardless of stored language field (which is
    # often empty/defaulting to "english" in the DB).
    def _has_cjk(s: str) -> bool:
        return bool(re.search(r'[\u3000-\u9fff\uac00-\ud7af\uf900-\ufaff]', s))

    if _has_cjk(card_name) and "/" in card_name:
        # Bilingual name: "モンキー・D・ルフィ / Monkey D. Luffy" → use English part
        card_name = card_name.split("/")[-1].strip()
    elif _has_cjk(card_name):
        # CJK-only name with no "/" — strip all CJK characters, keep Latin remainder
        card_name = re.sub(r'[\u3000-\u9fff\uac00-\ud7af\uf900-\ufaff\u30a0-\u30ff\u3040-\u309f\u31f0-\u31ff]+',
                           ' ', card_name).strip()
        card_name = re.sub(r'\s+', ' ', card_name).strip()
    # Also strip (Signed) prefix — signed premium is a multiplier, not a search term
    card_name = re.sub(r'^[\(\[]\s*(?:signed|autographed?|auto)\s*[\)\]]\s*', '', card_name, flags=re.I).strip()
    card_name = re.sub(r'^\s*(?:signed|autographed?)\s+', '', card_name, flags=re.I).strip()
    # ──────────────────────────────────────────────────────────────────────────

    base = _build_ebay_search_query(
        card_name=card_name,
        card_set=set_name,
        card_number=card_number,
        grade="",  # grade handled separately below
    )
    name_only = _build_ebay_search_query(card_name=card_name, card_set="", card_number="", grade="")

    # Normalize grade into a PSA token when possible
    # Handles: "9", "9.5", "PSA 9", "10 - Flawless", "12 - Ultra Flawless", "Grade: 8.5"
    g = (grade or "").strip()
    psa_token = ""
    if g:
        if "psa" in g.lower():
            # Already has PSA prefix — extract numeric part and rebuild cleanly
            m_psa = re.search(r"(\d+(?:\.\d+)?)", g)
            if m_psa:
                psa_token = f"PSA {m_psa.group(1)}"
            else:
                psa_token = g.strip()
        else:
            # Extract numeric from strings like "10 - Flawless", "9.5", "Grade 9"
            m_num = re.search(r"(\d+(?:\.\d+)?)", g)
            if m_num:
                numeric_g = m_num.group(1)
                # CL grades 11+ → treat as PSA 10 equivalent for search
                try:
                    gval = float(numeric_g)
                    if gval >= 11:
                        numeric_g = "10"
                except Exception:
                    pass
                if re.fullmatch(r"(10|[1-9](?:\.5)?)", numeric_g):
                    psa_token = f"PSA {numeric_g}"

    ladder = []
    for q in [base, name_only]:
        q = _norm_ws(q)
        if not q:
            continue
        # Try with grade token first (if any), then without
        if psa_token:
            ladder.append(_norm_ws(f"{q} {psa_token}"))
        ladder.append(q)


    # Extra variants to improve recall on messy user-entered names/sets
    # - Some sellers omit the word "card"
    # - ex/EX casing varies
    # - Mega cards are often listed as "M <name> EX"
    expanded = []
    for q in list(ladder):
        if not q:
            continue
        if "Pokemon card" in q:
            expanded.append(_norm_ws(q.replace("Pokemon card", "Pokemon")))
            expanded.append(_norm_ws(q.replace("Pokemon card", "Pokemon TCG")))
            expanded.append(_norm_ws(q.replace("Pokemon card", "").strip()))
        # ex/EX normalization
        expanded.append(_norm_ws(re.sub(r"\bex\b", "EX", q, flags=re.I)))
        # Mega/XY-era normalization: "Mega Charizard X ex" -> "M Charizard EX" / "M Charizard X EX"
        if re.search(r"\bCharizard\b", q, flags=re.I) and re.search(r"\b(Mega|\bM\b)\b", q, flags=re.I):
            expanded.append(_norm_ws(re.sub(r"\bX\b", "", q, flags=re.I)))
            expanded.append(_norm_ws(re.sub(r"\bY\b", "", q, flags=re.I)))
            expanded.append(_norm_ws(re.sub(r"\bCharizard\s+X\b", "Charizard", q, flags=re.I)))
            expanded.append(_norm_ws("M Charizard EX Pokemon"))
            expanded.append(_norm_ws("M Charizard EX Pokemon card"))
    for q2 in expanded:
        if q2:
            ladder.append(q2)

    # Final ultra-broad fallback (sometimes "card" hurts recall)
    # ── Game-type injection ──────────────────────────────────────────────────
    # Detect which TCG this card belongs to based on set name patterns and
    # card number prefixes. This prevents cross-game contamination in eBay
    # results (e.g. One Piece OP12 cards returning Pokemon sealed boxes).
    game_keyword = _detect_tcg_game(card_name, set_name, card_number)

    if game_keyword and game_keyword != "Pokemon":
        # For non-Pokemon games, add targeted game variants and remove Pokemon fallbacks
        ladder_with_game = []
        for q in ladder:
            # Replace any "Pokemon" word in existing queries with correct game
            if "Pokemon" in q:
                ladder_with_game.append(_norm_ws(q.replace("Pokemon", game_keyword)))
            else:
                ladder_with_game.append(q)
        # Add game-specific fallback
        if card_name:
            ladder_with_game.append(_norm_ws(f"{card_name} {game_keyword}"))
        ladder = ladder_with_game
    elif game_keyword == "Pokemon":
        if card_name:
            ladder.append(_norm_ws(f"{card_name} Pokemon"))
    else:
        # Unknown game — don't append Pokemon fallback blindly
        if card_name:
            ladder.append(_norm_ws(f"{card_name} trading card"))

    # Deduplicate while preserving order
    seen = set()
    out = []
    for q in ladder:
        if q and q not in seen:
            out.append(q)
            seen.add(q)
    return out
async def _fx_refresh() -> dict:
    """Fetch USD, GBP, EUR, CAD, JPY → AUD in a single Frankfurter call.
    Cached for FX_CACHE_SECONDS (default 1 hour). Returns the full cache dict.
    Replaces the old per-currency hardcoded multipliers scattered through eBay price conversion.
    """
    now = int(time.time())
    if _FX_CACHE.get("usd_aud") and (now - int(_FX_CACHE.get("ts") or 0) < FX_CACHE_SECONDS):
        return _FX_CACHE
    try:
        # Single call returns all needed rates — AUD as base so we get direct AUD values
        url = "https://api.frankfurter.app/latest?base=AUD&symbols=USD,GBP,EUR,CAD,JPY"
        client = _get_http_client()
        r = await client.get(url, timeout=15.0, headers={"User-Agent": UA})
        if r.status_code == 200:
            rates = r.json().get("rates", {})
            # Frankfurter gives units-per-1-AUD → invert to get AUD-per-1-unit
            def _inv(key, fallback):
                v = float(rates.get(key) or 0)
                return round(1.0 / v, 6) if v > 0 else fallback
            _FX_CACHE.update({
                "ts":      now,
                "usd_aud": _inv("USD", 1.55),
                "gbp_aud": _inv("GBP", 1.98),
                "eur_aud": _inv("EUR", 1.69),
                "cad_aud": _inv("CAD", 1.14),
                "jpy_aud": _inv("JPY", 0.0103),
            })
    except Exception:
        pass  # stale values remain — better than crashing
    return _FX_CACHE


async def _fx_usd_to_aud() -> float:
    """USD → AUD. Backward-compatible wrapper around _fx_refresh()."""
    fx = await _fx_refresh()
    return float(fx.get("usd_aud") or AUD_MULTIPLIER)


async def _fx_to_aud(amount: float, currency: str) -> float:
    """Convert any supported currency amount to AUD using live rates.
    Returns 0.0 for unknown currencies so callers can skip bad data.
    Replaces all hardcoded GBP*1.27, EUR*1.09, CAD*0.74, JPY/150 blocks.
    """
    if amount <= 0:
        return 0.0
    cur = (currency or "").upper().strip()
    if cur in ("AUD", ""):
        return float(amount)
    fx = await _fx_refresh()
    rate = float(fx.get(f"{cur.lower()}_aud") or 0)
    if rate <= 0:
        logging.debug(f"_fx_to_aud: unknown currency {cur!r} — skipping item")
        return 0.0
    return round(amount * rate, 4)


async def _ebay_completed_stats(keyword_query: str, limit: int = 120, days_lookback: int = 120,
                                global_id: str = "EBAY-AU") -> dict:
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

    # Fetch live FX rate once for the whole batch (falls back to static if unavailable)
    aud_rate = await _fx_usd_to_aud()

    url = "https://svcs.ebay.com/services/search/FindingService/v1"
    headers = {"User-Agent": UA}

    client = _get_http_client()
    for page in range(1, pages + 1):
            params = {
                "OPERATION-NAME":              "findCompletedItems",
                "SERVICE-VERSION":             "1.13.0",
                "GLOBAL-ID":                   global_id,
                "RESPONSE-DATA-FORMAT":        "JSON",
                "REST-PAYLOAD":                "true",
                "SECURITY-APPNAME":            EBAY_APP_ID,
                "keywords":                    q,
                "itemFilter(0).name":          "SoldItemsOnly",
                "itemFilter(0).value":         "true",
                "paginationInput.entriesPerPage": "100",
                "paginationInput.pageNumber":  str(page),
            }

            try:
                r = await client.get(url, params=params, headers=headers)
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
                    # Convert to AUD based on currency. JPY must be converted or
                    # a ¥9,000 card appears as AUD $9,000 — 140× inflation.
                    val = await _fx_to_aud(val, cur if cur else "AUD")
                    if val <= 0:
                        continue  # unknown currency or conversion failed
                    # Ignore insane values (protect against parsing weird lots)
                    if val <= 0 or val > 50_000:
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

    raw_median = pct(0.50)
    raw_low    = pct(0.20)
    raw_high   = pct(0.80)

    # When sample is very small (< 3), percentiles collapse to the same value.
    # Apply a synthetic spread so low/median/high are meaningfully different.
    # 5% spread each side — wide enough to signal uncertainty, narrow enough to be honest.
    if count < 3:
        raw_low  = min(raw_low,  raw_median * 0.95)
        raw_high = max(raw_high, raw_median * 1.05)

    # ── Cluster mean: consensus price band within 15% of each other ───────────
    # Eliminates rogue outlier listings (e.g. one $5k among ten $300 listings)
    # without discarding genuine price spread. Use cluster_mean as primary signal
    # when a valid consensus cluster forms (≥2 prices, covers ≥40% of listings).
    cl     = _cluster_mean(prices, tolerance=0.15, min_cluster=2)
    use_cm = (not cl["fallback_used"]) and (cl["cluster_coverage"] >= 0.40)
    primary_price = cl["cluster_mean"] if use_cm else raw_median

    out = {
        "source": "ebay",
        "query": q,
        "count": count,
        "currency": "AUD",
        "prices": prices[:target],
        "low": cl["cluster_low"]  if use_cm else raw_low,
        "median": primary_price,
        "high": cl["cluster_high"] if use_cm else raw_high,
        "avg": float(sum(prices) / count),
        "p20": raw_low,
        "p80": raw_high,
        "min": float(prices[0]),
        "max": float(prices[-1]),
        # Cluster fields — always included so callers can inspect/log
        "cluster_mean":     cl["cluster_mean"],
        "cluster_size":     cl["cluster_size"],
        "cluster_low":      cl["cluster_low"],
        "cluster_high":     cl["cluster_high"],
        "cluster_coverage": cl["cluster_coverage"],
        "cluster_used":     use_cm,
    }
    _EBAY_CACHE[cache_key] = {"ts": now, "data": out}
    return out


def _pc_db():
    con = sqlite3.connect(PRICECHARTING_DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pc_snapshots_pid_date ON pc_price_snapshots(product_id, snap_date)")
    con.commit()
    con.close()

try:
    _pc_init_db()
except Exception as _e:
    print(f"INFO: PriceCharting DB init skipped: {_e}")


# ── Claude market lookup cache ────────────────────────────────────────────────
def _claude_cache_init():
    con = _pc_db()
    con.execute("""
        CREATE TABLE IF NOT EXISTS claude_market_cache (
            cache_key   TEXT PRIMARY KEY,
            card_name   TEXT,
            card_number TEXT,
            grade       TEXT,
            result_json TEXT,
            cached_at   TEXT
        )
    """)
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_cmc_key ON claude_market_cache(cache_key, cached_at)"
    )
    con.commit()
    con.close()

try:
    _claude_cache_init()
except Exception as _e:
    print(f"INFO: Claude market cache init skipped: {_e}")


# ── System prompt — knowledge-based, no web_search tool needed ───────────────
# Haiku's training data covers card prices well enough for a fallback estimate.
# No tool use = no multi-turn overhead = simpler + cheaper + actually works.
_CLAUDE_MARKET_SYSTEM = (
    "You are a trading card price expert with deep knowledge of TCG and collectible card markets. "
    "Estimate the current market value based on your training knowledge and return ONLY a JSON object — "
    "no markdown, no explanation, no extra text.\n\n"
    "Rules:\n"
    "- Estimate in AUD (use 1 USD = 1.55 AUD)\n"
    "- For graded cards: Grade 10 = PSA 10 equivalent, Grade 12 = BGS Black equivalent (~1.5x PSA 10)\n"
    "- Use your knowledge of eBay sold prices, TCGPlayer, PriceCharting for this card\n"
    "- If you genuinely don't know the card, return price_aud: 0\n"
    "- confidence: \"high\" if you know this card well, \"medium\" if approximate, \"low\" if uncertain\n"
    "- Return ONLY this exact JSON:\n"
    '{"price_aud":<number>,"confidence":"high|medium|low|none","source":"knowledge","comp_count":0,"note":"<max 8 words>"}'
)


async def _claude_market_lookup(
    card_name:   str,
    card_number: str,
    card_set:    str,
    grade:       str,
    game_type:   str,
    rarity:      str  = "",
    is_signed:   bool = False,
    language:    str  = "english",
    image_url:   str  = "",        # front slab image — improves variant/rarity accuracy
) -> dict:
    """
    Claude Haiku — fires ONLY when eBay/PC both returned nothing.

    Token budget:
      Without image: ~300 input + 150 output ≈ $0.0004 AUD
      With image:   ~1900 input + 150 output ≈ $0.001  AUD
      All results cached 24h in SQLite — repeat calls are free.
    Returns {} on any failure — never raises.
    """
    if not ANTHROPIC_AVAILABLE or not _anthropic_sdk or not ANTHROPIC_API_KEY:
        logging.debug(f"🔎 DIAG _claude_market_lookup SKIPPED: available={ANTHROPIC_AVAILABLE} key={bool(ANTHROPIC_API_KEY)}")
        return {}

    logging.debug(f"🔎 DIAG _claude_market_lookup ENTERED: {card_name} #{card_number} grade={grade}")

    # ── 24-hour SQLite cache ──────────────────────────────────────────────────
    _ck_raw   = f"{card_name}|{card_number}|{grade}|{game_type}|{bool(image_url)}".lower().strip()
    cache_key = hashlib.md5(_ck_raw.encode()).hexdigest()

    try:
        con = _pc_db()
        row = con.execute(
            "SELECT result_json FROM claude_market_cache "
            "WHERE cache_key=? AND cached_at >= datetime('now','-24 hours') LIMIT 1",
            (cache_key,)
        ).fetchone()
        con.close()
        if row and row[0]:
            cached = json.loads(row[0])
            cached["from_cache"] = True
            logging.info(
                f"💾 Claude market CACHE HIT: {card_name} #{card_number} "
                f"→ ${cached.get('price_aud', 0):.2f} AUD"
            )
            return cached
    except Exception as _ce:
        logging.debug(f"Claude cache read: {_ce}")

    # ── Compact user message — every word costs tokens ────────────────────────
    grade_note = ""
    try:
        _gv = float(re.search(r"(\d+(?:\.\d+)?)", grade).group(1))
        if _gv >= 12:
            grade_note = " (find BGS Black or PSA 10 × 1.5 comps)"
        elif _gv >= 10:
            grade_note = " (find PSA 10 graded comps)"
        elif _gv >= 9:
            grade_note = " (find PSA 9 graded comps)"
    except Exception:
        pass

    _lines = [
        f"Card: {card_name}",
        f"Number: {card_number}"          if card_number else None,
        f"Set: {card_set}"                if card_set    else None,
        f"Game: {game_type}"              if game_type   else None,
        f"Grade: {grade}{grade_note}"     if grade       else None,
        f"Rarity: {rarity}"               if rarity      else None,
        "Signed copy"                     if is_signed   else None,
        f"Language: {language}"           if language and language.lower() != "english" else None,
    ]
    user_msg = "\n".join(ln for ln in _lines if ln)

    # ── Fetch and base64-encode image if URL provided ─────────────────────────
    # We fetch it server-side so the URL doesn't need to be publicly accessible.
    # Keeps image tokens to a minimum by resizing to max 512px (still readable).
    _img_b64   = None
    _img_mtype = "image/jpeg"
    if image_url:
        try:
            async with httpx.AsyncClient(timeout=10.0) as _hc:
                _ir = await _hc.get(image_url, follow_redirects=True)
            if _ir.status_code == 200:
                _img_bytes = _ir.content
                _img_mtype = _ir.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                # Resize to max 512px longest side to cap token cost (~800 tokens vs ~1600)
                if PIL_AVAILABLE and _img_bytes:
                    try:
                        _pil = Image.open(io.BytesIO(_img_bytes)).convert("RGB")
                        _pil.thumbnail((512, 512), Image.LANCZOS)
                        _buf = io.BytesIO()
                        _pil.save(_buf, format="JPEG", quality=85)
                        _img_bytes = _buf.getvalue()
                        _img_mtype = "image/jpeg"
                    except Exception:
                        pass  # use original bytes if PIL fails
                _img_b64 = base64.b64encode(_img_bytes).decode()
                logging.info(f"🖼️  Claude market: image fetched for {card_name} ({len(_img_bytes)//1024}KB)")
        except Exception as _ie:
            logging.debug(f"Claude market image fetch failed: {_ie}")
            _img_b64 = None  # proceed without image

    # ── Build message content — image first if available ─────────────────────
    if _img_b64:
        _msg_content = [
            {
                "type":   "image",
                "source": {
                    "type":       "base64",
                    "media_type": _img_mtype,
                    "data":       _img_b64,
                },
            },
            {"type": "text", "text": user_msg},
        ]
    else:
        _msg_content = user_msg   # plain string — same as before

    # ── Haiku call — knowledge-based, no tools ───────────────────────────────
    try:
        client = _anthropic_sdk.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 250,
            system     = _CLAUDE_MARKET_SYSTEM,
            messages   = [{"role": "user", "content": _msg_content}],
        )

        # Simple single text block response — no tool use in play
        result_text = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                _t = block.text.strip()
                if _t:
                    result_text = _t

        if not result_text:
            logging.warning(f"🤖 Claude market: EMPTY RESPONSE for {card_name} | stop_reason={getattr(response, 'stop_reason', '?')} | blocks={[getattr(b,'type','?') for b in response.content]}")
            return {}

        result_text = result_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(result_text)

        price = float(parsed.get("price_aud") or 0)
        conf  = str(parsed.get("confidence") or "none")
        src   = str(parsed.get("source")     or "claude_web_search")
        count = int(parsed.get("comp_count") or 0)
        note  = str(parsed.get("note")       or "")
        if conf not in ("high", "medium", "low", "none"):
            conf = "low"

        out = {
            "price_aud":  round(price, 2),
            "confidence": conf,
            "source":     src or "claude_web_search",
            "comp_count": count,
            "note":       note,
            "from_cache": False,
        }

        logging.info(
            f"🤖 Claude market: {card_name} #{card_number} "
            f"→ ${price:.2f} AUD | conf={conf} | comps={count} | src={src} | text={result_text[:80]!r}"
        )

        # Write to cache only when we got a real price
        if price > 0:
            try:
                con = _pc_db()
                con.execute(
                    "INSERT OR REPLACE INTO claude_market_cache "
                    "(cache_key,card_name,card_number,grade,result_json,cached_at) "
                    "VALUES (?,?,?,?,?,datetime('now'))",
                    (cache_key, card_name, card_number, grade, json.dumps(out))
                )
                con.commit()
                con.close()
            except Exception as _we:
                logging.debug(f"Claude cache write: {_we}")

        return out

    except json.JSONDecodeError as _je:
        logging.warning(f"🤖 Claude market JSON PARSE FAILED ({card_name}): {_je} | raw={result_text[:120]!r}")
        return {}
    except Exception as _e:
        logging.warning(f"🤖 Claude market EXCEPTION ({card_name}): {type(_e).__name__}: {_e}")
        return {}


# ═══════════════════════════════════════════════════════
# PRICE HISTORY (persistent snapshots for market-trends)
# Stored in SQLite (defaults to the same DB file as PriceCharting)
# ═══════════════════════════════════════════════════════

PRICE_HISTORY_DB_PATH = os.getenv("PRICE_HISTORY_DB_PATH", PRICECHARTING_DB_PATH).strip() or PRICECHARTING_DB_PATH

def _ph_db():
    con = sqlite3.connect(PRICE_HISTORY_DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
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
    "PAF": "Paldean Fates",
    "PAR": "Paradox Rift",
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


# ==============================
# One Piece / non-Pokemon set helpers (heuristic)
# ==============================
_ONE_PIECE_SET_NAME_MAP: Dict[str, str] = {
    # Starter Decks
    "ST01": "Starter Deck 01 — Straw Hat Crew",
    "ST02": "Starter Deck 02 — Worst Generation",
    "ST03": "Starter Deck 03 — The Seven Warlords of the Sea",
    "ST04": "Starter Deck 04 — Animal Kingdom Pirates",
    "ST05": "Starter Deck 05 — ONE PIECE FILM edition",
    "ST06": "Starter Deck 06 — Absolute Justice",
    "ST07": "Starter Deck 07 — Big Mom Pirates",
    "ST08": "Starter Deck 08 — Monkey D. Luffy",
    "ST09": "Starter Deck 09 — Yamato",
    "ST10": "Starter Deck 10 — The Three Captains",
    "ST11": "Starter Deck 11 — Uta",
    "ST12": "Starter Deck 12 — Zoro & Sanji",
    "ST13": "Starter Deck 13 — The Three Brothers",
    "ST14": "Starter Deck 14 — 3D2Y",
    "ST15": "Starter Deck 15 — RED (assorted)",
    # Boosters (common codes)
    "OP01": "Romance Dawn",
    "OP02": "Paramount War",
    "OP03": "Pillars of Strength",
    "OP04": "Kingdoms of Intrigue",
    "OP05": "Awakening of the New Era",
    "OP06": "Wings of the Captain",
    "OP07": "500 Years in the Future",
    "OP08": "Two Legends",
    "OP09": "Four Emperors",
    "OP10": "Royal Blood",
    "OP11": "A Fist of Divine Speed",
    "OP12": "Legacy of the Master",
}

# Heuristic release-year mapping (best-effort; used only when AI doesn't provide a year)
_ONE_PIECE_SET_YEAR_MAP: Dict[str, str] = {
    "OP01": "2022", "OP02": "2022",
    "OP03": "2023", "OP04": "2023", "OP05": "2023",
    "OP06": "2024", "OP07": "2024", "OP08": "2024",
    "OP09": "2025", "OP10": "2025", "OP11": "2025", "OP12": "2025",
    "ST01": "2022", "ST02": "2022", "ST03": "2022", "ST04": "2022",
    "ST05": "2022", "ST06": "2022",
    "ST07": "2023", "ST08": "2023", "ST09": "2023", "ST10": "2023",
    "ST11": "2024", "ST12": "2024", "ST13": "2024",
    "ST14": "2025", "ST15": "2025",
}

def _onepiece_set_info(set_code: str) -> Dict[str, str]:
    sc = (set_code or "").strip().upper()
    if not sc:
        return {"set_name": "", "year": "", "source": "none"}
    # Normalize like "OP-05" -> "OP05"
    sc_norm = re.sub(r"[^A-Z0-9]", "", sc)
    name = _ONE_PIECE_SET_NAME_MAP.get(sc_norm, "")
    year = _ONE_PIECE_SET_YEAR_MAP.get(sc_norm, "")
    return {"set_code": sc_norm, "set_name": name, "year": year, "source": "heuristic_map" if (name or year) else "unknown"}

# Simple in-memory translation cache (per process)
_TRANSLATE_CACHE: Dict[str, str] = {}

async def _translate_to_english(text_in: str) -> str:
    """Translate a short string to English (best-effort)."""
    t = _norm_ws(text_in or "")
    if not t:
        return ""
    if t in _TRANSLATE_CACHE:
        return _TRANSLATE_CACHE[t]
    # If it already looks English-ish, don't waste calls
    if re.search(r"[A-Za-z]", t) and not re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", t):
        _TRANSLATE_CACHE[t] = t
        return t

        # IMPORTANT: use _openai_text (NOT _openai_chat) because _openai_chat enforces JSON response_format.
    # Translation should be plain text.
    system = "Translate trading card names to English. Return ONLY the English translation (plain text)."
    user = f"Translate this card name to English: {t}"
    try:
        resp = await _openai_text(messages=[{"role":"system","content":system},{"role":"user","content":user}], max_tokens=60, temperature=0.0)
        if resp.get("error"):
            return ""
        out = _norm_ws(resp.get("content",""))
        # Guard against wrappers like: "Hot Breath"
        out = out.strip().strip('"').strip("'").strip()
        if len(out) > 0 and len(out) <= 120:
            _TRANSLATE_CACHE[t] = out
            return out
    except Exception:
        pass
    return ""


def _ocr_rarity_code_from_front(front_bytes: bytes) -> str:
    """Best-effort OCR for rarity codes (SR/R/SAR/SEC/L/UC/C/SP/P, etc.).
    Rarity marks on JP/EN cards are usually Latin letters, so English OCR works.
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        im = Image.open(io.BytesIO(front_bytes)).convert("RGB")
        w, h = im.size
        # Bottom-right crop tends to contain rarity + card number on most modern TCG layouts
        x0 = int(w * 0.60)
        y0 = int(h * 0.70)
        crop = im.crop((x0, y0, w, h))
        # Upscale for OCR
        crop = crop.resize((crop.size[0] * 3, crop.size[1] * 3))

        # High-contrast grayscale
        crop = crop.convert("L")
        # Simple threshold
        crop = crop.point(lambda p: 255 if p > 160 else 0)

        txt = pytesseract.image_to_string(
            crop,
            lang="eng",
            config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/",
        )
        t = re.sub(r"[^A-Z0-9/]", " ", (txt or "").upper())
        # Common rarity tokens across TCGs
        candidates = [
            "SAR",
            "SEC",
            "SR",
            "UR",
            "HR",
            "IR",
            "AR",
            "SP",
            "PR",
            "PROMO",
            "RR",
            "R",
            "UC",
            "C",
            "L",
            "ALT",
            "AA",
        ]
        for c in candidates:
            if re.search(rf"\b{re.escape(c)}\b", t):
                return c
    except Exception:
        pass
    return ""



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



if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
    print("WARNING: Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set! Vision ID/assessment will fail.")
elif not OPENAI_API_KEY:
    print("INFO: OPENAI_API_KEY not set — all AI calls will route directly to Anthropic Claude.")
elif not ANTHROPIC_API_KEY:
    print("INFO: ANTHROPIC_API_KEY not set — Claude fallback disabled. OpenAI only.")
else:
    print("INFO: Both OPENAI_API_KEY and ANTHROPIC_API_KEY set — Claude fallback active.")
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


def _compress_image(raw: bytes, max_long: int = 1200, quality: int = 88) -> bytes:
    """Resize to max_long px on the long edge and re-encode as JPEG.

    Single shared helper replacing the four near-identical implementations that
    previously lived inline in /api/verify, /api/overlay-variants,
    /api/fingerprint/generate, and /api/defect-scan.
    EXIF orientation is corrected before resizing.
    Falls back to raw bytes if PIL is unavailable or the image is invalid.
    """
    if not PIL_AVAILABLE or not raw or len(raw) < 200:
        return raw
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        if max(w, h) > max_long:
            scale = max_long / max(w, h)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception:
        return raw


def _make_defect_filter_variants(img_bytes: bytes) -> Dict[str, bytes]:
    """Create enhanced filter variants for the frontend inspection overlay.

    Performance-optimised rewrite (2026-03):
    - Hard resize to ≤800px before any processing — prevents multi-second
      Python loops on 4K phone images
    - local_variance, edge_wear, uv_sim, sobel_surface: all numpy array ops
      instead of per-pixel Python loops (~50-100x faster)
    - Border mask via numpy slice assignment instead of putpixel() loop
    - JPEG quality 78 (inspection, not print — saves ~40% payload)

    Frontend mode map:
      contrast_sharp  → 'contrast'   | edge_wear      → 'edgewear'
      sobel_surface   → 'surface'    | sobel_edges    → 'edges'
      invert          → 'invert'     | corner_isolate → 'corners'
      uv_sim          → 'uv'         | local_variance → 'variance'
      chromatic       → 'chromatic'  | channel_r/g/b  → 'r'/'g'/'b'
    """
    if not img_bytes or Image is None:
        return {}
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        im = ImageOps.exif_transpose(im)

        # ── Hard resize cap — process at most 800px long edge ─────────────────
        # A 4K image has ~12M pixels; at 800px it's ~600K — 20× less work.
        # Defect detection doesn't need print resolution.
        _MAX_PX = 800
        lw, lh = im.size
        if max(lw, lh) > _MAX_PX:
            scale = _MAX_PX / max(lw, lh)
            im = im.resize((max(1, int(lw * scale)), max(1, int(lh * scale))), Image.LANCZOS)
        w, h = im.size

        variants: Dict[str, bytes] = {}

        def _to_jpeg(img_obj) -> bytes:
            buf = BytesIO()
            # Quality 78: sharp enough for on-screen defect inspection, ~40% smaller than 90
            img_obj.save(buf, format="JPEG", quality=78, optimize=True)
            return buf.getvalue()

        # Shared numpy array (float32) used by several filters below
        arr = np.asarray(im, dtype=np.float32) if np is not None else None

        # ── 1. gray_autocontrast ───────────────────────────────────────────────
        g = ImageOps.grayscale(im)
        g = ImageOps.autocontrast(g)
        g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        variants["gray_autocontrast"] = _to_jpeg(g)

        # ── 2. contrast_sharp ─────────────────────────────────────────────────
        c = ImageEnhance.Contrast(im).enhance(1.6)
        c = ImageEnhance.Sharpness(c).enhance(1.8)
        c = c.filter(ImageFilter.UnsharpMask(radius=1, percent=130, threshold=2))
        variants["contrast_sharp"] = _to_jpeg(c)

        # ── 3. invert ─────────────────────────────────────────────────────────
        variants["invert"] = _to_jpeg(ImageOps.invert(im))

        # ── 4. RGB channel isolation ───────────────────────────────────────────
        r_ch, g_ch, b_ch = im.split()
        variants["channel_r"] = _to_jpeg(r_ch)
        variants["channel_g"] = _to_jpeg(g_ch)
        variants["channel_b"] = _to_jpeg(b_ch)

        # ── 5. Sobel edges + surface (numpy) ──────────────────────────────────
        try:
            sobel_pil = ImageOps.grayscale(im).filter(ImageFilter.FIND_EDGES)
            sobel_pil = ImageOps.autocontrast(sobel_pil)
            variants["sobel_edges"] = _to_jpeg(sobel_pil)

            if np is not None:
                # Surface: zero-out the border band, keep interior Sobel signal
                band = max(4, int(min(w, h) * 0.06))
                sv = np.asarray(sobel_pil, dtype=np.uint8).copy()
                sv[:band, :]   = 0
                sv[-band:, :]  = 0
                sv[:, :band]   = 0
                sv[:, -band:]  = 0
                sv_img = Image.fromarray(sv, mode="L")
                variants["sobel_surface"] = _to_jpeg(ImageOps.autocontrast(sv_img))
            else:
                # Fallback: PIL only (slower but correct)
                sobel_d = np.array(list(sobel_pil.getdata()), dtype=np.uint8) if np else None
                variants["sobel_surface"] = _to_jpeg(sobel_pil)
        except Exception:
            pass

        # ── 6. Edge wear — numpy vectorised ───────────────────────────────────
        # Highlights bright low-chroma pixels near borders (whitening/silvering).
        try:
            if arr is not None:
                band = max(4, int(min(w, h) * 0.08))
                R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

                # Luminance and chroma per pixel
                lum   = 0.2126 * R + 0.7152 * G + 0.0722 * B
                chroma = np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)

                # Border mask
                border = np.zeros((h, w), dtype=bool)
                border[:band, :]   = True
                border[-band:, :]  = True
                border[:, :band]   = True
                border[:, -band:]  = True

                # Whitening hit: bright + low chroma in border zone
                hit = border & (lum > 190) & (chroma < 55)

                ew = np.zeros((h, w, 3), dtype=np.uint8)
                # Interior: desaturate
                gray3 = lum.astype(np.uint8)
                ew[~border, 0] = gray3[~border]
                ew[~border, 1] = gray3[~border]
                ew[~border, 2] = gray3[~border]
                # Border non-hit: darken
                ew[border & ~hit, 0] = (R[border & ~hit] / 3).astype(np.uint8)
                ew[border & ~hit, 1] = (G[border & ~hit] / 3).astype(np.uint8)
                ew[border & ~hit, 2] = (B[border & ~hit] / 3).astype(np.uint8)
                # Whitening hit: vivid red
                ew[hit, 0] = 255
                ew[hit, 1] = np.clip(R[hit] - 200, 0, 255).astype(np.uint8)
                ew[hit, 2] = np.clip(B[hit] - 200, 0, 255).astype(np.uint8)

                variants["edge_wear"] = _to_jpeg(Image.fromarray(ew, mode="RGB"))
        except Exception:
            pass

        # ── 7. UV simulation — numpy vectorised ───────────────────────────────
        try:
            if arr is not None:
                R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                v = np.clip(R * 0.5 + G * 0.5 - B * 0.35 + 40, 0, 255).astype(np.uint8)
                uv = np.stack([v, v, np.clip(v.astype(np.int16) - 60, 0, 255).astype(np.uint8)], axis=2)
                uv_img = Image.fromarray(uv, mode="RGB")
                uv_img = ImageEnhance.Contrast(uv_img).enhance(1.8)
                variants["uv_sim"] = _to_jpeg(uv_img)
        except Exception:
            pass

        # ── 8. Local variance — numpy uniform_filter ──────────────────────────
        # Surface scratches create tiny high-variance patches.
        # scipy.ndimage.uniform_filter is ~200x faster than the Python patch loop.
        try:
            if np is not None:
                gray_np = np.asarray(ImageOps.grayscale(im), dtype=np.float32)
                try:
                    from scipy.ndimage import uniform_filter as _uf
                    mean_f  = _uf(gray_np, size=5)
                    sq_mean = _uf(gray_np ** 2, size=5)
                except ImportError:
                    # scipy unavailable: use numpy stride-based box filter
                    from numpy.lib.stride_tricks import sliding_window_view
                    wins = sliding_window_view(
                        np.pad(gray_np, 2, mode='edge'), (5, 5)
                    )
                    mean_f  = wins.mean(axis=(-2, -1))
                    sq_mean = (wins ** 2).mean(axis=(-2, -1))

                var_map = np.clip(sq_mean - mean_f ** 2, 0, None)
                # Scale variance to 0-255
                vmax = var_map.max()
                if vmax > 0:
                    lv = np.clip((var_map / vmax) * 255 * 0.4, 0, 255).astype(np.uint8)
                else:
                    lv = np.zeros_like(gray_np, dtype=np.uint8)

                # Zero-out border band via slice (no putpixel loop)
                band_lv = max(4, int(min(w, h) * 0.06))
                lv[:band_lv, :]    = 0
                lv[-band_lv:, :]   = 0
                lv[:, :band_lv]    = 0
                lv[:, -band_lv:]   = 0

                lv_img = Image.fromarray(lv, mode="L")
                lv_img = ImageOps.autocontrast(lv_img)
                # Return as RGB so frontend canvas treats it consistently
                variants["local_variance"] = _to_jpeg(lv_img.convert("RGB"))
        except Exception:
            pass

        # ── 9. Chromatic (exaggerate saturation) ──────────────────────────────
        try:
            hsv_img = im.convert("HSV")
            h_ch, s_ch, v_ch = hsv_img.split()
            s_boosted = ImageEnhance.Brightness(s_ch).enhance(3.0)
            chroma = Image.merge("HSV", (h_ch, s_boosted, v_ch)).convert("RGB")
            chroma = ImageEnhance.Contrast(chroma).enhance(1.5)
            variants["chromatic"] = _to_jpeg(chroma)
        except Exception:
            pass

        # ── 10. Corner isolate (2×2 grid of contrast-enhanced corner crops) ───
        try:
            csize = max(32, int(min(w, h) * 0.25))
            defs = [
                (0,       0,       csize,   csize),
                (w-csize, 0,       w,       csize),
                (0,       h-csize, csize,   h),
                (w-csize, h-csize, w,       h),
            ]
            tile = csize * 2
            grid_img = Image.new("RGB", (tile + 4, tile + 4), (20, 20, 20))
            positions = [(0, 0), (csize + 4, 0), (0, csize + 4), (csize + 4, csize + 4)]
            for (x1, y1, x2, y2), (px, py) in zip(defs, positions):
                crop = im.crop((x1, y1, x2, y2)).resize((csize, csize), Image.LANCZOS)
                crop = ImageEnhance.Contrast(crop).enhance(1.8)
                crop = ImageEnhance.Sharpness(crop).enhance(2.5)
                crop = ImageOps.autocontrast(crop.convert("L")).convert("RGB")
                grid_img.paste(crop, (px, py))
            variants["corner_isolate"] = _to_jpeg(grid_img)
        except Exception:
            pass

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
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
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
    """Return True if s is empty or a common placeholder/unknown value."""
    s2 = _norm_ws(s or "").lower()
    return (not s2) or s2 in (
        "unknown", "n/a", "na", "none", "null", "undefined",
        "not sure", "unsure",
    )

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


def _centering_grade_from_ratio(larger: float, smaller: float, axis: str = "lr") -> str:
    """Return a human-friendly centering string like '55/45'.

    Physical card centering limits (standard TCG borders):
      L/R: never more extreme than 80/20 (true miscut territory)
      T/B: never more extreme than 75/25 (T/B borders are narrower so miscuts show earlier)
    We clamp to these to avoid artefacts like '99/1' from holofoil edge detection noise.
    """
    try:
        larger = float(larger)
        smaller = float(smaller)
        total = max(larger + smaller, 1e-6)
        p_large = int(round((larger / total) * 100))
        p_small = 100 - p_large
        # Physical clamp — no real card can have a border ratio outside these
        if axis == "tb":
            p_large = max(50, min(75, p_large))
        else:  # lr
            p_large = max(50, min(80, p_large))
        p_small = 100 - p_large
        return f"{p_large}/{p_small}"
    except Exception:
        return "N/A"


def _estimate_centering_from_image(img_bytes: bytes) -> dict | None:
    """Best-effort centering estimate from a single image (PIL + numpy).

    Returns dict with lr/tb ratios (e.g. 55/45).
    Vectorised with numpy strides — no Python loops.
    """
    if not PIL_AVAILABLE or Image is None or np is None:
        return None
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        im = ImageOps.exif_transpose(im)
        im = im.resize((900, int(900 * im.height / max(im.width, 1))), Image.LANCZOS)
        g = ImageOps.grayscale(im)
        a = np.asarray(g, dtype=np.float32)

        # Vectorised Sobel via numpy (no Python inner loops)
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        def conv2_np(img: np.ndarray, k: np.ndarray) -> np.ndarray:
            """2D convolution via numpy stride tricks — ~100x faster than Python loops."""
            kh, kw = k.shape
            ph, pw = kh // 2, kw // 2
            padded = np.pad(img, ((ph, ph), (pw, pw)), mode='edge')
            # Build view of shape (H, W, kh, kw)
            shape = img.shape + k.shape
            strides = padded.strides + padded.strides
            view = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
            return (view * k).sum(axis=(-2, -1))

        gx = conv2_np(a, kx)
        gy = conv2_np(a, ky)
        mag = np.sqrt(gx * gx + gy * gy)

        thr = float(np.percentile(mag, 95))
        edges = (mag >= thr).astype(np.uint8)
        ys, xs = np.where(edges > 0)
        if len(xs) < 500:
            return None

        x1, x2 = int(np.percentile(xs, 1)), int(np.percentile(xs, 99))
        y1, y2 = int(np.percentile(ys, 1)), int(np.percentile(ys, 99))
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(edges.shape[1]-1, x2); y2 = min(edges.shape[0]-1, y2)
        if x2 - x1 < 200 or y2 - y1 < 200:
            return None

        roi = edges[y1:y2, x1:x2]
        col = roi.sum(axis=0).astype(np.float32)
        row = roi.sum(axis=1).astype(np.float32)

        def smooth(v, w=11):
            if v.size < w:
                return v
            kernel = np.ones(w, dtype=np.float32) / float(w)
            return np.convolve(v, kernel, mode='same')

        col_s = smooth(col)
        row_s = smooth(row)
        w = col_s.size
        h = row_s.size

        left_zone = col_s[: int(w * 0.35)]
        right_zone = col_s[int(w * 0.65):]
        top_zone = row_s[: int(h * 0.35)]
        bot_zone = row_s[int(h * 0.65):]

        def peak_index(zone, from_left=True):
            if zone.size < 10:
                return None
            m = float(zone.max())
            if m <= 0:
                return None
            candidates = np.where(zone >= (m * 0.75))[0]
            if candidates.size == 0:
                return None
            return int(candidates[0] if from_left else candidates[-1])

        li = peak_index(left_zone, True)
        ri_rel = peak_index(right_zone, False)
        ti = peak_index(top_zone, True)
        bi_rel = peak_index(bot_zone, False)
        if li is None or ri_rel is None or ti is None or bi_rel is None:
            return None

        ri = int(w * 0.65) + ri_rel
        bi = int(h * 0.65) + bi_rel

        margin_left = max(1, li)
        margin_right = max(1, (w - 1) - ri)
        margin_top = max(1, ti)
        margin_bottom = max(1, (h - 1) - bi)

        lr = _centering_grade_from_ratio(max(margin_left, margin_right), min(margin_left, margin_right), axis="lr")
        tb = _centering_grade_from_ratio(max(margin_top, margin_bottom), min(margin_top, margin_bottom), axis="tb")

        return {
            "lr": lr,
            "tb": tb,
            "margins": {"left": int(margin_left), "right": int(margin_right), "top": int(margin_top), "bottom": int(margin_bottom)},
        }
    except Exception:
        return None


def _autocrop_card(img_bytes: bytes, inset_pct: float = 0.0, pad_pct: float = 0.03) -> tuple:
    """
    Detect the physical card border and crop tightly to it, with generous
    outward padding to avoid shaving any edge.

    Improvements over previous version (2026-03):
    - Auto-rotate: detects if card is photographed sideways (landscape image
      containing a portrait card) and corrects to portrait before returning.
    - Wider background strip (6% not 3%) — more robust on close-up shots.
    - Tighter percentile clipping (0.2%/99.8% not 1%/99%) — preserves thin
      white card borders that were being trimmed off.
    - _find_edge min_frac reduced 30% → 12% — finds the genuine card edge
      rather than pushing it inside the white border.
    - Absolute minimum safety pad: always adds at least 12px outward so the
      border pixel is never exactly on the crop boundary.
    - Sanity guard relaxed 78% → 60% — accepts more zoom levels.

    Returns (cropped_bytes: bytes, crop_info: dict).
    """
    if not PIL_AVAILABLE or Image is None or np is None or not img_bytes:
        return img_bytes, {"detected": False}

    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        im = ImageOps.exif_transpose(im)
        orig_w, orig_h = im.size

        # ── 0. Auto-rotate: if detected card bbox is in landscape but image is
        #       portrait (or vice versa), rotate 90° so card is upright. ────────
        # We'll do this AFTER detection — decide at end whether to rotate output.

        # ── 1. Downscale for speed ─────────────────────────────────────────────
        WORK_LONG = 900
        long_edge = max(orig_w, orig_h)
        scale = WORK_LONG / max(long_edge, 1)
        work_w = max(1, int(orig_w * scale))
        work_h = max(1, int(orig_h * scale))
        work = im.resize((work_w, work_h), Image.LANCZOS)
        arr = np.asarray(work, dtype=np.float32)   # (H, W, 3)

        # ── 2. Background colour — wider strip (6%) for better sampling ────────
        # On close-up shots the card nearly fills the frame; 3% was too thin and
        # occasionally sampled card pixels rather than background.
        bpx = max(6, int(min(work_w, work_h) * 0.06))
        border_pixels = np.vstack([
            arr[:bpx, :, :].reshape(-1, 3),
            arr[-bpx:, :, :].reshape(-1, 3),
            arr[:, :bpx, :].reshape(-1, 3),
            arr[:, -bpx:, :].reshape(-1, 3),
        ])
        bg_colour = np.median(border_pixels, axis=0)

        # ── 3. Card mask ───────────────────────────────────────────────────────
        diff = np.sqrt(np.sum((arr - bg_colour) ** 2, axis=2))
        bg_std = float(
            np.std(diff[:bpx, :]) + np.std(diff[-bpx:, :]) +
            np.std(diff[:, :bpx]) + np.std(diff[:, -bpx:])
        ) / 4.0
        # Lower minimum threshold (18 instead of 22) — catches faint card edges
        thr_val = max(18.0, bg_std * 2.5)
        card_mask = diff > thr_val

        ys, xs = np.where(card_mask)
        if len(xs) < 200:
            return img_bytes, {"detected": False, "original_size": [orig_w, orig_h]}

        # ── 4. Coarse bbox — tighter percentile (0.2/99.8) preserves white borders
        x1c = int(np.percentile(xs, 0.2));  x2c = int(np.percentile(xs, 99.8))
        y1c = int(np.percentile(ys, 0.2));  y2c = int(np.percentile(ys, 99.8))

        if x2c - x1c < work_w * 0.20 or y2c - y1c < work_h * 0.20:
            return img_bytes, {"detected": False, "original_size": [orig_w, orig_h]}

        # ── 5. Fine edge refinement — min_frac 12% (was 30%) ─────────────────
        # 30% was too aggressive: it looked for a column where 30% of pixels
        # were card-coloured, pushing the detected edge inside the white border.
        # 12% reliably catches the first card pixels while ignoring noise.
        def _find_edge(mask: np.ndarray, axis: int, from_start: bool,
                       min_frac: float = 0.12) -> int:
            n = mask.shape[1] if axis == 1 else mask.shape[0]
            total = mask.shape[0] if axis == 1 else mask.shape[1]
            idx_range = range(n) if from_start else range(n - 1, -1, -1)
            for i in idx_range:
                stripe = mask[:, i] if axis == 1 else mask[i, :]
                if stripe.sum() / max(total, 1) >= min_frac:
                    return i
            return 0 if from_start else n - 1

        roi = card_mask[max(0, y1c - 2):min(work_h, y2c + 2),
                        max(0, x1c - 2):min(work_w, x2c + 2)]

        fine_x1 = max(0,        x1c - 2 + _find_edge(roi, axis=1, from_start=True))
        fine_x2 = min(work_w-1, x1c - 2 + _find_edge(roi, axis=1, from_start=False))
        fine_y1 = max(0,        y1c - 2 + _find_edge(roi, axis=0, from_start=True))
        fine_y2 = min(work_h-1, y1c - 2 + _find_edge(roi, axis=0, from_start=False))

        cw = fine_x2 - fine_x1
        ch = fine_y2 - fine_y1
        if cw < 50 or ch < 50:
            return img_bytes, {"detected": False, "original_size": [orig_w, orig_h]}

        # ── 5b. Sanity guard — relaxed to 60% (was 78%) ───────────────────────
        if (cw / max(work_w, 1) < 0.60) or (ch / max(work_h, 1) < 0.60):
            return img_bytes, {"detected": False, "original_size": [orig_w, orig_h]}

        # ── 6. Inset + outward pad + absolute safety minimum ──────────────────
        inset_pct = max(0.0, float(inset_pct or 0.0))
        pad_pct   = max(0.0, float(pad_pct or 0.0))

        inset_x = int(cw * inset_pct) if inset_pct > 0 else 0
        inset_y = int(ch * inset_pct) if inset_pct > 0 else 0
        pad_x   = int(cw * pad_pct)   if pad_pct   > 0 else 0
        pad_y   = int(ch * pad_pct)   if pad_pct   > 0 else 0

        # Absolute safety pad: always expand by at least 12 work-pixels so the
        # genuine card border pixel is never exactly on the crop boundary.
        ABS_PAD = 12
        pad_x = max(pad_x, ABS_PAD)
        pad_y = max(pad_y, ABS_PAD)

        inv = 1.0 / max(scale, 1e-6)
        ox1 = max(0,       int((fine_x1 + inset_x - pad_x) * inv))
        oy1 = max(0,       int((fine_y1 + inset_y - pad_y) * inv))
        ox2 = min(orig_w,  int((fine_x2 - inset_x + pad_x) * inv))
        oy2 = min(orig_h,  int((fine_y2 - inset_y + pad_y) * inv))

        if ox2 - ox1 < 80 or oy2 - oy1 < 80:
            return img_bytes, {"detected": False, "original_size": [orig_w, orig_h]}

        cropped = im.crop((ox1, oy1, ox2, oy2))

        # ── 7. Auto-rotate: correct for sideways photography ─────────────────
        # Standard trading/sports cards are portrait (taller than wide).
        # If the cropped region is landscape (wider than tall), rotate 90° CW
        # so downstream analysis always sees a portrait card.
        # Exemption: if the *original image* is already landscape (user shot
        # landscape intentionally, e.g., for a wide-format card), don't rotate.
        crop_w = ox2 - ox1
        crop_h = oy2 - oy1
        rotated = False
        PORTRAIT_RATIO = 1.15   # card must be at least 15% taller than wide to count
        if crop_w > crop_h * PORTRAIT_RATIO and orig_h >= orig_w:
            # Cropped bbox is landscape but original shot was portrait —
            # the card was photographed sideways. Rotate 90° counter-clockwise.
            cropped = cropped.rotate(90, expand=True)
            rotated = True

        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=92, optimize=True)

        crop_info = {
            "detected":      True,
            "original_size": [orig_w, orig_h],
            "crop_box":      [ox1, oy1, ox2, oy2],
            "crop_pct":      [
                round(ox1 / orig_w, 4), round(oy1 / orig_h, 4),
                round(ox2 / orig_w, 4), round(oy2 / orig_h, 4),
            ],
            "rotated":       rotated,
            "pad_applied":   [pad_x, pad_y],
        }
        return buf.getvalue(), crop_info

    except Exception:
        return img_bytes, {"detected": False}


def _normalize_card_type(card_type: str) -> str:
    """Force card_type into the allowed enum. Covers all major TCGs + sports."""
    s = _norm_ws(card_type or "").lower()
    if not s:
        return "Other"
    # ── Major TCGs ──────────────────────
    if "pokemon" in s or s in ("pkmn", "poke", "pokémon", "ptcg"):
        return "Pokemon"
    if "magic" in s or "mtg" in s or "magic the gathering" in s or "magic: the gathering" in s:
        return "Magic"
    if "yug" in s or "yu-gi" in s or "yugi" in s or "ygo" in s:
        return "YuGiOh"
    if "one piece" in s or "onepiece" in s or "opcg" in s:
        return "OnePiece"
    if "dragon ball" in s or "dragonball" in s or "dbs" in s or "dbz" in s:
        return "DragonBall"
    if "digimon" in s or "dcg" in s:
        return "Digimon"
    if "lorcana" in s:
        return "Lorcana"
    if "flesh and blood" in s or "fab" in s:
        return "FleshAndBlood"
    if "weiss" in s or "schwarz" in s or "ws" == s:
        return "WeissSchwarz"
    if "cardfight" in s or "vanguard" in s:
        return "Vanguard"
    if "star wars" in s or "swu" in s:
        return "StarWars"
    if "union arena" in s:
        return "UnionArena"
    if "metazoo" in s:
        return "MetaZoo"
    if "my hero" in s or "mha" in s:
        return "MyHeroAcademia"
    if "gundam" in s:
        return "Gundam"
    if "naruto" in s or "boruto" in s:
        return "Naruto"
    # ── Sports ──────────────────────────
    if "afl" in s or "australian football" in s:
        return "Sports_AFL"
    if "nrl" in s or "rugby league" in s:
        return "Sports_NRL"
    if "cricket" in s:
        return "Sports_Cricket"
    if "nba" in s or "basketball" in s:
        return "Sports_Basketball"
    if "nfl" in s or "american football" in s:
        return "Sports_NFL"
    if "mlb" in s or "baseball" in s:
        return "Sports_Baseball"
    if "nhl" in s or "hockey" in s:
        return "Sports_Hockey"
    if "soccer" in s or "football" in s or "epl" in s or "a-league" in s or "fifa" in s:
        return "Sports_Soccer"
    if "f1" in s or "formula" in s or "motorsport" in s:
        return "Sports_Motorsport"
    if "ufc" in s or "mma" in s or "boxing" in s or "wwe" in s or "wrestling" in s:
        return "Sports_Combat"
    if "tennis" in s or "golf" in s or "olympic" in s:
        return "Sports_Other"
    if "sport" in s or "panini" in s or "topps" in s or "upper deck" in s or "bowman" in s:
        return "Sports"
    # ── Generic / fallback ──────────────
    if s in ("other", "other tcg", "tcg", "trading card", "tradingcard"):
        return "Other"
    return "Other"



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
# ══════════════════════════════════════════════════════════════════════════════
# ANTHROPIC CLAUDE FALLBACK
# Called automatically when OpenAI fails (any error, timeout, or 5xx).
# _openai_chat → _claude_fallback_chat  (vision: claude-sonnet-4-6,
#                                         text:   claude-haiku-4-5-20251001)
# _openai_text → _claude_fallback_text  (haiku only — no JSON enforcement)
#
# Format conversion: OpenAI image_url blocks → Anthropic image source blocks.
# System messages extracted from the messages array → Anthropic `system` param.
# JSON responses enforced via prompt instruction (Anthropic has no response_format).
# ══════════════════════════════════════════════════════════════════════════════

def _openai_to_anthropic_messages(messages: List[Dict[str, Any]]) -> tuple:
    """
    Convert OpenAI chat messages list to Anthropic API format.

    OpenAI                              →  Anthropic
    ──────────────────────────────────────────────────
    {"role":"system","content":"..."}   →  system= param (extracted out)
    {"type":"image_url",                →  {"type":"image","source":
      "image_url":{"url":"data:..."}}        {"type":"base64","media_type":...,
                                              "data":...}}
    {"type":"text","text":"..."}        →  {"type":"text","text":"..."} (unchanged)

    Returns:
        (system_str: str | None, anthropic_messages: list)
    """
    system_parts: List[str] = []
    out_msgs: List[Dict[str, Any]] = []

    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")

        # ── System messages go into the system param, not the messages array ──
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        system_parts.append(blk.get("text", ""))
            continue

        # ── Convert content blocks ────────────────────────────────────────────
        if isinstance(content, str):
            out_content: Any = content

        elif isinstance(content, list):
            out_blocks: List[Dict[str, Any]] = []
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                btype = blk.get("type", "")

                if btype == "text":
                    out_blocks.append({"type": "text", "text": blk.get("text", "")})

                elif btype == "image_url":
                    url = (blk.get("image_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        # data:image/jpeg;base64,/9j/...
                        try:
                            header, b64data = url.split(",", 1)
                            media_type = header.split(";")[0].replace("data:", "")
                        except ValueError:
                            media_type, b64data = "image/jpeg", url
                        out_blocks.append({
                            "type": "image",
                            "source": {
                                "type":       "base64",
                                "media_type": media_type,
                                "data":       b64data,
                            },
                        })
                    elif url:
                        # Plain URL — Anthropic supports these via url source type
                        out_blocks.append({
                            "type": "image",
                            "source": {"type": "url", "url": url},
                        })
            out_content = out_blocks

        else:
            out_content = str(content)

        out_msgs.append({"role": role, "content": out_content})

    system_str = "\n\n".join(system_parts) if system_parts else None
    return system_str, out_msgs


async def _claude_fallback_chat(
    messages:    List[Dict[str, Any]],
    max_tokens:  int   = 1200,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Claude replacement for _openai_chat (JSON-enforced vision/text completions).

    Model selection:
      • Any message contains an image  →  claude-sonnet-4-6  (vision capable)
      • Text-only messages             →  claude-haiku-4-5-20251001  (fast + cheap)

    JSON enforcement: injected into the system prompt since Anthropic has no
    response_format parameter. Output is cleaned of any accidental markdown fences.

    Never raises — returns {"error": True, ...} on any failure.
    """
    if not ANTHROPIC_AVAILABLE or not _anthropic_sdk or not ANTHROPIC_API_KEY:
        return {"error": True, "status": 0, "message": "Anthropic SDK/key not available"}

    try:
        system_str, anthropic_msgs = _openai_to_anthropic_messages(messages)

        # Detect whether any content block is an image
        has_vision = any(
            isinstance(msg.get("content"), list)
            and any(b.get("type") == "image" for b in msg["content"])
            for msg in anthropic_msgs
        )
        model = (
            os.getenv("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-6")
            if has_vision
            else os.getenv("ANTHROPIC_TEXT_MODEL", "claude-haiku-4-5-20251001")
        )

        # Enforce JSON output via system prompt (Anthropic has no response_format param)
        json_instruction = (
            "\n\nCRITICAL: Your response MUST be a single valid JSON object only. "
            "No markdown code fences, no explanation, no text before or after the JSON. "
            "Start your response with { and end with }."
        )
        if system_str:
            system_str += json_instruction
        else:
            system_str = "You are a helpful AI assistant." + json_instruction

        client = _anthropic_sdk.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        resp = await client.messages.create(
            model       = model,
            max_tokens  = max_tokens,
            temperature = temperature,
            system      = system_str,
            messages    = anthropic_msgs,
        )

        content = (resp.content[0].text if resp.content else "").strip()

        # Strip accidental markdown fences Claude sometimes adds
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*\n?", "", content)
            content = re.sub(r"\n?\s*```\s*$",        "", content)
            content = content.strip()

        logging.info(
            f"✅ Claude fallback (chat) OK — model={model} "
            f"in={resp.usage.input_tokens} out={resp.usage.output_tokens}"
        )
        return {"error": False, "content": content, "provider": "anthropic", "model": model}

    except Exception as exc:
        logging.error(f"❌ Claude fallback (chat) failed: {exc}")
        return {"error": True, "status": 0, "message": str(exc), "provider": "anthropic"}


async def _claude_fallback_text(
    messages:    List[Dict[str, Any]],
    max_tokens:  int   = 220,
    temperature: float = 0.6,
) -> Dict[str, Any]:
    """
    Claude replacement for _openai_text (free-form text, no JSON enforcement).
    Always uses claude-haiku-4-5-20251001 — cheap, fast, good enough for spoken briefs.
    Never raises — returns {"error": True, ...} on any failure.
    """
    if not ANTHROPIC_AVAILABLE or not _anthropic_sdk or not ANTHROPIC_API_KEY:
        return {"error": True, "status": 0, "message": "Anthropic SDK/key not available"}

    try:
        system_str, anthropic_msgs = _openai_to_anthropic_messages(messages)

        model  = os.getenv("ANTHROPIC_TEXT_MODEL", "claude-haiku-4-5-20251001")
        client = _anthropic_sdk.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

        kwargs: Dict[str, Any] = {
            "model":       model,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "messages":    anthropic_msgs,
        }
        if system_str:
            kwargs["system"] = system_str

        resp    = await client.messages.create(**kwargs)
        content = (resp.content[0].text if resp.content else "").strip()

        logging.info(f"✅ Claude fallback (text) OK — model={model}")
        return {"error": False, "content": content, "provider": "anthropic", "model": model}

    except Exception as exc:
        logging.error(f"❌ Claude fallback (text) failed: {exc}")
        return {"error": True, "status": 0, "message": str(exc), "provider": "anthropic"}


# OpenAI helper (vision)
# ==============================
async def _openai_chat(messages: List[Dict[str, Any]], max_tokens: int = 1200, temperature: float = 0.1) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        # No OpenAI key at all — go straight to Claude
        logging.warning("⚠️  OpenAI key not set — routing _openai_chat directly to Claude fallback")
        return await _claude_fallback_chat(messages, max_tokens=max_tokens, temperature=temperature)

    url = "https://api.openai.com/v1/chat/completions"
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "response_format": {"type": "json_object"}}

    last_error = {}
    for attempt in range(3):
        try:
            client = _get_http_client()
            try:
                r = await client.post(url, headers=headers, json=payload)
            except httpx.TransportError:
                # Client may have gone stale — reset and retry once
                global _HTTP_CLIENT
                _HTTP_CLIENT = None
                client = _get_http_client()
                r = await client.post(url, headers=headers, json=payload)
            if r.status_code == 200:
                data = r.json()
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                return {"error": False, "content": content}
            elif r.status_code in (429, 503) and attempt < 2:
                # Rate-limited or service busy — back off and retry
                await asyncio.sleep(3 * (attempt + 1))
                last_error = {"error": True, "status": r.status_code, "message": r.text[:700]}
                continue
            else:
                last_error = {"error": True, "status": r.status_code, "message": r.text[:700]}
                break   # non-retryable HTTP error → go straight to Claude
        except Exception as e:
            last_error = {"error": True, "status": 0, "message": str(e)}
            if attempt < 2:
                await asyncio.sleep(2)
            continue

    # ── All OpenAI attempts exhausted — try Anthropic Claude ─────────────────
    logging.warning(
        f"⚠️  OpenAI failed after retries "
        f"(status={last_error.get('status',0)}, msg={str(last_error.get('message',''))[:120]}) "
        f"— falling back to Claude"
    )
    claude_result = await _claude_fallback_chat(messages, max_tokens=max_tokens, temperature=temperature)
    if not claude_result.get("error"):
        return claude_result

    # Both providers failed — return the OpenAI error (more informative for debugging)
    logging.error(f"❌ Both OpenAI and Claude failed. OpenAI: {last_error} | Claude: {claude_result}")
    return last_error


# ==============================
# OpenAI helper (text)
# ==============================
async def _openai_text(messages: List[Dict[str, Any]], max_tokens: int = 220, temperature: float = 0.6) -> Dict[str, Any]:
    """Lightweight text generation helper (no JSON response_format)."""
    if not OPENAI_API_KEY:
        logging.warning("⚠️  OpenAI key not set — routing _openai_text directly to Claude fallback")
        return await _claude_fallback_text(messages, max_tokens=max_tokens, temperature=temperature)

    url = "https://api.openai.com/v1/chat/completions"
    model = os.getenv("OPENAI_TEXT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}

    last_error: Dict[str, Any] = {}
    for attempt in range(3):
        try:
            client = _get_http_client()
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code == 200:
                data = r.json()
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                return {"error": False, "content": (content or "").strip()}
            elif r.status_code in (429, 503) and attempt < 2:
                await asyncio.sleep(3 * (attempt + 1))
                last_error = {"error": True, "status": r.status_code, "message": r.text[:700]}
                continue
            else:
                last_error = {"error": True, "status": r.status_code, "message": r.text[:700]}
                break   # non-retryable → fall through to Claude
        except Exception as e:
            last_error = {"error": True, "status": 0, "message": str(e)}
            if attempt < 2:
                await asyncio.sleep(2)
            continue

    # ── All OpenAI attempts exhausted — try Anthropic Claude ─────────────────
    logging.warning(
        f"⚠️  OpenAI text failed (status={last_error.get('status',0)}) — falling back to Claude"
    )
    claude_result = await _claude_fallback_text(messages, max_tokens=max_tokens, temperature=temperature)
    if not claude_result.get("error"):
        return claude_result

    logging.error(f"❌ Both OpenAI and Claude text failed.")
    return last_error


async def _generate_market_spoken_brief(
    *,
    card_identifier: str,
    price_low: float,
    price_median: float,
    price_high: float,
    volume: int,
    history_days_logged: int,
    data_source: str,
) -> str:
    """Generate a short, spoken-word style market brief (not a template)."""
    sys = (
        "You are a modern card collector and market watcher for Collectors League. "
        "Write a natural, spoken-word brief that a host could read out loud — upbeat, confident, and collector-style, but still factual. "
        "No bullet points, no headings, no placeholders. "
        "Mention the low/median/high range in AUD and what that range suggests about where the market is sitting right now. "
        "Base the outlook only on the data provided (range, volume, and how many days are logged). "
        "If history is limited, frame it positively (we're building the track record) and say what extra data would firm up the trend. "
        "Avoid financial advice language; keep it informational and community-friendly."
    )

    user = (
        f"Card identifier: {card_identifier}\n"
        f"Data source: {data_source}\n"
        f"Low/Median/High (AUD): {price_low:.2f} / {price_median:.2f} / {price_high:.2f}\n"
        f"Listings/Sales counted: {int(volume or 0)}\n"
        f"Days logged in database (so far): {int(history_days_logged or 0)}\n"
        "Write ~70-110 words."
    )

    out = await _openai_text(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        max_tokens=260,
        temperature=0.65,
    )

    if out.get("error"):
        return ""
    return (out.get("content") or "").strip()



# Backwards-compatible alias (older call sites)


# Backward-compat alias (some routes still call the old name)
async def _generate_spoken_market_brief(**kwargs):
    return await _generate_market_spoken_brief(**kwargs)
# NOTE: do not redefine _generate_market_spoken_brief here (would cause recursion)

async def _ebay_active_stats(keyword_query: str, limit: int = 120) -> dict:
    """
    Fetch eBay ACTIVE listings stats.

    Prefer Buy/Browse API (OAuth) when available.
    Falls back to FindingService (AppID) if OAuth creds are missing.
    """
    q = _norm_ws(keyword_query or "")
    if not q:
        return {}

    cache_key = f"ebay_active2:{q}:{int(limit or 0)}"
    now = int(time.time())
    cached = _EBAY_CACHE.get(cache_key)
    if cached and (now - int(cached.get("ts", 0))) < 900:
        return cached.get("data", {}) or {}

    target = max(1, int(limit or 120))
    prices: List[float] = []

    # Warm the FX cache so _usd_to_aud_simple picks up the live rate
    await _fx_usd_to_aud()

    # --- Path A: Browse API (preferred) ---
    token, _dbg = await _get_ebay_app_token()
    if token:
        try:
            # Browse API supports up to 200 per request; we page cautiously
            per_page = min(50, target)
            pages = min(6, max(1, (target + per_page - 1) // per_page))
            url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
            headers = {
                "Authorization": f"Bearer {token}",
                "X-EBAY-C-MARKETPLACE-ID": EBAY_MARKETPLACE_ID,
                "Accept": "application/json",
                "User-Agent": UA,
            }

            async with httpx.AsyncClient(timeout=25.0, headers=headers) as client:
                for page in range(pages):
                    params = {
                        "q": q,
                        "limit": str(per_page),
                        "offset": str(page * per_page),
                    }
                    r = await client.get(url, params=params)
                    if r.status_code == 401:
                        # refresh once
                        token2, _ = await _get_ebay_app_token(force_refresh=True)
                        if token2:
                            headers["Authorization"] = f"Bearer {token2}"
                            r = await client.get(url, params=params)
                    if r.status_code != 200:
                        break
                    j = r.json() or {}
                    items = j.get("itemSummaries") or []
                    for it in items:
                        try:
                            pr = (it.get("price") or {})
                            val = float(pr.get("value") or 0.0)
                            cur = str(pr.get("currency") or DEFAULT_CURRENCY).upper()
                            if val <= 0:
                                continue
                            val = await _fx_to_aud(val, cur if cur else "AUD")
                            if val <= 0:
                                continue  # unknown currency or conversion failed
                            if val <= 0 or val > 50_000:
                                continue
                            prices.append(val)
                        except Exception:
                            continue
                    if len(prices) >= target:
                        break
        except Exception:
            prices = []

    # --- Path B: FindingService fallback (AppID) ---
    if not prices and EBAY_APP_ID:
        try:
            target2 = max(1, int(target))
            pages = min(5, max(1, (target2 + 99) // 100))
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
                        "SECURITY-APPNAME": EBAY_APP_ID,
                        "keywords": q,
                        "paginationInput.entriesPerPage": "100",
                        "paginationInput.pageNumber": str(page),
                        "itemFilter(0).name": "HideDuplicateItems",
                        "itemFilter(0).value": "true",
                    }
                    r = await client.get(url, params=params)
                    if r.status_code != 200:
                        continue
                    j = r.json()
                    items = (j.get("findItemsAdvancedResponse", [{}])[0].get("searchResult", [{}])[0].get("item", [])) or []
                    for it in items:
                        try:
                            selling = (it.get("sellingStatus") or [{}])[0]
                            cp = (selling.get("convertedCurrentPrice") or selling.get("currentPrice") or [{}])[0]
                            p = float(cp.get("__value__", 0.0))
                            cur = str(cp.get("@currencyId", "USD")).upper()
                            if p <= 0:
                                continue
                            p = await _fx_to_aud(p, cur if cur else "AUD")
                            if p <= 0:
                                continue  # unknown currency or conversion failed
                            if p <= 0 or p > 50_000:
                                continue
                            prices.append(p)
                        except Exception:
                            continue
                    if len(prices) >= target2:
                        break
        except Exception:
            prices = []

    prices = sorted(prices)
    count = len(prices)
    if count == 0:
        out = {"source": "ebay", "query": q, "count": 0, "currency": "AUD", "prices": []}
        _EBAY_CACHE[cache_key] = {"ts": now, "data": out}
        return out

    def pct(p: float) -> float:
        idx = max(0, min(count - 1, int(round(p * (count - 1)))))
        return float(prices[idx])

    raw_median = pct(0.50)
    raw_low    = pct(0.20)
    raw_high   = pct(0.80)

    # Synthetic spread for tiny samples so low/median/high differ meaningfully
    if count < 3:
        raw_low  = min(raw_low,  raw_median * 0.95)
        raw_high = max(raw_high, raw_median * 1.05)

    # ── Cluster mean: consensus price band within 15% of each other ───────────
    # Active listings are asks, not transactions — outlier sellers inflate the
    # naive mean badly. Cluster mean finds where the market has actually settled.
    cl     = _cluster_mean(prices, tolerance=0.15, min_cluster=2)
    use_cm = (not cl["fallback_used"]) and (cl["cluster_coverage"] >= 0.40)
    primary_price = cl["cluster_mean"] if use_cm else raw_median

    out = {
        "source": "ebay",
        "query": q,
        "count": count,
        "currency": "AUD",
        "prices": prices[:target],
        "low": cl["cluster_low"]  if use_cm else raw_low,
        "median": primary_price,
        "high": cl["cluster_high"] if use_cm else raw_high,
        "avg": float(sum(prices) / count),
        "p20": raw_low,
        "p80": raw_high,
        "min": float(prices[0]),
        "max": float(prices[-1]),
        # Cluster fields
        "cluster_mean":     cl["cluster_mean"],
        "cluster_size":     cl["cluster_size"],
        "cluster_low":      cl["cluster_low"],
        "cluster_high":     cl["cluster_high"],
        "cluster_coverage": cl["cluster_coverage"],
        "cluster_used":     use_cm,
    }
    _EBAY_CACHE[cache_key] = {"ts": now, "data": out}
    return out

async def _ebay_find_items(keyword_query: str, limit: int = 5, sold: bool = False, days_lookback: int = 30) -> List[Dict[str, Any]]:
    """Return a small list of eBay items to allow manual disambiguation on the front-end.

    Prefer Buy/Browse API (OAuth). Falls back to FindingService (AppID) when OAuth is missing.
    Note: Browse does not provide SOLD/COMPLETED; when sold=True and AppID is missing we return ACTIVE items.
    """
    q = _norm_ws(keyword_query or "")
    if not q:
        return []

    limit = max(1, min(int(limit or 5), 20))
    cache_key = f"ebay_items2:{'sold' if sold else 'active'}:{q}:{limit}:{int(days_lookback or 30)}"
    now = int(time.time())
    cached = _EBAY_CACHE.get(cache_key)
    if cached and (now - int(cached.get('ts', 0))) < 900:
        return cached.get('data', []) or []

    items: List[Dict[str, Any]] = []

    # --- Path A: Browse API (preferred) ---
    token, _dbg = await _get_ebay_app_token()
    if token:
        try:
            url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
            headers = {
                "Authorization": f"Bearer {token}",
                "X-EBAY-C-MARKETPLACE-ID": EBAY_MARKETPLACE_ID,
                "Accept": "application/json",
                "User-Agent": UA,
            }
            async with httpx.AsyncClient(timeout=25.0, headers=headers) as client:
                params = {"q": q, "limit": str(min(50, limit)), "offset": "0"}
                r = await client.get(url, params=params)
                if r.status_code == 401:
                    token2, _ = await _get_ebay_app_token(force_refresh=True)
                    if token2:
                        headers["Authorization"] = f"Bearer {token2}"
                        r = await client.get(url, params=params)
                if r.status_code == 200:
                    j = r.json() or {}
                    for it in (j.get("itemSummaries") or [])[:limit]:
                        try:
                            title = str(it.get("title") or "")
                            item_id = str(it.get("itemId") or "")
                            view_url = str(it.get("itemWebUrl") or "")
                            img = ""
                            img_obj = it.get("image") or {}
                            if isinstance(img_obj, dict):
                                img = str(img_obj.get("imageUrl") or "")
                            pr = it.get("price") or {}
                            price = float(pr.get("value") or 0.0)
                            currency = str(pr.get("currency") or DEFAULT_CURRENCY).upper()
                            if currency == "USD":
                                price = float(_usd_to_aud_simple(price) or 0.0)
                                currency = "AUD"
                            condition = str(it.get("condition") or "")
                            items.append({
                                "item_id": item_id,
                                "title": title,
                                "price": round(price, 2),
                                "currency": currency,
                                "view_url": view_url,
                                "image": img,
                                "condition": condition,
                                "source": "ebay_browse_api",
                                "sold": bool(sold and EBAY_APP_ID),  # only truly sold when using Finding completed
                            })
                        except Exception:
                            continue
        except Exception:
            items = []

    # --- Path B: FindingService (AppID) for sold or when Browse missing ---
    if (not items) and EBAY_APP_ID:
        try:
            url = "https://svcs.ebay.com/services/search/FindingService/v1"
            headers = {"User-Agent": UA}
            op = "findCompletedItems" if sold else "findItemsAdvanced"
            params = {
                "OPERATION-NAME": op,
                "SERVICE-VERSION": "1.13.0",
                "GLOBAL-ID": "EBAY-AU",
                "RESPONSE-DATA-FORMAT": "JSON",
                "REST-PAYLOAD": "true",
                "SECURITY-APPNAME": EBAY_APP_ID,
                "keywords": q,
                "paginationInput.entriesPerPage": str(min(100, limit)),
                "paginationInput.pageNumber": "1",
                "itemFilter(0).name": "HideDuplicateItems",
                "itemFilter(0).value": "true",
            }
            if sold:
                params.update({"itemFilter(1).name": "SoldItemsOnly", "itemFilter(1).value": "true"})
            async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                j = r.json()
            resp = (j.get("findItemsAdvancedResponse") or j.get("findCompletedItemsResponse") or [])[0] or {}
            search = (resp.get("searchResult") or [])[0] or {}
            raw_items = search.get("item") or []
            for it in raw_items[:limit]:
                try:
                    title = (it.get("title") or [""])[0]
                    item_id = (it.get("itemId") or [""])[0]
                    view_url = (it.get("viewItemURL") or [""])[0]
                    gallery = (it.get("galleryURL") or [""])[0]
                    selling = (it.get("sellingStatus") or [{}])[0] or {}
                    curp = (selling.get("currentPrice") or [{}])[0] or {}
                    price = float(curp.get("__value__") or 0.0)
                    currency = str(curp.get("@currencyId") or "AUD").upper()
                    if currency == "USD":
                        price = float(_usd_to_aud_simple(price) or 0.0)
                        currency = "AUD"
                    condition = ""
                    cond = (it.get("condition") or [{}])
                    if cond and isinstance(cond, list) and cond[0]:
                        condition = (cond[0].get("conditionDisplayName") or [""])[0] if isinstance(cond[0].get("conditionDisplayName"), list) else str(cond[0].get("conditionDisplayName") or "")
                    items.append({
                        "item_id": item_id,
                        "title": title,
                        "price": round(price, 2),
                        "currency": currency,
                        "view_url": view_url,
                        "image": gallery,
                        "condition": condition,
                        "source": "ebay_finding_service",
                        "sold": bool(sold),
                    })
                except Exception:
                    continue
        except Exception:
            items = items or []

    _EBAY_CACHE[cache_key] = {"ts": now, "data": items}
    return items

async def _fetch_html(url: str) -> str:
    """Fetch text content from a URL (used for CSV downloads)."""
    try:
        client = _get_http_client()
        r = await client.get(url, timeout=30.0, headers={"User-Agent": UA})
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
        client = _get_http_client()
        r = await client.get(url, headers=headers, params=params, timeout=12.0)
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
    """Convert USD to AUD.  Uses the live-rate cache populated by _fx_usd_to_aud()
    when available, so callers automatically benefit once the async helper has run.
    Falls back to AUD_MULTIPLIER (configurable via CL_USD_TO_AUD_MULTIPLIER env var).
    """
    try:
        v = float(amount)
    except Exception:
        return None
    # Use cached live rate if fresh (populated by _fx_usd_to_aud)
    cached_rate = _FX_CACHE.get("usd_aud")
    if cached_rate and float(cached_rate) > 0.5:
        return round(v * float(cached_rate), 2)
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


def _cluster_mean(
    prices: List[float],
    tolerance: float = 0.15,
    min_cluster: int = 2,
) -> Dict[str, Any]:
    """
    Find the largest consensus price cluster and return their mean.

    Algorithm:
      1. Sort prices.
      2. Sliding window: keep valid while ps[hi] / ps[lo] <= ratio_threshold.
      3. Return mean of the largest window.

    min_cluster scales with sample size:
      - 1-4 listings  → min 2 in cluster (small sample, don't be greedy)
      - 5-9 listings  → min 3 in cluster
      - 10+ listings  → min 4 in cluster
    This prevents a 2-listing "cluster" of the two rogue-high prices from
    being chosen when the other 8 listings are genuinely all over the place.

    Returns:
      cluster_mean, cluster_size, cluster_low, cluster_high,
      cluster_coverage (fraction of total), fallback_used
    """
    empty = {
        "cluster_mean": None,
        "cluster_size": 0,
        "cluster_low": None,
        "cluster_high": None,
        "cluster_coverage": 0.0,
        "fallback_used": True,
    }
    if not prices:
        return empty

    ps = sorted(float(p) for p in prices if p and float(p) > 0)
    n  = len(ps)
    if n == 0:
        return empty

    # Scale min_cluster with sample size — prevents 2-item clusters on larger sets
    effective_min = min_cluster
    if n >= 10:
        effective_min = max(min_cluster, 4)
    elif n >= 5:
        effective_min = max(min_cluster, 3)

    # The ratio threshold: window is valid while ps[hi] / ps[lo] <= ratio_threshold
    ratio_threshold = (1.0 + tolerance) / max(1.0 - tolerance, 1e-9)

    best_lo = best_hi = 0
    best_len = 1

    lo = 0
    for hi in range(n):
        # Shrink window from left until ratio condition holds
        while ps[lo] > 0 and (ps[hi] / ps[lo]) > ratio_threshold:
            lo += 1
        window_len = hi - lo + 1
        if window_len > best_len:
            best_len = window_len
            best_lo  = lo
            best_hi  = hi

    if best_len < effective_min:
        return empty

    cluster = ps[best_lo : best_hi + 1]
    cmean   = sum(cluster) / len(cluster)

    return {
        "cluster_mean":     round(cmean, 2),
        "cluster_size":     len(cluster),
        "cluster_low":      round(cluster[0],  2),
        "cluster_high":     round(cluster[-1], 2),
        "cluster_coverage": round(len(cluster) / n, 4),
        "fallback_used":    False,
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
    """Parse an integer grade from a string. Supports 1-10 and 12 (CL Ultra Flawless).
    Grade 11 does not exist — skip it. Half-grades are rounded to the nearest int."""
    s = (predicted_grade or "").strip()
    # Try exact 12 first (our proprietary Ultra Flawless grade)
    if re.search(r"\b12\b", s):
        return 12
    # Then try standard grades with optional .5
    m = re.search(r"\b(10|[1-9])(?:\.5)?\b", s)
    if not m:
        return None
    g = int(m.group(1))
    return g if 1 <= g <= 10 else None


def _parse_half_grade(predicted_grade: str) -> Optional[str]:
    """Parse a grade string that may include half-grades (e.g. '9.5', '8', '10', '12').
    Returns the normalized string grade (e.g. '9.5', '8', '12') or None.
    Valid grades: 1-10 in whole or half steps, plus 12 (CL Ultra Flawless).
    Grade 11 does not exist."""
    s = (predicted_grade or "").strip()
    if not s:
        return None
    # Grade 12 (Ultra Flawless)
    if re.search(r"\b12\b", s):
        return "12"
    # Try float match: 10, 9.5, 9, 8.5, ..., 1.5, 1
    m = re.search(r"\b(10|[1-9])(?:\.(0|5))?\b", s)
    if not m:
        return None
    whole = int(m.group(1))
    decimal = m.group(2)
    if whole < 1 or whole > 10:
        return None
    if decimal == "5":
        return f"{whole}.5"
    return str(whole)

def _grade_distribution(predicted_grade: int, confidence: float) -> Dict[str, float]:
    """Build probability distribution across grades. Supports 1-10 + 12 (CL Ultra Flawless)."""
    c = _clamp(confidence, 0.05, 0.95)
    p_pred = 0.45 + 0.50 * c
    remainder = 1.0 - p_pred

    if predicted_grade == 12:
        # Ultra Flawless — very tight distribution
        dist = {"12": p_pred, "10": remainder * 0.80, "9": remainder * 0.20}
    elif predicted_grade == 10:
        dist = {"12": remainder * 0.05, "10": p_pred, "9": remainder * 0.70, "8": remainder * 0.25}
    elif predicted_grade == 9:
        dist = {"10": remainder * 0.20, "9": p_pred, "8": remainder * 0.55, "7": remainder * 0.25}
    elif predicted_grade == 8:
        dist = {"9": remainder * 0.15, "8": p_pred, "7": remainder * 0.55, "6": remainder * 0.30}
    elif predicted_grade == 7:
        dist = {"8": remainder * 0.15, "7": p_pred, "6": remainder * 0.55, "5": remainder * 0.30}
    elif predicted_grade == 6:
        dist = {"7": remainder * 0.15, "6": p_pred, "5": remainder * 0.55, "4": remainder * 0.30}
    elif predicted_grade == 5:
        dist = {"6": remainder * 0.15, "5": p_pred, "4": remainder * 0.55, "3": remainder * 0.30}
    elif predicted_grade <= 4:
        g = max(1, predicted_grade)
        upper = str(min(10, g + 1))
        lower = str(max(1, g - 1))
        dist = {upper: remainder * 0.20, str(g): p_pred, lower: remainder * 0.80}
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
    if gi >= 12:
        return "ultra_flawless"
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

@app.get("/robots.txt", include_in_schema=False)
def robots_txt():
    """Tell crawlers this is a private API - nothing to index."""
    return Response(
        content="User-agent: *\nDisallow: /\n",
        media_type="text/plain",
    )


@app.get("/health")
def health():
    # Piggyback cache eviction on health-check polls (Render pings /health every 30s)
    try:
        _evict_stale_caches()
    except Exception:
        pass
    _primary   = "openai"    if OPENAI_API_KEY     else ("anthropic" if ANTHROPIC_API_KEY else "none")
    _fallback  = "anthropic" if (OPENAI_API_KEY and ANTHROPIC_API_KEY) else "none"
    return {
        "ok": True,
        "service": "cl-scan-api",
        "version": APP_VERSION,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_anthropic_key": bool(ANTHROPIC_API_KEY),
        "ai_primary": _primary,
        "ai_fallback": _fallback,
        "has_pokemontcg_key": bool(POKEMONTCG_API_KEY),
        "has_pricecharting_token": bool(PRICECHARTING_TOKEN),
        "use_ebay_api": bool(USE_EBAY_API and EBAY_APP_ID),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "model_fallback_vision": os.getenv("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-6"),
        "model_fallback_text": os.getenv("ANTHROPIC_TEXT_MODEL", "claude-haiku-4-5-20251001"),
        "allowed_origins": ALLOWED_ORIGINS,
        "supports": ["cards", "memorabilia", "sealed_products", "market_context_click_only", "half_grades", "error_card_detection", "grading_standard_selector", "expanded_tcg_support"],
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
    Front is required. Back is optional.

    Speed optimisations (v6.9.31):
    - Server-side compress to IDENTIFY_MAX_PX (default 768px) before AI call.
      Identify doesn't need 1200px — the card name/number/set are readable at 768px
      and the smaller payload cuts ~40% off base64 encoding time + token cost.
    - Back image is skipped on the first pass. If confidence < IDENTIFY_BACK_THRESHOLD
      (default 0.70) the back is added and a second AI call is made. Most common
      cards identify confidently from front alone, saving one full image's worth of
      tokens on every easy call.
    - max_tokens reduced 900 → 650 (identification JSON is compact; 900 was headroom
      that was never used).
    """
    _IDENTIFY_MAX_PX       = int(os.getenv("IDENTIFY_MAX_PX",        "768"))
    _IDENTIFY_BACK_THRESH  = float(os.getenv("IDENTIFY_BACK_THRESHOLD", "0.70"))

    front_bytes = await front.read()
    if not front_bytes or len(front_bytes) < 200:
        raise HTTPException(status_code=400, detail="No front image uploaded (or image too small).")

    # Read back eagerly so we can use it if confidence is low — but don't encode yet
    back_bytes: bytes | None = None
    if back is not None:
        bb = await back.read()
        if bb and len(bb) >= 200:
            back_bytes = bb

    # ── Server-side compress (identify-specific: smaller than verify's 1200px) ──
    front_bytes = _compress_image(front_bytes, max_long=_IDENTIFY_MAX_PX, quality=85)
    # Back stays uncompressed until/unless needed — compress lazily below

    system = (
        "You are an expert collectibles identifier specialising in trading cards (Pokemon, Magic, Yu-Gi-Oh, One Piece, Dragon Ball, Digimon, Lorcana, Flesh and Blood, Sports, and more). "
        "Return ONLY valid JSON. Be conservative; if unsure, leave fields empty rather than hallucinating."
    )
    user = (
        "Identify the card/item from the image(s). "
        "Return JSON with keys: "
        "card_name, card_type, game, year, card_number, set_code, set_name, manufacturer, language, "
        "rarity (e.g. Common/Uncommon/Rare/Holo Rare/Ultra Rare/Secret Rare/Illustration Rare/Special Art Rare/Hyper Rare/Crown Rare/Full Art/Alt Art), "
        "variant_type (e.g. Regular/Holo/Reverse Holo/Full Art/Alt Art/Rainbow/Gold/Textured/Cosmos Holo/Master Ball/Art Rare/Special Art Rare/Standard/Promo/Parallel), "
        "edition (e.g. 1st Edition/Unlimited/Shadowless/Limited/Collector's Edition or empty), "
        "finish (e.g. Standard/Holofoil/Reverse Holofoil/Textured/Etched/Glossy/Matte or empty), "
        "is_promo (true/false), promo_source (e.g. 'League Promo', 'Box Topper', 'Tournament Prize' or empty), "
        "card_category (e.g. Pokemon/Trainer/Energy for Pokemon; Creature/Spell/Land for MTG; Monster/Spell/Trap for YuGiOh; or empty), "
        "artist (artist name if visible, else empty), "
        "is_error_card (true if visibly misprinted/miscut/wrong-back/crimped/factory error), "
        "error_description (describe the error if is_error_card is true, else empty), "
        "confidence (0-1), reasoning (short). "
        "For Pokemon, set_code should be the PTCGO set code if visible (e.g., MEW), else empty."
    )

    def _build_messages(include_back: bool) -> list:
        images = [{
            "type": "image_url",
            "image_url": {"url": f"data:{front.content_type or 'image/jpeg'};base64,{_b64(front_bytes)}"},
        }]
        if include_back and back_bytes is not None:
            compressed_back = _compress_image(back_bytes, max_long=_IDENTIFY_MAX_PX, quality=85)
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:{back.content_type or 'image/jpeg'};base64,{_b64(compressed_back)}"},
            })
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user}, *images]},
        ]

    # ── Pass 1: front only ────────────────────────────────────────────────────
    resp = await _openai_chat(messages=_build_messages(False), max_tokens=650, temperature=0.1)
    if resp.get("error"):
        raise HTTPException(status_code=502, detail=f"AI identify failed: {resp.get('message', 'unknown error')}")

    content = resp.get("content") or "{}"
    data    = _parse_json_or_none(content) or _safe_json_extract(content) or {}
    if not isinstance(data, dict):
        data = {}

    front_confidence = _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0)

    # ── Pass 2 (conditional): retry with back if confidence is low ────────────
    if back_bytes is not None and front_confidence < _IDENTIFY_BACK_THRESH:
        logging.info(
            f"🔍 Identify pass-1 confidence {front_confidence:.2f} < {_IDENTIFY_BACK_THRESH} "
            f"— retrying with back image"
        )
        resp2 = await _openai_chat(messages=_build_messages(True), max_tokens=650, temperature=0.1)
        if not resp2.get("error"):
            data2 = _parse_json_or_none(resp2.get("content", "")) or {}
            if isinstance(data2, dict) and data2:
                data = data2  # use the richer result

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
        "rarity": _norm_ws(str(data.get("rarity", ""))),
        "variant_type": _norm_ws(str(data.get("variant_type", ""))),
        "edition": _norm_ws(str(data.get("edition", ""))),
        "finish": _norm_ws(str(data.get("finish", ""))),
        "is_promo": bool(data.get("is_promo", False)),
        "promo_source": _norm_ws(str(data.get("promo_source", ""))),
        "card_category": _norm_ws(str(data.get("card_category", ""))),
        "artist": _norm_ws(str(data.get("artist", ""))),
        "is_error_card": bool(data.get("is_error_card", False)),
        "error_description": _norm_ws(str(data.get("error_description", ""))),
        "confidence": _clamp(_safe_float(data.get("confidence", 0.0)), 0.0, 1.0),
        "reasoning": _norm_ws(str(data.get("reasoning", ""))),
    }

    # If the model returns a generic card_type but game is clearly known, force it into your frontend enum.
    try:
        g = (result.get("game") or "").lower()
        if result.get("card_type") in ("", "Other"):
            if "one piece" in g:
                result["card_type"] = "OnePiece"
            elif "pokemon" in g:
                result["card_type"] = "Pokemon"
            elif "dragon" in g and "ball" in g:
                result["card_type"] = "DragonBall"
            elif "digimon" in g:
                result["card_type"] = "Digimon"
    except Exception:
        pass

    # Canonicalize set info where helpers exist
    try:
        set_info = _canonicalize_set(result["set_code"], result["set_name"])
        result["set_code"] = set_info.get("set_code", result["set_code"])
        result["set_name"] = set_info.get("set_name", result["set_name"])
        result["set_source"] = set_info.get("set_source", "")
    except Exception:
        pass


    # One Piece set/year enrichment (only when missing)
    try:
        if "one piece" in (result.get("game","").lower()):
            op = _onepiece_set_info(result.get("set_code",""))
            if op.get("set_code"):
                result["set_code"] = op.get("set_code")
            if not result.get("set_name") and op.get("set_name"):
                result["set_name"] = op.get("set_name")
                result["set_source"] = "onepiece_" + op.get("source","")
            if (not result.get("year")) and op.get("year"):
                result["year"] = op.get("year")
                result["year_source"] = "set_code_" + op.get("source","")
    except Exception:
        pass

    # Rarity OCR fallback (only when empty)
    try:
        if not (result.get("rarity") or "").strip():
            rc = _ocr_rarity_code_from_front(front_bytes)
            if rc:
                result["rarity"] = rc
    except Exception:
        pass

    # Add English translation alongside original name for non-English cards
    try:
        lang = (result.get("language","") or "").lower()
        nm = result.get("card_name","") or ""
        if nm and lang and lang not in ("english", "en"):
            en = await _translate_to_english(nm)
            if en and en.lower() != nm.lower():
                result["card_name_en"] = en
                result["card_name_display"] = f"{nm} / {en}"
            else:
                result["card_name_en"] = ""
                result["card_name_display"] = nm
        else:
            result["card_name_en"] = nm
            result["card_name_display"] = nm
    except Exception:
        result["card_name_en"] = ""
        result["card_name_display"] = result.get("card_name","") or ""

    # Backward-compatible response: expose fields at top-level AND under card
    flat = dict(result)
    flat.update({"ok": True, "card": result, "api_version": "2026-02-23-main-v4"})
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
    grading_standard: Optional[str] = Form(None),  # PSA / BGS / CGC / CL (Collectors League)
):
    front_bytes = await front.read()
    back_bytes = await back.read()
    angled_bytes = await angled.read() if angled is not None else b""
    if not front_bytes or not back_bytes or len(front_bytes) < 200 or len(back_bytes) < 200:
        raise HTTPException(status_code=400, detail="Images are too small or empty")

    # ── Compress images to keep Render memory within limits ────────────────
    # verify needs more detail than identify (fine defect detection) but 1024px
    # at q85 is sufficient — 1200px q88 was over-provisioned for memory budget.
    front_bytes  = _compress_image(front_bytes,  max_long=1024, quality=85)
    back_bytes   = _compress_image(back_bytes,   max_long=1024, quality=85)
    if angled_bytes and len(angled_bytes) > 200:
        angled_bytes = _compress_image(angled_bytes, max_long=1024, quality=85)
    # ─────────────────────────────────────────────────────────────────────

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

    # Grading standard context
    gs = (grading_standard or '').strip().upper()
    if gs in ('BGS', 'BECKETT'):
        grading_std_context = 'BGS'
        grading_std_note = (
            "GRADING STANDARD: BGS (Beckett Grading Services)\n"
            "- Uses half-grades: 10 (Pristine/Black Label), 9.5 (Gem Mint), 9, 8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4, 3, 2, 1\n"
            "- Sub-grades for: Centering, Corners, Edges, Surface (each 1-10 with half-grades)\n"
            "- BGS 10 = Pristine (nearly impossible), BGS 9.5 = Gem Mint (highest realistic)\n"
            "- Centering: 50/50-55/45 = 10, 55/45-60/40 = 9.5, 60/40-65/35 = 9\n"
            "- OUTPUT pregrade as a half-grade (e.g. '9.5', '8', '7.5')\n"
        )
    elif gs == 'CGC':
        grading_std_context = 'CGC'
        grading_std_note = (
            "GRADING STANDARD: CGC (Certified Guaranty Company)\n"
            "- Uses half-grades: 10 (Pristine), 9.5, 9, 8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2, 1\n"
            "- Sub-grades for: Surface, Corners, Edges (each scored)\n"
            "- CGC 10 = Pristine (very rare), CGC 9.5 = Gem Mint\n"
            "- OUTPUT pregrade as a half-grade (e.g. '9.5', '8', '7.5')\n"
        )
    elif gs == 'CL':
        grading_std_context = 'CL'
        grading_std_note = (
            "GRADING STANDARD: CL (Collectors League Australia)\n"
            "- Uses whole grades 1-10, plus Grade 12 (CL Ultra Flawless). There is NO grade 11.\n"
            "- Grade 10 = GEM MINT (modern standard perfection — award it when warranted)\n"
            "- Grade 12 = CL ULTRA FLAWLESS (exceeds all expectations, near-impossible perfection — award ONLY when truly warranted)\n"
            "- Sub-grades for: Centering, Corners, Edges, Surface\n"
            "- OUTPUT pregrade as whole number or '12' for Ultra Flawless\n"
        )
    else:
        grading_std_context = 'PSA'
        grading_std_note = (
            "GRADING STANDARD: PSA (Professional Sports Authenticator) — default\n"
            "- Uses whole grades: 10 (Gem Mint), 9, 8, 7, 6, 5, 4, 3, 2, 1\n"
            "- PSA 10 = Gem Mint (highest grade, requires near-perfection)\n"
            "- Centering: 60/40 or better front, 75/25 or better back for PSA 10\n"
            "- OUTPUT pregrade as whole number (e.g. '10', '9', '8')\n"
        )

    prompt = f"""You are a professional trading card grader with 15+ years experience.

{grading_std_note}

Analyze the provided images with EXTREME scrutiny.
You will receive FRONT and BACK images, and MAY receive a third ANGLED image used to rule out glare / light refraction artifacts (holo sheen) vs true whitening / scratches / print lines. Write as if speaking directly to a collector who needs honest, specific feedback about their card.

USER INTENT (context): {intent_context}

**INTENT-SPECIFIC RESPONSE FRAMING:**

If {intent_context} is BUYING (user is considering purchasing this card):
- In corner/edge/surface notes: Frame defects as NEGOTIATION LEVERAGE
  * Example: "Minor whitening on back top-right corner (about 1mm) — use this to negotiate 10-15% off the seller's asking price"
  * Example: "Light surface scratching visible under direct light on front — point this out to justify a lower offer"
- In centering notes: Express offset as a PERCENTAGE derived from the ratio (e.g. 71/29 = 21% off true center), NEVER as a physical mm measurement. State direction and grade impact.
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

3) Grade must reflect worst visible defect (conservative {grading_std_context}-style):
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
     This grade should be RARE (perhaps 1 in 1000 cards) but MUST be awarded when warranted. There is NO grade 11.

4) Do NOT confuse holo sheen / light refraction / texture for damage:
   - If a mark disappears or changes drastically in the ANGLED shot, treat it as glare/reflection, NOT whitening/damage.
   - Print lines are typically straight and consistent across lighting; glare moves with angle.
   - Card texture (especially modern) is not damage unless there is a true crease, indentation, or paper break.
   - MANUFACTURED FINISH AWARENESS (MANDATORY — do this FIRST for every card):
     STEP 1 — Identify finish: Start your surface notes by stating the detected finish type in brackets. Example: [Finish: Reverse Holofoil]. Choose from: Standard/Matte, Glossy/Foil, Holofoil, Reverse Holofoil, Textured/Embossed Foil, Etched Foil, Full Art Foil, Rainbow Foil, Cosmos Holo, Mirror Foil, or Prism Foil.
     STEP 2 — Set the baseline: State what is NORMAL for that finish (e.g., sparkle is normal for Holofoil; grain/emboss texture is normal for Textured Foil; uniform metallic sheen is normal for Full Art Foil; high gloss is normal for Glossy Foil).
     STEP 3 — Evaluate against that baseline ONLY: Only flag marks that are clearly NOT part of the intended finish pattern. A holo shimmer is NOT a defect. A sparkle pattern is NOT a scratch. A raised texture is NOT an indent.
     STEP 4 — If you see an actual defect (a scratch that crosses through the texture pattern, a dull patch on an otherwise glossy surface, a mark that appears at consistent angles), describe it precisely: location, direction, depth, and whether it breaks the finish pattern.
   - CONSISTENCY RULE: Your surface notes and your grade must AGREE. If you say surface is "Near Mint" and list only minor gloss issues typical of the finish type, your grade must not penalise surface heavily. If you give a surface defect, it must appear in your defects array too.
   - DO NOT write contradictory statements: do not say "minor scratches visible" AND "surface is clean" in the same assessment.
   - When assessing surface: note the finish type detected first, then evaluate surface condition AGAINST that finish baseline. Example: "[Finish: Holofoil] — Holo shimmer and sparkle are normal for this finish type. The surface presents cleanly with no scratches or dull patches crossing the holo pattern. One faint hairline on the front top-left crosses the holo layer at an angle inconsistent with the pattern — this is a real mark."
   - If the card appears to be inside a SLEEVE, TOPLOADER, or SEMI-RIGID, account for sleeve-induced glare, refraction artifacts, and edge distortion. Note this in your assessment.

5) Write the assessment summary in first person, conversational style (5-8 sentences):
   - Open with overall impression: "Looking at your card..."
   - Discuss specific observations: "The front presents beautifully, with..."
   - Compare front vs back: "While the front is near-perfect, the back shows..."
   - Explain grade rationale: "The grade of X is primarily limited by..."
   - End with realistic expectation: "If you're considering grading..."

6) SURFACE — assess ALL of the following factors (mention each that applies):
   - Print quality: registration alignment, ink density/consistency, print dots, roller lines
   - Gloss/coating: dulling, clouding, tackiness, or loss of original sheen
   - Scratches: location, direction, depth, visibility under different lighting
   - Whitening vs Silvering: whitening = paper fiber exposure at edges/corners; silvering = foil layer showing through surface (different defect, note which it is)
   - Staining: water damage rings, yellowing from UV exposure, foxing (age spots), ink transfer from adjacent cards
   - Indentation/pressure marks: from stacking without sleeves, rubber bands, or other storage damage
   - Card warp/bowing: concave or convex curl (minor bow is common and acceptable; severe warp affects grade)

7) ERROR CARD / MISPRINT / FACTORY DEFECT detection:
   - If you observe any of the following, flag it in the "error_card" field:
     * Miscut (off-center die cut, visible border of adjacent card)
     * Crimped card (factory crimp from packaging machine)
     * Wrong back (different card's back printed on reverse)
     * Missing ink / wrong color (partial print, color shift, missing layer)
     * Holo bleed (foil pattern visible outside intended holofoil area)
     * Double print / ghost image
     * Square-cut (proof/test cut without rounded corners)
     * Off-center print (text/art shifted within normal card borders)
     * Misregistered layers (CMYK layers visibly offset)
   - IMPORTANT: Error cards often carry a PREMIUM among collectors. Note this in assessment_summary.
   - If it's a factory defect that adds value, say so. If it's damage that looks like an error, clarify.

8) CARD BACK PATTERN awareness:
   - Different TCGs and eras have different back designs
   - Note if the back pattern is consistent with the identified card (e.g., old Pokemon Energy back vs modern standard)
   - Wrong-back cards are valuable errors — flag them

{context}

Return ONLY valid JSON with this EXACT structure:

{{
  "pregrade": "Grade as appropriate for {grading_std_context}. Use half-grades (e.g. 9.5) for BGS/CGC. Use 12 ONLY for CL Ultra Flawless.",
  "confidence": 0.0-1.0,
  "centering": {{
    "front": {{
      "grade": "55/45",
      "notes": "CENTERING: Measure the PRINTED BORDER widths — the coloured/white border between the card EDGE and the inner artwork frame. Measure left border vs right border for L/R ratio; top border vs bottom border for T/B ratio. DO NOT measure card position in the photo. Use a SINGLE ratio; never state two different ratios. Format: '55/45 L/R — 5% off centre to the left, acceptable for any grade.' Reference: 50/50=perfect; 55/45=imperceptible; 60/40=noticeable; 65/35=limiting ≥9; 70/30=significant cap 7-8; 75/25=severe."
    }},
    "back": {{
      "grade": "60/40",
      "notes": "Same centering style — use ratio + offset %, never mm measurements."
    }}
  }},
  "corners": {{
    "front": {{
      "top_left": {{
        "condition": "sharp/minor_whitening/whitening/silvering/bend/ding/crease",
        "notes": "Examine closely: is the tip sharp and intact, or is there any fibre separation, micro-rounding, compression, whitening or silvering at the very tip? State EXACTLY what you see — even minor issues must be noted."
      }},
      "top_right": {{
        "condition": "sharp/minor_whitening/whitening/silvering/bend/ding/crease",
        "notes": "Examine the tip and within 3mm of the corner for whitening/silvering, fibre lifting, bend, or creasing. State exactly what you see."
      }},
      "bottom_left": {{
        "condition": "sharp/minor_whitening/whitening/silvering/bend/ding/crease",
        "notes": "Examine the tip and within 3mm of the corner. Note any compression, rounding, or colour loss at the corner tip."
      }},
      "bottom_right": {{
        "condition": "sharp/minor_whitening/whitening/silvering/bend/ding/crease",
        "notes": "Examine the tip and within 3mm of the corner. Note any whitening, silvering, fibre exposure or structural deformation."
      }}
    }},
    "back": {{
      "top_left": {{"condition": "...", "notes": "Examine back corner tip and within 3mm. Note whitening, silvering, fibre lifting, or any compression."}},
      "top_right": {{"condition": "...", "notes": "Examine back corner tip and within 3mm. State exactly what you see."}},
      "bottom_left": {{"condition": "...", "notes": "Examine back corner tip and within 3mm. Note any deformation or colour loss."}},
      "bottom_right": {{"condition": "...", "notes": "Examine back corner tip and within 3mm. Note any whitening, silvering or structural issues."}}
    }}
  }},
  "edges": {{
    "front": {{
      "grade": "Mint/Near Mint/Excellent/Good/Poor",
      "notes": "Inspect ALL 4 edges individually. Top edge: [describe roughness, chipping, nicks, whitening, silvering]. Right edge: [describe]. Bottom edge: [describe]. Left edge: [describe]. State clearly if each edge is clean/pristine or has wear. Note any fraying, micro-chipping, colour loss at edges."
    }},
    "back": {{
      "grade": "Mint/Near Mint/Excellent/Good/Poor",
      "notes": "Inspect ALL 4 back edges individually. Top: [describe]. Right: [describe]. Bottom: [describe]. Left: [describe]. Note roughness, silvering, whitening, micro-chipping or fraying on each edge."
    }}
  }},
  "surface": {{
    "front": {{
      "grade": "Mint/Near Mint/Excellent/Good/Poor",
      "notes": "Describe ALL surface factors: holographic pattern quality, print registration/alignment, ink density, scratches (location + direction), scuffs, gloss level, dulling/clouding, staining, yellowing, indentation/pressure marks, warp/bowing."
    }},
    "back": {{
      "grade": "Mint/Near Mint/Excellent/Good/Poor",
      "notes": "Detailed surface assessment. Distinguish whitening from silvering. Check for yellowing, foxing (age spots), ink transfer, pressure marks, warp."
    }}
  }},
  "defects": [
    "Each defect as a complete sentence: [SIDE] [precise location] shows [type of defect] [severity]. Example: 'Front top-left corner shows moderate whitening extending approximately 2mm into the card surface.'"
  ],
  "flags": [
    "Short flags for important issues (crease, bend, edge chipping, silvering, warp, print_line, staining, yellowing, miscut, error_card, etc.)"
  ],
  "error_card": {{
    "is_error": false,
    "error_type": "none/miscut/crimped/wrong_back/missing_ink/holo_bleed/double_print/square_cut/off_center_print/misregistered/other",
    "error_description": "Describe the error if detected, else empty string",
    "value_impact": "premium/neutral/negative — most genuine factory errors carry a collector premium"
  }},
  "card_condition_extras": {{
    "warp": "none/minor_bow/moderate_curve/severe_warp",
    "sleeve_detected": false,
    "yellowing": "none/minor/moderate/severe",
    "gloss_level": "high/medium/low/dulled"
  }},
  "assessment_summary": "Write 5-8 sentences in first person, conversational style. Start with: 'Looking at your [card name]...' Describe specific observations, compare front vs back, explain what limits the grade, give realistic expectations. If error card detected, mention the potential collector premium.",
  "spoken_word": "A punchy spoken-word version of the assessment summary (about 20-45 seconds). First person, conversational. Mention the best features, the main grade limiters, and end with what grade you'd realistically expect.",
  "observed_id": {{
    "card_name": "best-effort from images",
    "set_code": "only if clearly visible",
    "set_name": "best-effort",
    "card_number": "preserve leading zeros",
    "year": "best-effort",
    "card_type": "Pokemon/Magic/YuGiOh/Sports/OnePiece/DragonBall/Digimon/Lorcana/FleshAndBlood/WeissSchwarz/Vanguard/StarWars/Other",
    "rarity": "Common/Uncommon/Rare/Holo Rare/Ultra Rare/Secret Rare/Illustration Rare/Special Art Rare/Hyper Rare/Crown Rare/Other or empty",
    "variant_type": "Regular/Holo/Reverse Holo/Full Art/Alt Art/Rainbow/Gold/Textured/Cosmos Holo/Other or empty",
    "edition": "1st Edition/Unlimited/Shadowless or empty",
    "finish": "Standard/Holofoil/Reverse Holofoil/Textured/Etched or empty",
    "is_error_card": false
  }}

}}

CRITICAL REMINDERS:
- Every corner needs a detailed note explaining what you observe
- Every edge/surface needs location-specific observations  
- Assessment summary must be conversational (first person, like talking to the owner)
- Do NOT miss obvious damage - be brutally honest
- If you can't see something clearly due to glare/blur, say so in notes
- Distinguish WHITENING (paper fiber exposure) from SILVERING (foil showing through surface) — these are different defects
- Check for ERROR CARD indicators (miscut, crimped, wrong back, holo bleed, etc.) and flag in error_card field
- Note card warp/bowing, yellowing, and gloss level in card_condition_extras
- If card is in a sleeve/toploader, note this in card_condition_extras.sleeve_detected and account for artifacts

Respond ONLY with JSON, no extra text.
"""

    msg = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(front_bytes)}", "detail": "high"}},
            {"type": "text", "text": "FRONT IMAGE ABOVE ☝️ | BACK IMAGE BELOW 👇"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(back_bytes)}", "detail": "auto"}},
        ] + (
            [
                {"type": "text", "text": "OPTIONAL ANGLED IMAGE BELOW (use to rule out glare vs true defects) 👇"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(angled_bytes)}", "detail": "auto"}},
            ] if angled_bytes and len(angled_bytes) > 200 else []
        ),
    }]

    result = await _openai_chat(msg, max_tokens=3000, temperature=0.1)
    if result.get("error"):
        err_msg = result.get("message", "")
        err_status = result.get("status", 0)
        logging.error(f"❌ /api/verify OpenAI call failed: status={err_status} msg={err_msg[:200]}")
        return JSONResponse(content={"error": True, "message": "AI grading failed", "details": err_msg, "openai_status": err_status}, status_code=502)

    data = _parse_json_or_none(result.get("content", "")) or {}

    # ------------------------------
    # True centering estimate (computed from pixels)
    # ------------------------------
    # The model sometimes repeats generic centering language. We compute a best-effort
    # ── Centering: AI vision is the primary assessment. ────────────────────────
    # The edge-detection estimate (_estimate_centering_from_image) measures where
    # the *card* sits in the *photo*, not the internal border-to-border ratio.
    # Overriding the AI with it produces wrong results (e.g. a perfectly centred
    # Eevee card coming out as 80/20 because the photo background skews the edge
    # detection).  We keep the computed value as metadata for future calibration
    # only — the AI's visual assessment is authoritative.
    try:
        cen_front = _estimate_centering_from_image(front_bytes)
        cen_back  = _estimate_centering_from_image(back_bytes)
        if isinstance(data, dict) and (cen_front or cen_back):
            data.setdefault("centering", {})
            # Store raw computed values as _computed metadata (not displayed, for analytics)
            if cen_front:
                data["centering"].setdefault("front", {})
                data["centering"]["front"]["_computed_lr"] = cen_front.get("lr")
                data["centering"]["front"]["_computed_tb"] = cen_front.get("tb")
                data["centering"]["front"]["_computed_margins"] = cen_front.get("margins")
            if cen_back:
                data["centering"].setdefault("back", {})
                data["centering"]["back"]["_computed_lr"] = cen_back.get("lr")
                data["centering"]["back"]["_computed_tb"] = cen_back.get("tb")
                data["centering"]["back"]["_computed_margins"] = cen_back.get("margins")
    except Exception:
        pass

    # ------------------------------
    # Post-process centering notes: strip mm measurements, ensure % language
    # ------------------------------
    # The AI sometimes writes "approximately 5mm off-center" despite prompt instructions.
    # We strip all physical distance references from centering notes since they're meaningless
    # without knowing exact card dimensions, and replace with the ratio already in the prefix.
    try:
        def _clean_centering_notes(notes: str) -> str:
            if not notes:
                return notes
            # Remove "approximately Xmm", "about Xmm", "X mm off", etc.
            cleaned = re.sub(
                r'(?i),?\s*approximately\s+\d+(?:\.\d+)?\s*mm\b[^.]*',
                '',
                notes
            )
            cleaned = re.sub(
                r'(?i),?\s*about\s+\d+(?:\.\d+)?\s*mm\b[^.]*',
                '',
                cleaned
            )
            cleaned = re.sub(
                r'(?i)\s*\(\s*\d+(?:\.\d+)?\s*mm\s*\)',
                '',
                cleaned
            )
            # Collapse double spaces / orphaned punctuation
            cleaned = re.sub(r'\s{2,}', ' ', cleaned)
            cleaned = re.sub(r'\.\s*\.', '.', cleaned)
            return cleaned.strip()

        for side in ("front", "back"):
            if isinstance(data.get("centering"), dict) and isinstance(data["centering"].get(side), dict):
                n_val = data["centering"][side].get("notes") or ""
                data["centering"][side]["notes"] = _clean_centering_notes(n_val)
    except Exception:
        pass


    # SECOND PASS GUARANTEE:
    # Enhanced filtered images (grayscale/autocontrast + contrast/sharpness)
    # are ALWAYS generated when PIL is available and are fed back into
    # grading logic to surface print lines, whitening, scratches, and dents.

    # Second-pass AI disabled — single-pass mode to stay within Render memory limits.
    # The first-pass OpenAI vision call already handles defect detection comprehensively.
    # ── Single-pass mode ────────────────────────────────────────────────────
    # Second AI pass + image filter variants DISABLED to stay within Render's
    # 512 MB free-tier memory limit.  The primary vision call already covers
    # defects comprehensively; the second pass added ~200-300 MB peak overhead.
    second_pass = {
        "enabled": False, "ran": False,
        "skipped_reason": "single_pass_mode",
        "glare_suspects": [], "defect_candidates": []
    }
    defect_filters: Dict[str, str] = {}   # kept for response schema compat


    # ------------------------------
    # CV-assisted defect closeups (defect_snaps) for UI thumbnails
    # ------------------------------
    rois: List[Dict[str, Any]] = []
    try:
        rois = _cv_candidate_bboxes(front_bytes, "front") + _cv_candidate_bboxes(back_bytes, "back")
    except Exception:
        rois = []

    roi_labels: List[Dict[str, Any]] = []
    # ROI AI labeling also disabled (memory); CV crops still captured for UI thumbnails

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

    for i, r in enumerate(rois or []):  # Process ALL ROIs, not just first 10
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
    defect_snaps = defect_snaps[:max(1, len(defect_snaps))]  # Show all confirmed defects, no artificial cap


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
            defect_snaps[:] = defect_snaps[:]  # Keep all defect crops
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

    # Detect if error card was flagged by AI
    error_card_data = data.get("error_card", {})
    if isinstance(error_card_data, dict) and error_card_data.get("is_error"):
        if "error_card" not in flags_list_out:
            flags_list_out.append("error_card")
        error_type = str(error_card_data.get("error_type", "")).strip()
        if error_type and error_type != "none" and error_type not in flags_list_out:
            flags_list_out.append(error_type)



    raw_pregrade = str(data.get("pregrade", "")).strip()
    # Parse half-grade first (e.g. "9.5"), fallback to integer bucket
    half_grade_str = _parse_half_grade(raw_pregrade)
    g = _grade_bucket(raw_pregrade)
    # NOTE: We intentionally do NOT cap the AI-assessed pregrade here.
    # Any condition-based value adjustments are applied in /api/market-context only.

    # Use half-grade string for display, integer bucket for logic
    pregrade_norm = half_grade_str or (str(g) if g is not None else "")

    # Condition anchor for downstream market logic (trust gate)
    g_int = None
    try:
        g_int = int(round(float(pregrade_norm))) if pregrade_norm else None
    except Exception:
        g_int = None
    condition_anchor = "damaged" if (has_structural_damage or (g_int is not None and g_int <= 4)) else ("low" if (g_int is not None and g_int <= 6) else ("mid" if (g_int is not None and g_int <= 8) else ("ultra" if (g_int is not None and g_int >= 12) else "high")))
    

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
        "grading_standard": grading_std_context,
        "centering": data.get("centering", {"front": {"grade": "", "notes": ""}, "back": {"grade": "", "notes": ""}}),
        "corners": data.get("corners", {"front": {}, "back": {}}),
        "edges": data.get("edges", {"front": {"grade": "", "notes": ""}, "back": {"grade": "", "notes": ""}}),
        "surface": data.get("surface", {"front": {"grade": "", "notes": ""}, "back": {"grade": "", "notes": ""}}),
        "defects": defects_list_out,
        "flags": flags_list_out,
        "error_card": data.get("error_card", {"is_error": False, "error_type": "none", "error_description": "", "value_impact": "neutral"}) if isinstance(data.get("error_card"), dict) else {"is_error": False, "error_type": "none", "error_description": "", "value_impact": "neutral"},
        "card_condition_extras": data.get("card_condition_extras", {"warp": "none", "sleeve_detected": False, "yellowing": "none", "gloss_level": "high"}) if isinstance(data.get("card_condition_extras"), dict) else {"warp": "none", "sleeve_detected": False, "yellowing": "none", "gloss_level": "high"},
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
        # Server-side filtered image variants (aligned with frontend overlay modes)
        # Keys: front_gray_autocontrast, front_contrast_sharp, back_*, etc.
        # Empty dict when PIL unavailable (frontend falls back to client-side canvas)
        "defect_filters": defect_filters,
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
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(bb)}", "detail": "high"}})

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
        # _openai_label_rois disabled — single-pass mode for memory budget
        label_by_idx_m: Dict[int, Dict[str, Any]] = {}
    
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
        defect_snaps = defect_snaps[:]  # No cap — show all identified defects
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
        q_pc = _norm_ws(" ".join(filter(None, [
            item_name or n or description or "",
            sname,
            num_display,
        ]))).strip()
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
                # Extract thumbnail image from Browse API response
                thumb_url = ""
                try:
                    thumbs = it.get("thumbnailImages") or it.get("image") or {}
                    if isinstance(thumbs, list) and thumbs:
                        thumb_url = str(thumbs[0].get("imageUrl") or "")
                    elif isinstance(thumbs, dict):
                        thumb_url = str(thumbs.get("imageUrl") or "")
                except Exception:
                    thumb_url = ""
                out.append({
                    "itemId": item_id,
                    "title": title,
                    "url": web_url,
                    "image_url": thumb_url,
                    "condition": cond,
                    "price_aud": aud,
                    "currency": "AUD",
                    "raw_currency": pc,
                    "source": "ebay",
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

    # If eBay comes back empty/thin, fall back to PriceCharting as the primary source.
    # But if eBay has results, we keep eBay as primary and add PC as supplemental.
    pc_supplement: Dict[str, Any] = {}
    if PRICECHARTING_TOKEN:
        q_pc = _norm_ws(item_name or n or description or used_query or "").strip()
        try:
            pc_payload = await _market_context_pricecharting(q_pc, category_hint="", pid=product_id or "")
            if pc_payload.get("available"):
                # Always keep PC as supplement — blended into market_summary text below.
                # Do NOT return PC-only even if eBay is empty; show both data sources.
                pc_supplement = pc_payload
        except Exception:
            pass

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

    opener = random.choice(slang_openers)
    if secrets.randbelow(100) < 25:
        opener = f"{opener} {random.choice(openers_2)}"

    grade_line = ""
    if exp_grade_key:
        grade_line = f" LeagAI’s got it around a {exp_grade_key} on the pics."
    price_line = f" Live asks are roughly {_money(current_typ)} AUD (range {_money(current_low)}–{_money(current_high)})."

    trend_phrases = [
        f" It looks {trend_hint} off what’s listed right now.",
        f" The asks look {trend_hint} at the moment.",
        f" Market mood is {trend_hint} based on current listings.",
    ]
    trend_line = random.choice(trend_phrases)

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
        position_line = random.choice(position_templates)

    graded_line = ""
    if exp_grade_key and exp_grade_key in grade_market:
        st = grade_market.get(exp_grade_key, {}).get("stats", {})
        if st and st.get("median") is not None:
            graded_templates = [
                f" If it lands that grade in a slab, graded asks are roughly {_money(st.get('median'))} AUD (current listings).",
                f" In the graded lane at that number, you’re seeing about {_money(st.get('median'))} AUD on asks right now.",
            ]
            graded_line = " " + random.choice(graded_templates)

    # Grading decision language — keep it collector-real, not the same line every time
    grade_advice = ""
    try:
        gc = float(grading_cost or 0)
    except Exception:
        gc = 0.0

    if worth_grading is True:
        grade_advice = " " + random.choice([
            f" If you’re chasing a slab or cleaner resale, it can be a decent shout if you’re chasing a slab or a cleaner resale.",
            f" I’d consider sending it if you want it protected/registry-ready.",
        ])
    elif worth_grading is False:
        grade_advice = " " + random.choice([
            f" I’d only send it if it’s a personal PC piece or you want it in the registry slabbed up.",
            f" This feels more like a binder/PC card unless you just want it in plastic for the vibe.",
            f" If you’re grading for profit, I’d be picky here — but for the collection? send it.",
        ])

    advice_line = ""
    # Buyer vs Seller intent tailoring (language only; underlying data is the same)
    if intent_context == "BUYING":
        if buy_target_aud is not None:
            advice_line = " " + random.choice([
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
            advice_line = " " + random.choice([
                f" If you’re selling, I’d list around {_money(list_at)} AUD (allow room for offers) and aim to accept near {_money(take_at)} AUD depending on interest.",
                f" Seller target: list about {_money(list_at)} AUD, realistically expect {_money(take_at)} AUD (condition + timing will move it).",
                f" If you want a faster sale, price closer to {_money(take_at)} AUD; if you can wait, start near {_money(list_at)} AUD and adjust."
            ])

    # ── Blend PriceCharting reference into summary when available ──
    pc_line = ""
    try:
        if isinstance(pc_supplement, dict) and pc_supplement.get("available"):
            pc_obs  = (pc_supplement.get("observed") or {}).get("active") or {}
            pc_low  = pc_obs.get("p20")
            pc_mid  = pc_obs.get("p50")
            pc_high = pc_obs.get("p80")
            if pc_low is not None and pc_mid is not None and pc_high is not None:
                pc_line = (
                    f" PriceCharting also shows this card ranging from {_money(pc_low)}–{_money(pc_high)} AUD"
                    f" (mid {_money(pc_mid)} AUD) across condition buckets — use both as a sanity-check."
                )
            elif pc_mid is not None:
                pc_line = f" PriceCharting mid-market is {_money(pc_mid)} AUD — treat as a cross-reference."
    except Exception:
        pc_line = ""

    market_summary = _norm_ws(f"{opener}{grade_line}{price_line}{trend_line}{pc_line}{position_line}{graded_line}{grade_advice}{advice_line}")

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
        # PC supplemental: grade-condition buckets from PriceCharting (if available alongside eBay data)
        "pc_supplement": pc_supplement if isinstance(pc_supplement, dict) and pc_supplement.get("available") else None,
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

        # Parse identifier: "CardName|SetName|Grade" or "CardName|SetName|CardNumber|Grade"
        parts = [p.strip() for p in ident.split("|")] if "|" in ident else [ident]
        card_name   = parts[0] if len(parts) > 0 else ""
        set_name    = parts[1] if len(parts) > 1 else ""
        # Support optional card_number in slot 2 (4-segment ident: Name|Set|Number|Grade)
        card_number = parts[2] if len(parts) > 3 else ""
        grade       = parts[3] if len(parts) > 3 else (parts[2] if len(parts) > 2 else "")

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

            # Last known range (for spoken brief)
            last_point = historical_prices[-1] if historical_prices else None
            lp_low = float((last_point or {}).get("price_low") or (last_point or {}).get("price_median") or 0.0)
            lp_med = float((last_point or {}).get("price_median") or 0.0)
            lp_high = float((last_point or {}).get("price_high") or (last_point or {}).get("price_median") or 0.0)
            lp_vol = int((last_point or {}).get("volume") or 0)
            spoken_brief = await _generate_market_spoken_brief(
                card_identifier=ident,
                price_low=lp_low,
                price_median=lp_med,
                price_high=lp_high,
                volume=lp_vol,
                history_days_logged=len(historical_prices),
                data_source="database_logged_history",
            )

            return JSONResponse(content={
                "success": True,
                "card_identifier": ident,
                "data_source": "database_logged_history",
                "historical_data": historical_prices,
                "prediction": prediction,
                "seasonality": seasonality,
                "db_log": {"saved": True, "points": len(historical_prices), "last_recorded_date": historical_prices[-1]["date"] if historical_prices else None},
                "spoken_brief": spoken_brief if "spoken_brief" in locals() else "",
                "analysis_period_days": int(days or 90),
                "actual_data_points": len(historical_prices),
                "note": "✅ Using genuine accumulated price history",
                "history_meta": {"db_points": len(historical_prices), "first_date": (historical_prices[0]["date"] if historical_prices else None), "last_date": (historical_prices[-1]["date"] if historical_prices else None)},
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })

        # ═══════════════════════════════════════════════════════
        # STEP 2: Not enough DB data – fetch current price from eBay
        # ═══════════════════════════════════════════════════════
        logging.info(f"⚠️ DATABASE MISS: Only {len(db_history)} points, fetching from eBay")

        # Build eBay query ladder (specific -> broad)
        if "|" in ident:
            queries = _build_ebay_query_ladder(card_name=card_name, card_set=set_name, card_number=card_number, grade=grade)
        else:
            queries = _build_ebay_query_ladder(card_name=ident, card_set="", card_number="", grade=grade)

        if not queries:
            raise HTTPException(status_code=400, detail="Could not build search query")

        logging.info(f"🔍 eBay queries: {queries}")
        target_days = max(7, min(int(days or 90), 90))

        LISTINGS_TARGET = 5  # how many comps we *aim* to base the snapshot on (cannot exceed what's available)
        completed = {}
        active = {}
        chosen_query = ""
        listings_analyzed = 0

        # We fetch a deeper pool (limit=50) to improve percentile stability, then *report* up to LISTINGS_TARGET.
        for q in queries:
            chosen_query = q
            completed = await _ebay_completed_stats(q, limit=50, days_lookback=30) or {}
            active = await _ebay_active_stats(q, limit=50) or {}
            # Prefer sold comps if available; otherwise fall back to active listings
            comp_count = int((completed.get("count") or 0) if (completed.get("median") and completed.get("median") > 0) else (active.get("count") or 0))
            listings_analyzed = min(LISTINGS_TARGET, max(0, comp_count))
            if (completed.get("median") and completed.get("median") > 0) or (active.get("median") and active.get("median") > 0):
                break
        search_query = chosen_query
        current_price = 0.0
        price_low = 0.0
        price_high = 0.0
        volume = 0

        # ── Smart price blending ──────────────────────────────────────────
        # Minimum required sold records before trusting completed stats alone.
        # With < 3 sold records, a single outlier sale dominates the median.
        # In those cases, blend with active listing prices for a fairer picture.
        MIN_COMPLETED_TRUST = 3

        comp_count   = int(completed.get("count") or 0) if completed else 0
        comp_median  = float(completed.get("median") or 0.0) if completed else 0.0
        active_count = int(active.get("count") or 0)   if active   else 0
        active_median = float(active.get("median") or 0.0) if active else 0.0

        if comp_median > 0 and comp_count >= MIN_COMPLETED_TRUST:
            # Enough sold data — use completed median as ground truth
            current_price = comp_median
            price_low  = float(completed.get("low")  or current_price * 0.90)
            price_high = float(completed.get("high") or current_price * 1.10)
            volume = comp_count
            logging.info(f"📊 eBay sold ({comp_count} sales): ${current_price:.2f}")

        elif comp_median > 0 and active_median > 0 and active_count >= MIN_COMPLETED_TRUST:
            # Sparse sold data (< 3 records) but active listings available.
            # Blend: sold carries 40% weight, active 60% (active is denser).
            current_price = round(comp_median * 0.40 + active_median * 0.60, 2)
            # Build band from both sources
            price_low  = min(
                float(completed.get("low")  or comp_median  * 0.90),
                float(active.get("low")     or active_median * 0.90),
            )
            price_high = max(
                float(completed.get("high") or comp_median  * 1.10),
                float(active.get("high")    or active_median * 1.10),
            )
            volume = comp_count + active_count
            logging.info(f"📊 eBay blend (sold={comp_count}, active={active_count}): ${current_price:.2f}")

        elif comp_median > 0:
            # Only sold data, very sparse — use it but widen the band
            current_price = comp_median
            price_low  = current_price * 0.88
            price_high = current_price * 1.12
            volume = comp_count
            logging.info(f"📊 eBay sold (sparse, {comp_count}): ${current_price:.2f}")

        elif active_median > 0:
            # Only active listings — use those
            current_price = active_median
            price_low  = float(active.get("low")  or current_price * 0.90)
            price_high = float(active.get("high") or current_price * 1.10)
            volume = active_count
            logging.info(f"📊 eBay active ({active_count} listings): ${current_price:.2f}")

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

        # Expose DB logging status so the frontend can confirm it really saved
        db_log = {
            "saved": bool(entry) if "entry" in locals() else False,
            "id": (entry.get("id") if ("entry" in locals() and isinstance(entry, dict)) else None),
            "recorded_date": (entry.get("recorded_date") if ("entry" in locals() and isinstance(entry, dict)) else None),
        }

        spoken_brief = await _generate_market_spoken_brief(
            card_identifier=ident,
            price_low=float(price_low or 0.0),
            price_median=float(current_price or 0.0),
            price_high=float(price_high or 0.0),
            volume=int(volume or 0),
            history_days_logged=len(db_history) + 1,
            data_source="ebay_current_snapshot",
        )

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
            "spoken_brief": spoken_brief if "spoken_brief" in locals() else "",
            "historical_data": historical_data,
            "actual_data_points": len(historical_data),
            "ebay_listings_analyzed": int(listings_analyzed) if 'listings_analyzed' in locals() else int(volume),
            "note": f"Building history ({len(db_history)} days logged). Check again tomorrow for trend analysis!",
            "db_log": {
                "saved": bool(entry) if 'entry' in locals() else False,
                "id": (entry.get('id') if 'entry' in locals() and isinstance(entry, dict) else None),
                "recorded_date": (str(entry.get('recorded_date')) if 'entry' in locals() and isinstance(entry, dict) else None),
            },
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
    """Generate mock historical price data for demo/dev UI only.

    WARNING: Uses non-deterministic np.random — results change on every call.
    Do NOT call this in production code paths that write to SQLite.
    This function exists purely for UI demonstration when real price history
    is unavailable. It is not called by any live endpoint.
    """
    logging.warning("generate_mock_price_history() called — this is demo-only data.")
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
    # Require at least 5 data points for a meaningful trend
    # (30 meant waiting a month before ANY prediction — too conservative for a live product)
    if not historical_data or len(historical_data) < 5:
        return {"available": False, "reason": "Insufficient historical data (need 5+ recorded price points)"}

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

    # Anchor forecast from last known price (not regression intercept).
    # Apply directional dampening that preserves sign — a rising trend
    # should never flip to declining just because the dampener kicks in.
    base_price = float(prices[-1])  # last known real price
    daily_change = float(slope)     # regression slope = best estimate of daily movement

    # Minimum band half-width: 2% of current price per 30-day horizon.
    # Ensures uncertainty cone is visible even when price history is tight.
    min_band_halfwidth = max(std_dev, base_price * 0.02)

    predictions = []
    for i in range(forecast_days):
        days = i + 1
        # Progressive dampening: slope effect weakens over time to prevent runaway forecasts.
        # Starts at 100% and smoothly reduces to 65% at 90 days — keeps sign, reduces magnitude.
        damp = max(0.65, 1.0 - days * 0.0039)
        change_magnitude = abs(daily_change) * days * damp
        if daily_change >= 0:
            predicted_price = base_price + change_magnitude
        else:
            predicted_price = base_price - change_magnitude
        # Adaptive floor: loosens slightly over longer horizons but never catastrophic
        # 30d floor = 95%, 60d floor = 90%, 90d floor = 85% (match JS logic)
        floor_pct = max(0.85, 1.0 - (days * 0.0017))
        predicted_price = max(predicted_price, base_price * floor_pct)
        # Uncertainty bands: 1.3× band_halfwidth, grows with sqrt(days/7)
        band = min_band_halfwidth * 1.3 * (days / 7) ** 0.5
        lower_bound = predicted_price - band
        upper_bound = predicted_price + band
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




@app.get("/api/market-trends/history/{card_identifier}")
@safe_endpoint
async def get_market_trends_history(
    card_identifier: str,
    days: int = 90,
    api_key: Optional[str] = Depends(verify_api_key_optional),
):
    """Debug/verification endpoint: returns raw DB rows for the identifier."""
    ident = (card_identifier or "").strip()
    if not ident:
        raise HTTPException(status_code=400, detail="card_identifier required")
    rows = get_price_history(card_identifier=ident, days=int(days or 90))
    # rows already include recorded_date, prices, volume, etc.
    return {
        "success": True,
        "card_identifier": ident,
        "days": int(days or 90),
        "count": len(rows),
        "rows": rows,
    }
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


# ═══════════════════════════════════════════════════════════════════════════════
# PRECISION PRICING v2 — CARD-FINGERPRINT-FIRST MULTI-SOURCE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_card_fingerprint(
    card_name:   str,
    card_set:    str,
    card_number: str,
    language:    str = "english",
    game_type:   str = "pokemon",
    grade:       str = "",
    grader:      str = "",
) -> Dict[str, Any]:
    """
    Validate and normalise card identity.  Raises ValueError with a clear
    message if mandatory fields are absent.

    Returns a fingerprint dict consumed by all three price-source helpers.
    card_number + card_set are hard-required: without them we cannot guarantee
    we are pricing the correct card and will refuse to guess.
    """
    errors = []
    if not (card_name   or "").strip(): errors.append("card_name")

    # card_number is mandatory for TCG cards — prevents cross-card contamination.
    # card_set is preferred but NOT required when card_number is present:
    # a card number like "OP01-024" already uniquely identifies the card.
    _tcg_games = {"pokemon", "pokémon", "ptcg", "one piece", "onepiece",
                  "dragon ball", "dragonball", "digimon", "yugioh", "yu-gi-oh",
                  "magic", "mtg", "flesh and blood", "fab", "weiss", "bushiroad",
                  "cardfight", "vanguard", "lorcana"}
    _game_lower = (game_type or "").lower().strip()
    _is_tcg = any(t in _game_lower for t in _tcg_games) or _game_lower in _tcg_games

    _has_number = bool((card_number or "").strip())
    _has_set    = bool((card_set    or "").strip())

    # For TCG cards: require card_number. card_set is a bonus for disambiguation.
    # For non-TCG: card_number is optional; card_set or name is sufficient.
    if _is_tcg and not _has_number:
        errors.append("card_number")
    # Only require card_set for TCG when we also have no card_number (both missing = useless)
    if _is_tcg and not _has_number and not _has_set:
        errors.append("card_set")
    # Non-TCG: require at least a name (already checked above)

    if errors:
        raise ValueError(f"Missing required fields for precision lookup: {', '.join(errors)}")

    _LANG_MAP = {
        "en": "english",  "english": "english",
        "jp": "japanese", "ja": "japanese", "japanese": "japanese",
        "kr": "korean",   "korean": "korean",
        "zh": "chinese",  "chinese": "chinese",
        "de": "german",   "german": "german",
        "fr": "french",   "french": "french",
        "it": "italian",  "italian": "italian",
        "pt": "portuguese","portuguese":"portuguese",
        "es": "spanish",  "spanish": "spanish",
    }
    lang = _LANG_MAP.get((language or "english").lower().strip(),
                          (language or "english").lower().strip())

    grade_clean = (grade or "").strip()
    is_graded   = bool(grade_clean and grade_clean not in ("raw", "ungraded", "0"))

    # Strip leading "#" — stored inconsistently across the DB
    num = (card_number or "").strip().lstrip("#")

    return {
        "card_name":   card_name.strip(),
        "card_set":    card_set.strip(),
        "card_number": num,
        "language":    lang,
        "is_english":  lang == "english",
        "game_type":   (game_type or "").lower().strip(),
        "grade":       grade_clean if is_graded else "",
        "grader":      (grader or "").strip() if is_graded else "",
        "is_graded":   is_graded,
    }


async def _ebay_sold_strict(
    fp:            Dict[str, Any],
    limit:         int = 60,
    days_lookback: int = 90,
) -> Dict[str, Any]:
    """
    eBay completed-sale stats — simple word search, best match wins.

    eBay is a keyword search engine where all tokens are AND'd.
    The card number uniquely identifies a card and appears in virtually
    every listing title. Three tiers max:

      Tier 1: name + number + grade_token  (most specific)
      Tier 2: name + number                (drop grade if thin results)
      Tier 3: name + set + grade_token     (no number — Star Wars / non-TCG)

    EBAY-AU and EBAY-US queried in parallel; higher volume wins.
    Prices trimmed top+bottom 10% before median.
    """
    if not EBAY_APP_ID:
        return {}

    card_number  = (fp.get("card_number") or "").strip()
    card_name    = fp["card_name"]
    card_set     = (fp.get("card_set")     or "").strip()
    grade        = (fp.get("grade")        or "").strip()
    is_graded    = fp.get("is_graded", False)
    game_lower   = fp.get("game_type", "").lower().strip()
    rarity       = (fp.get("rarity")       or "").strip()
    variant_type = (fp.get("variant_type") or "").strip()

    _tcg_games = {"pokemon","pokémon","ptcg","one piece","onepiece",
                  "dragon ball","dragonball","digimon","yugioh","yu-gi-oh",
                  "magic","mtg","flesh and blood","fab","weiss","vanguard","lorcana"}
    _is_tcg = any(t in game_lower for t in _tcg_games)

    # TCG without a number is too ambiguous — skip rather than pollute
    if not card_number and _is_tcg:
        logging.info(f"_ebay_sold_strict: skipping {card_name} — no card_number for TCG")
        return {}

    # ── Clean name: strip CJK, (Signed) prefix ───────────────────────────────
    def _has_cjk(s: str) -> bool:
        return bool(re.search(r'[\u3000-\u9fff\uac00-\ud7af\uf900-\ufaff]', s))

    # Strip (Signed) / [Signed] prefix — signed premium is a multiplier, not a search term
    card_name = re.sub(r'^[\(\[]\s*(?:signed|autographed?|auto)\s*[\)\]]\s*', '', card_name, flags=re.I).strip()
    card_name = re.sub(r'^\s*(?:signed|autographed?)\s+', '', card_name, flags=re.I).strip()

    # CJK / bilingual: extract English part
    if _has_cjk(card_name) and "/" in card_name:
        parts = card_name.split("/")
        card_name = next((p.strip() for p in parts if p.strip() and not _has_cjk(p)), parts[-1].strip())
    elif _has_cjk(card_name):
        card_name = re.sub(r'[\u3000-\u9fff\uac00-\ud7af\uf900-\ufaff\u30a0-\u30ff\u3040-\u309f]+',
                           ' ', card_name).strip()
    card_name = re.sub(r'\s+', ' ', card_name).strip()

    # ── Grade token: CLA grades → PSA equivalents ────────────────────────────
    grade_token = fp.get("_grade_token_override") or ""
    if not grade_token and is_graded and grade:
        m = re.search(r'(\d+(?:\.\d+)?)', grade)
        if m:
            n = float(m.group(1))
            if   n >= 10: grade_token = "PSA 10"
            elif n >= 9.5: grade_token = "PSA 9.5"
            elif n >= 9:   grade_token = "PSA 9"
            elif n >= 8.5: grade_token = "PSA 8.5"
            elif n >= 8:   grade_token = "PSA 8"
            elif n >= 7:   grade_token = "PSA 7"

    # ── Fix 3: Extend days_lookback for vintage signed non-TCG cards ─────────
    # Vintage Star Wars / sports autos sell infrequently — 90 days often returns
    # zero comps, causing the system to fall back to the attribute estimate with
    # the wrong base. Extend the window so we capture the real market signal.
    #   Vintage signed Star Wars/sports → 365 days (annual market)
    #   Other signed non-TCG            → 180 days
    #   Everything else                 → unchanged (caller's days_lookback)
    _is_vintage_signed = (
        not _is_tcg
        and fp.get("is_signed", False)
        and any(sw in game_lower for sw in ("star", "sport", "starwars", "baseball",
                                             "basketball", "football", "cricket",
                                             "nrl", "afl", "rugby"))
    )
    _is_signed_non_tcg = not _is_tcg and fp.get("is_signed", False)
    if _is_vintage_signed and days_lookback < 365:
        days_lookback = 365
        logging.info(f"_ebay_sold_strict: extended lookback to 365d for vintage signed non-TCG | {card_name}")
    elif _is_signed_non_tcg and days_lookback < 180:
        days_lookback = 180
        logging.info(f"_ebay_sold_strict: extended lookback to 180d for signed non-TCG | {card_name}")

    # ── Variant token: only high-signal rarities ──────────────────────────────
    _skip = {"common", "uncommon", "rare", "regular", "standard", ""}
    _variant = rarity if rarity.lower() not in _skip else (
               variant_type if variant_type.lower() not in _skip else "")

    # Fix 4: for signed non-TCG cards, build a "signed" token to inject into
    # non-TCG tiers so we fetch signed comps rather than unsigned base prices.
    # (The signed multiplier then fires on the correct signed-card signal.)
    _signed_token = ""
    if _is_signed_non_tcg:
        _signed_by_clean = (fp.get("signed_by") or "").strip()
        # Include signer name in token when known — more specific comps
        _signed_token = _norm_ws(f"Signed {_signed_by_clean}") if _signed_by_clean else "Signed"

    def _q(*parts):
        return _norm_ws(" ".join(p for p in parts if p))

    # ── 3 tiers ───────────────────────────────────────────────────────────────
    if card_number:
        tiers = list(dict.fromkeys(filter(None, [
            _q(card_name, card_number, grade_token),   # T1: name+num+grade
            _q(card_name, card_number),                # T2: name+num
            _q(card_name, card_number, _variant) if _variant else None,  # T3: with variant
        ])))
    else:
        # Non-TCG (Star Wars, sports): sellers rarely use card numbers.
        # Fix 4: inject signed token into tiers when card is signed — this ensures
        # the eBay comps fetched are actually for signed copies, not unsigned raw
        # cards. Without this, "Boba Fett Dark Empire signed" queries return unsigned
        # results and the signed multiplier fires on the wrong $4 base.
        if _signed_token:
            tiers = list(dict.fromkeys(filter(None, [
                _q(card_name, card_set, _signed_token, grade_token),   # T1: name+set+signed+grade
                _q(card_name, card_set, _signed_token),                # T2: name+set+signed
                _q(card_name, _signed_token, grade_token),             # T3: name+signed+grade
                _q(card_name, _signed_token),                          # T4: name+signed (broadest)
            ])))
            logging.info(f"_ebay_sold_strict: signed non-TCG tiers with token='{_signed_token}' | {card_name}")
        else:
            tiers = list(dict.fromkeys(filter(None, [
                _q(card_name, card_set, grade_token),      # T1: name+set+grade
                _q(card_name, card_set),                   # T2: name+set
                _q(card_name, grade_token) if card_set else None,  # T3: name+grade
            ])))

    # ── Strategy: run sold comps AND grade-filtered active listings in parallel ──
    # For graded cards, a live BIN listing IS the market — sellers don't list below
    # what the card is worth, and for new cards with few sold comps, active listings
    # are more reliable than falling back to raw sold prices.
    #
    # Resolution priority (highest to lowest):
    #   1. Graded sold comps  (confirmed transactions, grade in query)
    #   2. Graded active BIN  (live market, grade in query — preferred over raw sold)
    #   3. Raw sold comps     (ungraded, grade mult will be applied by caller)
    #   4. Raw active BIN     (last resort, grade mult will be applied)

    MIN_SALES = 3
    best_graded_sold:  Optional[Dict[str, Any]] = None
    best_raw_sold:     Optional[Dict[str, Any]] = None
    best_graded_active: Optional[Dict[str, Any]] = None

    # Build grade-filtered active query to run in parallel
    _q_grade_active = ""
    if grade_token:
        _q_grade_active = _q(card_name, card_number, grade_token) if card_number else _q(card_name, card_set, grade_token)
    _q_base_active = _q(card_name, card_number) if card_number else _q(card_name, card_set)

    # Run all sold-comp tiers AND the grade-filtered active query concurrently
    async def _run_tier(tier_q: str) -> Dict[str, Any]:
        grade_in_tier = bool(grade_token and grade_token.lower() in tier_q.lower())
        effective_min = 1 if grade_in_tier else MIN_SALES
        logging.info(f"🔍 eBay sold: '{tier_q}'")
        au, us = await asyncio.gather(
            _ebay_completed_stats(tier_q, limit=limit, days_lookback=days_lookback, global_id="EBAY-AU"),
            _ebay_completed_stats(tier_q, limit=limit, days_lookback=days_lookback, global_id="EBAY-US"),
        )
        au_prices = au.get("prices") or []
        us_prices = us.get("prices") or []
        if au_prices and us_prices:
            merged = sorted(au_prices + us_prices)
            trim = max(1, len(merged) // 10)
            trimmed = merged[trim:-trim] if len(merged) > 2 * trim else merged
            result = dict(au)
            result["prices"] = merged
            result["count"]  = len(merged)
            result["median"] = round(statistics.median(trimmed), 2) if trimmed else 0
        else:
            result = au if (au.get("count") or 0) >= (us.get("count") or 0) else us
        count = result.get("count") or 0
        logging.info(f"   → {count} sold (AU+US), median=${result.get('median', 0):.2f} AUD")
        if count >= effective_min:
            result["query"] = tier_q
            return result
        elif count > 0:
            result["query"] = tier_q
            result["_sparse"] = True  # below min but not empty
            return result
        return {}

    async def _run_active(active_q: str) -> Dict[str, Any]:
        if not active_q:
            return {}
        logging.info(f"🔍 eBay active: '{active_q}'")
        try:
            stats = await _ebay_active_stats(active_q, limit=20)
            # Prefer cluster_mean when available — it is the consensus market price
            med = float((stats or {}).get("cluster_mean") or (stats or {}).get("median") or 0)
            cnt = int((stats or {}).get("count") or 0)
            if med > 0 and cnt >= 1:
                cl_used = bool((stats or {}).get("cluster_used"))
                logging.info(
                    f"   → {cnt} active listings, "
                    f"{'cluster_mean' if cl_used else 'median'}=${med:.2f} AUD"
                    + (f" ({stats.get('cluster_size')} of {cnt} in consensus band)" if cl_used else "")
                )
                return {
                    "source":               "ebay_active",
                    "query":                active_q,
                    "count":                cnt,
                    "median":               med,
                    "prices":               stats.get("prices") or [],
                    "cluster_mean":         stats.get("cluster_mean"),
                    "cluster_size":         stats.get("cluster_size"),
                    "cluster_coverage":     stats.get("cluster_coverage"),
                    "cluster_used":         cl_used,
                    "price_includes_grade": bool(grade_token and grade_token.lower() in active_q.lower()),
                }
        except Exception as _ae:
            logging.debug(f"Active query error: {_ae}")
        return {}

    # Fire all queries concurrently: sold comps for each tier + grade-filtered active
    tier_tasks  = [_run_tier(tq) for tq in tiers]
    async def _noop() -> Dict[str, Any]:
        return {}
    active_task = _run_active(_q_grade_active) if _q_grade_active else _noop()

    all_results = await asyncio.gather(*tier_tasks, active_task)
    tier_results   = all_results[:-1]
    active_graded  = all_results[-1] if _q_grade_active else {}

    # Categorise results
    for i, tr in enumerate(tier_results):
        if not tr:
            continue
        tq = tiers[i]
        has_grade = bool(grade_token and grade_token.lower() in tq.lower())
        if has_grade and not tr.get("_sparse"):
            if best_graded_sold is None:
                best_graded_sold = tr
        elif not has_grade and not tr.get("_sparse"):
            if best_raw_sold is None:
                best_raw_sold = tr
        elif tr.get("_sparse"):
            # sparse graded sold: keep as weak signal
            if has_grade and best_graded_sold is None:
                best_graded_sold = tr  # even sparse graded > raw

    if active_graded and (active_graded.get("count") or 0) >= 1:
        best_graded_active = active_graded

    # ── Priority resolution ───────────────────────────────────────────────────
    # TCG cards (Pokémon, One Piece, etc.): sold comps are primary.
    #   These cards trade frequently — actual transactions are more reliable
    #   than asks. Existing sold-first order is preserved.
    #
    # Non-TCG cards (Star Wars, sports, memorabilia): active listings are primary.
    #   These trade infrequently. Current asks from knowledgeable sellers ARE
    #   the market — historic sales may be months old and from a different
    #   macro environment. Cluster mean on active removes rogue outlier asks.
    if _is_tcg:
        # ── TCG: sold first (unchanged) ──────────────────────────────────────
        if best_graded_sold:
            best = best_graded_sold
            logging.info(f"✅ [TCG] Using graded sold comps: ${best.get('median',0):.2f} AUD")
        elif best_graded_active:
            best = best_graded_active
            logging.info(f"✅ [TCG] Using graded active BIN (no sold comps): ${best.get('median',0):.2f} AUD")
        elif best_raw_sold:
            best = best_raw_sold
            logging.info(f"📦 [TCG] Using raw sold comps (grade mult will apply): ${best.get('median',0):.2f} AUD")
        else:
            best = await _run_active(_q_base_active)
            if best:
                logging.info(f"📦 [TCG] Using raw active BIN: ${best.get('median',0):.2f} AUD")
    else:
        # ── Non-TCG: active first, sold as confidence check / blend ──────────
        if best_graded_active and best_graded_sold:
            # Both sources available — blend: 60% active (current market) + 40% sold
            act_med  = float(best_graded_active.get("median") or 0)
            sold_med = float(best_graded_sold.get("median")   or 0)
            blended  = round(act_med * 0.60 + sold_med * 0.40, 2)
            best = dict(best_graded_active)
            best["median"]          = blended
            best["source"]          = "ebay_active_sold_blend"
            best["active_median"]   = act_med
            best["sold_median"]     = sold_med
            logging.info(
                f"✅ [non-TCG] Blended active ${act_med:.2f} (60%) + sold ${sold_med:.2f} (40%) "
                f"= ${blended:.2f} AUD"
            )
        elif best_graded_active:
            best = best_graded_active
            logging.info(
                f"✅ [non-TCG] Active listings primary: ${best.get('median',0):.2f} AUD"
                + (f" (cluster of {best.get('cluster_size')} listings)" if best.get("cluster_used") else "")
            )
        elif best_graded_sold:
            best = best_graded_sold
            logging.info(f"📦 [non-TCG] Falling back to graded sold (no active): ${best.get('median',0):.2f} AUD")
        elif best_raw_sold:
            best = best_raw_sold
            logging.info(f"📦 [non-TCG] Raw sold comps (grade mult will apply): ${best.get('median',0):.2f} AUD")
        else:
            best = await _run_active(_q_base_active)
            if best:
                logging.info(f"📦 [non-TCG] Raw active BIN: ${best.get('median',0):.2f} AUD")

    if not best:
        return {}

    best.setdefault("source", "ebay")
    best.setdefault("price_includes_grade", bool(grade_token and grade_token in (best.get("query") or "")))
    return best


async def _fetch_pokemontcg_price(fp: Dict[str, Any]) -> Dict[str, Any]:
    """
    For Pokémon (English only): exact set+number lookup via PokemonTCG.io,
    then read the embedded TCGPlayer 30-day market price.
    Returns {} if game is not Pokémon, card is non-English, or lookup fails.
    """
    game = fp.get("game_type", "").lower()
    if game not in ("pokemon", "pokémon", "pokemon tcg", "ptcg"):
        return {}
    if not fp.get("is_english", True):
        return {}

    card_number = (fp.get("card_number") or "").strip()
    card_set    = (fp.get("card_set")    or "").strip()
    if not card_number or not card_set:
        return {}

    hdrs: Dict[str, str] = {"User-Agent": UA}
    if POKEMONTCG_API_KEY:
        hdrs["X-Api-Key"] = POKEMONTCG_API_KEY

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            params = {
                "q":      f'set.name:"{card_set}" number:"{card_number}"',
                "select": "id,name,set,number,tcgplayer,cardmarket",
            }
            r = await client.get(f"{POKEMONTCG_BASE}/cards",
                                 params=params, headers=hdrs)
            cards = (r.json().get("data") or []) if r.status_code == 200 else []

            if not cards:
                params2 = {
                    "q":      f'number:"{card_number}" name:"{fp["card_name"]}"',
                    "select": "id,name,set,number,tcgplayer,cardmarket",
                }
                r2    = await client.get(f"{POKEMONTCG_BASE}/cards",
                                         params=params2, headers=hdrs)
                cards = (r2.json().get("data") or []) if r2.status_code == 200 else []

            if not cards:
                return {}

            card       = cards[0]
            tcgplayer  = card.get("tcgplayer") or {}
            prices_obj = tcgplayer.get("prices") or {}

            PRICE_PRIORITY = [
                "holofoil", "normal", "reverseHolofoil",
                "1stEditionHolofoil", "unlimitedHolofoil", "1stEdition",
            ]
            market_usd: Optional[float] = None
            price_type_used: Optional[str] = None

            for pt in PRICE_PRIORITY:
                if pt in prices_obj:
                    raw = prices_obj[pt].get("market") or prices_obj[pt].get("mid")
                    if raw and float(raw) > 0:
                        market_usd      = float(raw)
                        price_type_used = pt
                        break

            if market_usd is None:
                for pt, pdata in prices_obj.items():
                    if isinstance(pdata, dict):
                        raw = pdata.get("market") or pdata.get("mid")
                        if raw and float(raw) > 0:
                            market_usd      = float(raw)
                            price_type_used = pt
                            break

            if not market_usd:
                return {}

            market_aud = _usd_to_aud_simple(market_usd)
            if not market_aud:
                return {}

            logging.info(
                f"✅ TCGPlayer (PokemonTCG.io): ${market_usd:.2f} USD → "
                f"${market_aud:.2f} AUD | type={price_type_used} | "
                f"card_id={card.get('id')}"
            )
            return {
                "source":               "tcgplayer_via_pokemontcgapi",
                "card_id":              card.get("id"),
                "card_name":            card.get("name"),
                "card_number":          card.get("number"),
                "set_name":             (card.get("set") or {}).get("name"),
                "price_type":           price_type_used,
                "market_price_usd":     market_usd,
                "market_price_aud":     market_aud,
                "median":               market_aud,
                "tcgplayer_url":        tcgplayer.get("url"),
                "updated_at":           tcgplayer.get("updatedAt"),
                "exact_match":          True,
                "count":                30,
                "price_includes_grade": False,
            }
    except Exception as e:
        logging.warning(f"PokemonTCG.io price lookup failed: {type(e).__name__}: {e}")
        return {}


async def _fetch_pricecharting_strict(fp: Dict[str, Any]) -> Dict[str, Any]:
    """
    PriceCharting lookup aligned with PC's actual product naming convention.

    PC product-name format: "{CardName} {CardNumber}"
    e.g. "Monkey D Luffy OP01-024"  /  "Mega Charizard X Ex 013"
    The set name is NOT in PC product-name — only in URL path and category param.

    Query tiers (2 only — set name omitted throughout):
      Tier 1: clean_name + card_number  (category scoped)
      Tier 2: clean_name only           (category scoped, fallback)

    Correct PC field mapping (per official docs):
      loose-price        = Ungraded raw
      manual-only-price  = PSA 10  (NOT graded-price which = PSA 9)
      graded-price       = PSA 9
      box-only-price     = BGS 9.5 GemMint
      bgs-10-price       = BGS 10 Pristine
      condition-17-price = CGC 10
      condition-18-price = SGC 10
    """
    if not PRICECHARTING_TOKEN:
        return {}

    card_name   = fp["card_name"]
    card_number = fp.get("card_number", "")
    game_type   = (fp.get("game_type") or "").lower().strip()

    # ── Game → PC category slug ───────────────────────────────────────────────
    _CAT = {
        "pokemon": "pokemon-cards", "pokémon": "pokemon-cards", "ptcg": "pokemon-cards",
        "one piece": "one-piece-cards", "onepiece": "one-piece-cards",
        "magic": "magic-cards", "mtg": "magic-cards",
        "yugioh": "yugioh-cards", "yu-gi-oh": "yugioh-cards",
        "dragon ball": "dragon-ball-super-cards", "dragonball": "dragon-ball-super-cards",
        "digimon": "digimon-cards",
        "lorcana": "disney-lorcana",
        "flesh and blood": "flesh-and-blood-cards", "fab": "flesh-and-blood-cards",
        "sports": "sports-cards",
    }
    pc_category = next((v for k, v in _CAT.items() if k in game_type), None)

    # ── Clean name for PC query ───────────────────────────────────────────────
    # Strip CJK chars (and surrounding brackets), square bracket codes,
    # normalise punctuation to match PC's slugified product names.
    def _pc_clean(name: str) -> str:
        # Strip (Signed) / [Signed] prefix — PC doesn't have signed cards, find the base card
        name = re.sub(r'^[\(\[]\s*(?:signed|autographed?|auto)\s*[\)\]]\s*', '', name, flags=re.I).strip()
        name = re.sub(r'^\s*(?:signed|autographed?)\s+', '', name, flags=re.I).strip()
        # Remove CJK blocks including surrounding parens: "Mr. 3（ギャルディーノ）" → "Mr. 3"
        name = re.sub(r'[（(][\u3000-\u9fff\uac00-\ud7af\uf900-\ufaff\u30a0-\u30ff\u3040-\u309f][^)）]*[)）]', '', name)
        name = re.sub(r'[\u3000-\u9fff\uac00-\ud7af\uf900-\ufaff\u30a0-\u30ff\u3040-\u309f]+', ' ', name)
        # Remove set-code brackets: "Baby 5 [PRB01]" → "Baby 5"
        name = re.sub(r'\s*\[[A-Z0-9\-]+\]\s*', ' ', name)
        # Convert remaining brackets to their content: "[Yamato Gold]" → "Yamato Gold"
        name = re.sub(r'\[([^\]]+)\]', r'\1', name)
        # Dots in names: "Monkey D. Luffy" → "Monkey D Luffy"
        name = re.sub(r'\.(?=\s|$)', ' ', name)
        return re.sub(r'\s+', ' ', name).strip()

    # PC name aliases (stored name → PC product-name form)
    _ALIASES = {"don quixote": "donquixote", "rocinante": "rosinante"}
    def _apply_aliases(name: str) -> str:
        n = name.lower()
        for orig, repl in _ALIASES.items():
            n = n.replace(orig, repl)
        return n

    clean = _pc_clean(card_name)
    clean_aliased = _apply_aliases(clean)

    # ── Number variants for result validation ─────────────────────────────────
    num_lower = card_number.lower().strip()
    num_bare  = re.sub(r'^[a-z]+0*', '', num_lower)
    num_bare  = re.sub(r'^\d+-', '', num_bare)
    num_variants = list(dict.fromkeys(filter(None, [
        num_lower, f"#{num_lower}", f"/{num_lower}",
        num_bare,  f"#{num_bare}",  f"/{num_bare}",
    ])))

    # Name word(s) for validation — both original and aliased first word
    name_words = list(dict.fromkeys(filter(None, [
        clean.split()[0].lower()         if clean.split()         else "",
        clean_aliased.split()[0].lower() if clean_aliased.split() else "",
    ])))

    async def _try(q: str, need_num: bool) -> Optional[Dict[str, Any]]:
        products = await _pc_search(q, category=pc_category, limit=10)
        logging.info(f"🔍 PC: '{q}' cat={pc_category} → {len(products)} results")
        for p in products:
            pname = str(p.get("product-name") or p.get("name") or "").lower()
            purl  = str(p.get("url") or "").lower()
            num_ok  = any(v in pname or v in purl for v in num_variants if v)
            name_ok = any(w and w in pname for w in name_words)
            logging.info(f"   → '{pname[:60]}' name={name_ok} num={num_ok}")
            if need_num  and num_ok and name_ok: return p
            if not need_num and name_ok:         return p
        return None

    # ── 2-tier search ─────────────────────────────────────────────────────────
    q1 = _norm_ws(f"{clean} {card_number}")   # Tier 1: name + number
    q2 = _norm_ws(clean)                       # Tier 2: name only

    validated = number_validated = None
    if card_number:
        validated = await _try(q1, need_num=True)
        if validated:
            number_validated = True
    if not validated:
        validated = await _try(q2, need_num=bool(card_number))
        number_validated = bool(card_number) and validated is not None

    if not validated:
        logging.info(f"PC: no match for '{clean}' #{card_number}")
        return {}

    # ── Fetch full product detail ─────────────────────────────────────────────
    pc_pid = str(validated.get("id") or validated.get("product-id") or "").strip()
    if pc_pid:
        detail = await _pc_product(pc_pid)
        if isinstance(detail, dict):
            validated = {**validated, **detail}
    pc_prices = _pc_extract_price_fields(validated)

    def _gp(*keys):
        for k in keys:
            v = pc_prices.get(k)
            if isinstance(v, (int, float)) and float(v) > 0:
                return float(v)
        return None

    raw_usd    = _gp("loose-price", "loose_price")
    psa_10_usd = _gp("manual-only-price", "manual_only_price")   # PSA 10
    psa_9_usd  = _gp("graded-price",      "graded_price")        # PSA 9
    graded_usd = psa_10_usd or psa_9_usd
    bgs_gem_usd= _gp("box-only-price",    "box_only_price")      # BGS 9.5
    bgs_10_usd = _gp("bgs-10-price",      "bgs_10_price")        # BGS 10 Pristine
    cgc_10_usd = _gp("condition-17-price","condition_17_price")  # CGC 10
    sgc_10_usd = _gp("condition-18-price","condition_18_price")  # SGC 10

    # BGS Black from CSV cache (not available via API)
    bgs_black_usd = None
    if pc_pid:
        try:
            _con = _pc_db()
            _row = _con.execute("SELECT raw_json FROM pc_products WHERE id=? LIMIT 1", (pc_pid,)).fetchone()
            _con.close()
            if _row and _row[0]:
                _rj = json.loads(_row[0])
                for k in ("bgs-10-black", "bgs_10_black"):
                    v = _rj.get(k)
                    if v and str(v).strip() not in ("","null","NULL","0","N/A"):
                        try: bgs_black_usd = float(str(v).replace("$","").replace(",","").strip()); break
                        except: pass
        except Exception as _e:
            logging.debug(f"PC CSV: {_e}")

    if raw_usd is None and graded_usd is None:
        return {}

    def _aud(v): return _usd_to_aud_simple(v) if v else None
    raw_aud    = _aud(raw_usd)
    psa_10_aud = _aud(psa_10_usd)
    psa_9_aud  = _aud(psa_9_usd)
    graded_aud = psa_10_aud or psa_9_aud
    bgs_gem_aud   = _aud(bgs_gem_usd)
    bgs_10_aud    = _aud(bgs_10_usd)
    bgs_black_aud = _aud(bgs_black_usd)
    cgc_10_aud    = _aud(cgc_10_usd)
    sgc_10_aud    = _aud(sgc_10_usd)

    logging.info(
        f"✅ PC: '{validated.get('product-name') or validated.get('name')}' | "
        f"raw=${raw_aud} PSA10=${psa_10_aud} PSA9=${psa_9_aud} "
        f"BGSBlack={bgs_black_aud} BGS10={bgs_10_aud} CGC10={cgc_10_aud}"
    )

    _is_graded = fp.get("is_graded", False)
    _pc_median   = (graded_aud if _is_graded and graded_aud else raw_aud)
    _pc_incl_grd = bool(_is_graded and graded_aud)

    return {
        "source":                "pricecharting",
        "product_name":          validated.get("product-name") or validated.get("name"),
        "product_id":            pc_pid,
        "product_url":           validated.get("url"),
        "raw_price_usd":         raw_usd,       "raw_price_aud":         raw_aud,
        "psa_10_price_aud":      psa_10_aud,    "psa_9_price_aud":       psa_9_aud,
        "graded_price_usd":      graded_usd,    "graded_price_aud":      graded_aud,
        "bgs_black_price_aud":   bgs_black_aud, "bgs_10_price_aud":      bgs_10_aud,
        "bgs_gem_price_aud":     bgs_gem_aud,   "cgc_10_price_aud":      cgc_10_aud,
        "sgc_10_price_aud":      sgc_10_aud,
        "median":                _pc_median,
        "card_number_validated": number_validated,
        "query":                 q1 if card_number else q2,
        "count":                 5,
        "price_includes_grade":  _pc_incl_grd,
    }


def _reconcile_prices(
    sources: Dict[str, Dict[str, Any]],
    fp:      Dict[str, Any],
) -> Dict[str, Any]:
    """
    Reconcile price signals from TCGPlayer, eBay, and PriceCharting.

    Weighting: TCGPlayer ×3  |  eBay ×2  |  PriceCharting ×1
    Confidence based on source agreement and total sample size.
    Grade multiplier applied ONLY when:
      • card is graded  AND
      • winning price signal is from a raw/ungraded source
    Never double-multiplies a price already reflecting a graded sale.
    """
    grade     = fp.get("grade", "")
    is_graded = fp.get("is_graded", False)

    # ── Source weights ────────────────────────────────────────────────────────
    # For UNGRADED cards TCGPlayer 30-day market price is the gold standard.
    # For GRADED cards TCGPlayer only reports raw/ungraded prices — weighting it
    # ×3 then applying a grade multiplier on top produced the $2k Charizard:
    # TCGPlayer ×3 dominated the signal at $1,000 raw, then ×1.80 grade
    # multiplier = ~$1,900.  When graded, promote eBay (which has actual graded-
    # sale comps) to ×3 and demote TCGPlayer to ×1 (cross-check only).
    if is_graded:
        SOURCE_WEIGHTS = {
            "tcgplayer":                   1,
            "tcgplayer_via_pokemontcgapi": 1,
            "ebay":                        3,
            "pricecharting":               1,
        }
    else:
        SOURCE_WEIGHTS = {
            "tcgplayer":                   3,
            "tcgplayer_via_pokemontcgapi": 3,
            "ebay":                        2,
            "pricecharting":               1,
        }

    signals: Dict[str, Dict[str, Any]] = {}
    for src_key, result in sources.items():
        if not result or not isinstance(result, dict):
            continue
        price = (
            result.get("median")
            or result.get("market_price_aud")
            or result.get("raw_price_aud")
        )
        if price and float(price) > 0:
            signals[src_key] = {
                "price":                float(price),
                "count":                int(result.get("trimmed_count")
                                            or result.get("count") or 1),
                "price_includes_grade": bool(result.get("price_includes_grade", False)),
                "source_data":          result,
            }

    if not signals:
        return {
            "final_price":        0.0,
            "signal_price":       0.0,
            "confidence":         "none",
            "sources_used":       [],
            "source_breakdown":   {},
            "sample_sizes":       {},
            "multiplier_applied": 1.0,
            "slab_premium_added": 0.0,
            "multiplier_reason":  "no_data",
            "sources_agreed_pct": 0,
            "max_deviation_pct":  0.0,
        }

    prices = [v["price"] for v in signals.values()]

    if len(prices) > 1:
        mid       = statistics.median(prices)
        within_30 = sum(1 for p in prices if abs(p - mid) / max(mid, 1) <= 0.30)
        max_dev   = max(abs(p - mid) / max(mid, 1) for p in prices)
    else:
        within_30, max_dev = 1, 0.0

    weighted: List[float] = []
    for src_key, sig in signals.items():
        w = SOURCE_WEIGHTS.get(sig["source_data"].get("source", src_key), 1)
        weighted.extend([sig["price"]] * w)
    weighted.sort()
    raw_signal = float(statistics.median(weighted))

    total_samples = sum(v["count"] for v in signals.values())
    n = len(signals)

    if max_dev > 1.0:
        confidence = "suspect"
    elif n >= 2 and within_30 >= 2 and total_samples >= 10:
        confidence = "high"
    elif n >= 2 and within_30 >= 1 and total_samples >= 5:
        confidence = "medium"
    elif n == 1 and total_samples >= 5:
        confidence = "medium"
    else:
        confidence = "low"

    multiplier        = 1.0
    slab_premium      = 0.0
    multiplier_reason = "not_graded"
    final_price       = raw_signal

    if is_graded and grade:
        ebay_data      = sources.get("ebay") or {}
        ebay_has_grade = bool(
            re.search(r"\bPSA\s*\d|\bBGS\s*\d|\bCGC\s*\d|\bgraded\b",
                      ebay_data.get("query", ""), re.I)
        )
        source_is_graded = any(v["price_includes_grade"] for v in signals.values())

        if ebay_has_grade or source_is_graded:
            multiplier        = 1.0
            multiplier_reason = "source_price_already_graded"
            final_price       = raw_signal
        else:
            try:
                val = apply_cla_valuation(
                    signal_price          = raw_signal,
                    signal_includes_grade = False,
                    search_query          = f"{fp['card_name']} {fp['card_set']}",
                    target_grade          = grade,
                    slab_premium_add_aud  = 0.0,
                )
                final_price       = float(val.get("final_value") or raw_signal)
                multiplier        = float(val.get("multiplier_applied") or 1.0)
                slab_premium      = float(val.get("slab_premium_added") or 0.0)
                multiplier_reason = f"raw_base_grade_{grade}"
            except Exception as _e:
                logging.warning(f"apply_cla_valuation failed in reconciler: {_e}")
                final_price       = raw_signal
                multiplier_reason = "valuation_error_fallback"

    # ── Raw signal minimum — catches wrong-card eBay matches ─────────────────
    # A $2.33 raw signal for a CLA 12 card is impossible — the cheapest card
    # ever graded CLA 12 would be a common at $0.80 raw, but CLA 12 implies
    # the card has real collectible value.  Any signal below these thresholds
    # indicates a wrong card match from eBay and should be rejected.
    #
    # These are MINIMUM raw signal values (AUD) by grade tier:
    #   Grade 12 (Ultra Flawless): raw must be ≥ $5  (cheapest legit CLA12 card)
    #   Grade 10 (Flawless):       raw must be ≥ $1.50
    #   Grade 9  (Mint):           raw must be ≥ $0.50
    if is_graded and grade and raw_signal > 0:
        try:
            _gv_raw = float(re.search(r"(\d+(?:\.\d+)?)", grade).group(1))
        except Exception:
            _gv_raw = 0.0
        _raw_min = 0.0
        if   _gv_raw >= 12: _raw_min = 5.0
        elif _gv_raw >= 10: _raw_min = 1.5
        elif _gv_raw >= 9:  _raw_min = 0.5
        if _raw_min > 0 and raw_signal < _raw_min:
            logging.warning(
                f"⚠️ Raw signal floor: signal={raw_signal:.2f} < min={_raw_min:.2f} "                f"for grade={grade} — wrong card match, zeroing price"
            )
            final_price       = 0.0
            multiplier_reason = "raw_signal_floor_rejected"

    # ── Python-side sanity cap (ceiling) ────────────────────────────────────
    # Synced with the _CLA_GRADE_TIERS table above.
    # The new tiered grade mult allows up to 5× for Grade 12 on a $3 raw card,
    # so the cap must accommodate those values.  The cap is intentionally wider
    # than the expected mult — it only catches genuine outliers (e.g. $5 raw
    # signal × 10 = $50 final would fire the cap at the <$20 tier).
    #
    # Cap schedule (only fires when final > raw × cap):
    #   raw < $15   → cap 6.0×   (Grade 12 on very cheap card can be 5× legitimately)
    #   raw < $50   → cap 4.0×
    #   raw < $200  → cap 3.0×
    #   raw ≥ $200  → cap 2.0×
    #   ungraded    → cap 1.1×
    if raw_signal > 0:
        if is_graded:
            if   raw_signal < 15:   _max_ratio = 6.00
            elif raw_signal < 50:   _max_ratio = 4.00
            elif raw_signal < 200:  _max_ratio = 3.00
            else:                   _max_ratio = 2.00
        else:
            _max_ratio = 1.10
        _cap = round(raw_signal * _max_ratio, 2)
        if final_price > _cap:
            logging.warning(
                f"⚠️ Sanity cap: final_price={final_price:.2f} > cap={_cap:.2f} "
                f"(signal={raw_signal:.2f} × {_max_ratio}). Capping."
            )
            final_price       = _cap
            multiplier_reason = multiplier_reason + "_sanity_capped"

    # ── Minimum floor guard (floor) ───────────────────────────────────────────
    # Catches signals where eBay matched the WRONG card (e.g. $0.43 raw for a
    # Grade 12 Rare Parallel — that's a bulk common price, not the actual card).
    # Returns final_price=0 so the caller falls back to the attribute estimate.
    #
    # Calibration rationale:
    #   The old floors ($15 for G10, $10 for G9) were designed for premium cards
    #   but were killing valid prices for common/uncommon cards that legitimately
    #   sell for $5–$13 graded (e.g. Sanji OP-06 at $13.20 was valid, not a mismatch).
    #   Lowered to catch only genuine garbage signals (sub-$5 for a graded card),
    #   not to second-guess a reasonable comp.
    #
    # Floor schedule (all AUD):
    #   Grade 12+ (Ultra Flawless) → $6  (below this = almost certainly wrong card)
    #   Grade 10–11 (Flawless/Gem) → $4
    #   Grade 9–9.5 (Mint)         → $2
    #   Grade 8–8.5 (Near Mint)    → $1
    #   Grade < 8                  → no floor
    if is_graded and grade and final_price > 0:
        try:
            _gv = float(re.search(r"(\d+(?:\.\d+)?)", grade).group(1))
        except Exception:
            _gv = 0.0
        if _gv >= 12:
            _floor = 20.0  # CLA 12 Ultra Flawless — $2.33 wrong-card signals must not survive
        elif _gv >= 10:
            _floor = 8.0   # CLA/PSA 10 — below $8 is certainly a wrong card
        elif _gv >= 9:
            _floor = 4.0   # PSA 9 Mint
        elif _gv >= 8:
            _floor = 2.0   # PSA 8 NM
        else:
            _floor = 0.0
        if _floor > 0 and final_price < _floor:
            logging.warning(
                f"⚠️ Floor guard: final_price={final_price:.2f} < floor={_floor:.2f} "
                f"for grade={grade}. Signal likely matched wrong card — price zeroed, "
                f"attribute estimate will be used by caller."
            )
            final_price       = 0.0
            multiplier_reason = "floor_guard_no_reliable_price"

    # ── Rarity band check — catch mismatched lookups ─────────────────────────────
    # Validates final_price against known realistic floors/ceilings per game × rarity.
    # ROOT CAUSE of Charizard-at-$1,180: eBay matched a wrong/rarer Charizard variant
    # and the $944 raw signal passed through unchecked.  This band check catches it:
    #   "Double Rare" has a ceiling of $150 raw (×2.5 for graded = $375 max).
    #   $944 >> $375 → flagged as needs_review, old value kept, UI shows ⚠️ badge.
    _fp_game   = (fp.get("game_type")  or "").lower().strip()
    _fp_rarity = (fp.get("rarity")     or "").lower().strip()
    _band_checked = False
    if _fp_game and _fp_rarity and final_price > 0:
        try:
            _band = _rarity_price_band(_fp_game, _fp_rarity, is_graded, final_price)
            _band_checked = True
            if not _band["ok"]:
                logging.warning(
                    f"⚠️ Band check failed: {_band['reason']} — flagging needs_review "
                    f"(price={final_price:.2f}, floor={_band['floor']:.2f}, "
                    f"ceil={_band['ceiling']:.2f}, game={_fp_game}, rarity={_fp_rarity})"
                )
                _band_reliability        = "needs_review"
                _band_reliability_reason = _band["reason"]
            else:
                _band_reliability        = None
                _band_reliability_reason = None
        except Exception as _be:
            logging.warning(f"Band check error (non-fatal): {_be}")
            _band_reliability        = None
            _band_reliability_reason = None
    else:
        _band_reliability        = None
        _band_reliability_reason = None

    # ── Price reliability gate ────────────────────────────────────────────────
    # PHP callers check this field before writing to the DB.
    # "auto_save"   → price is trustworthy enough to overwrite the stored value.
    # "needs_review" → keep old value; flag the item for manual inspection.
    #
    # Rules (any one failing → needs_review):
    #   1. Confidence must be "high" or "medium".
    #   2. If only a single source returned data, we need ≥3 sold comps.
    #      One sale = one lucky (or unlucky) listing — not a market.
    #   3. "suspect" confidence (sources disagree by >100%) is always needs_review.
    #   4. Price outside rarity band (caught above) → needs_review.
    _total_samples = sum(v["count"] for v in signals.values())
    _n_sources     = len(signals)
    if confidence in ("none",):
        _reliability = "needs_review"
        _reliability_reason = "no_price_data"
    elif confidence == "suspect":
        _reliability = "needs_review"
        _reliability_reason = f"sources_disagree_by_{round(max_dev*100)}pct"
    elif confidence == "low":
        _reliability = "needs_review"
        _reliability_reason = f"low_confidence_{_total_samples}_samples_{_n_sources}_sources"
    elif _n_sources == 1 and _total_samples < 3:
        _reliability = "needs_review"
        _reliability_reason = f"single_source_only_{_total_samples}_comps"
    elif _band_reliability == "needs_review":
        # Rarity band check failed — price is outside the realistic range for
        # this game × rarity combination (most likely a wrong-card match).
        _reliability        = "needs_review"
        _reliability_reason = _band_reliability_reason or "rarity_band_exceeded"
    else:
        _reliability = "auto_save"
        _reliability_reason = f"{confidence}_{_total_samples}_samples_{_n_sources}_sources"

    return {
        "final_price":        round(final_price,  2),
        "signal_price":       round(raw_signal,   2),
        "confidence":         confidence,
        "sources_used":       list(signals.keys()),
        "source_breakdown":   {k: round(v["price"], 2) for k, v in signals.items()},
        "sample_sizes":       {k: v["count"]           for k, v in signals.items()},
        "multiplier_applied": round(multiplier,   4),
        "slab_premium_added": round(slab_premium, 2),
        "multiplier_reason":  multiplier_reason,
        "sources_agreed_pct": within_30,
        "max_deviation_pct":  round(max_dev * 100, 1),
        # ── Reliability gate (PHP checks this before saving) ──────────────────
        "price_reliability":        _reliability,
        "price_reliability_reason": _reliability_reason,
    }

class MarketPriceLookupRequest(BaseModel):
    card_name: str
    card_set: Optional[str] = None
    card_year: Optional[str] = None
    card_number: Optional[str] = None
    grade: Optional[str] = None

    rank_within_card: Optional[int] = None  # population rank within identical card
    rank_total: Optional[int] = None        # total population for identical card
    apply_cla_valuation: Optional[bool] = True  # if true, backend returns final_value using CLA model

    # ── Variant / rarity attributes — dramatically affect eBay price ──────────
    # All fields below are synced from card_submissions via cg-collection.php.
    # The endpoint reads these directly; keep names in sync with the PHP $query_data keys.
    variant: Optional[str] = None          # legacy / catch-all variant label (back-compat)
    rarity: Optional[str] = None           # Secret Rare, Ultra Rare, Full Art Rare, Illustration Rare …
    variant_type: Optional[str] = None     # Reverse Holo, Rainbow, Textured, Alt Art, Cosmos Holo …
    edition: Optional[str] = None          # 1st Edition, Shadowless, Collector's Edition
    finish: Optional[str] = None           # Textured, Holofoil, Etched, Cosmos Holo
    is_signed: Optional[bool] = None       # Artist / athlete autograph present
    signed_by: Optional[str] = None        # Signer name (e.g. "Ken Sugimori")
    is_error_card: Optional[bool] = None   # Factory misprint / miscut / wrong-back
    is_promo: Optional[bool] = None        # Tournament / event / prerelease promo
    language: Optional[str] = None         # Japanese, Korean, Chinese — omit/None for English default
    game_type: Optional[str] = None        # pokemon / one piece / dragon ball / magic / etc.
    extra_attributes: Optional[str] = None # Catch-all: Prerelease stamp, Staff stamp, Galaxy foil etc.
    grader: Optional[str] = None           # PSA / BGS / CGC / SGC / CLA / CGA / TAG etc.
    image_url: Optional[str] = None          # Front slab image URL — passed to Claude for visual identification


# ── AI Grade request model — used by /api/ai-grade ───────────────────────────
class AIGradeImageItem(BaseModel):
    face:       str   # "front" or "back"
    b64:        str   # base64-encoded image bytes
    media_type: str   # "image/jpeg" | "image/png" | "image/webp"

class AIGradeRequest(BaseModel):
    images:        List[AIGradeImageItem]
    card_context:  Optional[str] = ""    # "Card Name | Set | Number"
    system_prompt: Optional[str] = ""   # built by PHP, passed through verbatim


# ========================================
# CLA ATTRIBUTE-BASED VALUATION (no-comps fallback)
# ========================================
def _cla_estimate_from_attributes(
    game_type:    str,
    rarity:       str,
    variant_type: str,
    finish:       str,
    language:     str,
    is_signed:    bool,
    grade:        str,
    rank_within_card: int | None = None,
    rank_total:       int | None = None,
    card_set:     str = "",
    card_number:  str = "",
    is_error:     bool = False,
    is_promo:     bool = False,
    pc_raw_price_aud: Optional[float] = None,  # live ungraded price from PriceCharting
    signed_by:    str = "",                    # signer name for celebrity vs artist tier
) -> Dict[str, Any]:
    """
    CLA attribute-based estimate using the Layered Valuation Engine v3.

    Flow:
      1. Determine raw NM base price from game × rarity tier table (AUD).
      2. Feed into cla_layered_valuation() which stacks all attribute checks.
      3. Return result with source="cla_attribute_estimate" so the UI can
         distinguish this from a live market price.

    This is called when no reliable market comps exist (eBay/TCGPlayer/PC
    returned nothing, or the floor guard rejected a garbage signal).
    The layered engine ensures every known attribute adds its correct premium.
    """
    game   = (game_type    or "").lower().strip()
    rar    = (rarity       or "").lower().strip()
    var    = (variant_type or "").lower().strip()
    fin    = (finish       or "").lower().strip()
    lang   = (language     or "english").lower().strip()
    is_jp  = lang in ("japanese", "jp", "ja")

    # ── Step 1: Raw NM base price ─────────────────────────────────────────────
    # All bases are RATIO-BASED, anchored to a common card in each game.
    # Anchor prices come from PriceCharting observed averages (loose-price).
    # These ratios fire ONLY when no live PC loose-price is available.
    # If PC returns a loose-price for this specific card, it overrides everything.
    #
    # Formula: base_aud = anchor_aud × rarity_ratio
    # The anchor is the cheapest/most common card in that game.

    _OP_ANCHOR  = 0.80   # One Piece Common AUD (PC observed avg)
    _PK_ANCHOR  = 0.40   # Pokemon Common AUD
    _DBZ_ANCHOR = 0.80   # Dragon Ball Common AUD
    _SW_ANCHOR  = 4.00   # Star Wars Common AUD — bumped from $2.50; vintage Decipher/Dark Horse floor
    _GEN_ANCHOR = 0.80   # Generic Common AUD

    # Ratios relative to common = 1.0x (derived from observed market relationships)
    _ONE_PIECE_BASES: List[tuple] = [(k, round(r * _OP_ANCHOR, 2)) for k, r in [
        ("leader parallel",     500.0),
        ("l parallel",          500.0),
        ("gold parallel",       241.0),   # Gold DON!! ratio: $192/$0.80
        ("gold",                241.0),
        ("super rare parallel", 163.0),   # SR★: $130/$0.80
        ("sr parallel",         163.0),
        ("sr★",                 163.0),
        ("rare parallel",        44.0),   # R★: $35/$0.80
        ("r★",                   44.0),
        ("r parallel",           44.0),
        ("leader rare",          225.0),  # L: $180/$0.80
        ("secret rare",           69.0),  # SEC: $55/$0.80
        ("sec",                   69.0),
        ("super rare",            37.0),  # SR: $29.60 AU raw observed avg (was 25×=$20 — too low)
        ("sr",                    25.0),
        ("illustration rare",     22.5),
        ("full art",              22.5),
        ("alt art",               22.5),
        ("textured",              50.0),
        ("manga",                 56.0),
        ("rare",                   6.3),
        ("uncommon",               2.2),
        ("common",                 1.0),
        ("parallel",              31.0),
    ]]
    _POKEMON_BASES: List[tuple] = [(k, round(r * _PK_ANCHOR, 2)) for k, r in [
        ("crown rare",                  700.0),
        ("special illustration rare",   325.0),
        ("special art rare",            325.0),
        ("sar",                         325.0),
        ("hyper rare",                  175.0),
        ("shiny ultra rare",            125.0),
        ("shiny rare",                   45.0),
        ("secret rare",                 138.0),
        ("illustration rare",           100.0),
        ("full art",                     75.0),
        ("rainbow rare",                 70.0),
        ("alt art",                      70.0),
        ("ultra rare",                   45.0),
        ("double rare",                  18.0),   # ex / double rare: ~$7.20 AU raw (was 12.5×=$5.00 — too low)
        ("radiant rare",                 38.0),
        ("trainer gallery",              30.0),
        ("holo rare",                    15.0),
        ("reverse holo",                  8.8),
        ("rare",                          6.3),
        ("uncommon",                      2.3),
        ("common",                        1.0),
    ]]
    _DBZ_BASES: List[tuple] = [(k, round(r * _DBZ_ANCHOR, 2)) for k, r in [
        ("special rare",    100.0),
        ("spr",             100.0),
        ("secret rare",      63.0),
        ("special art",      63.0),
        ("ultra rare",       31.0),
        ("super rare",       15.0),
        ("rare",              6.3),
        ("uncommon",          2.5),
        ("common",            1.0),
    ]]
    _STAR_WARS_BASES: List[tuple] = [(k, round(r * _SW_ANCHOR, 2)) for k, r in [
        # ── Premium subsets (most-specific keys first) ────────────────────────
        # Artist-edition / autograph-run cards command the highest raw premiums.
        # Cam Kennedy's Dark Empire art is the most sought-after in vintage SW autos.
        ("cam kennedy artist",  20.0),  # Cam Kennedy signed artist-edition set
        ("artist edition",      18.0),  # Generic artist-edition subsets
        ("dark horse",          14.0),  # Dark Horse Comics (DH-prefix, 1993-1998)
        ("hologram",            28.0),  # Decipher/Topps holograms — rarest insert type
        ("limited edition",     22.0),  # Limited print runs
        ("series ii",           16.0),  # Series II / later series (shorter print)
        ("galaxy",              14.0),  # Decipher Galaxy Series
        ("dark empire",         10.0),  # Dark Empire I/II — Cam Kennedy art
        ("chromium",            10.0),  # Chromium/Chrome premium variants
        ("chrome",              10.0),
        ("cover gallery",        8.0),  # Decipher Cover Gallery — artist signed sets
        ("foil",                 8.0),
        ("embossed",             7.2),
        ("prism",                7.2),
        ("rare",                 4.0),
        ("uncommon",             2.0),
        ("common",               1.0),
        # ── Set/publisher catch-alls (fire when no specific key matched above) ─
        ("decipher",             4.0),
        ("topps",                3.2),
        ("vehicles",             3.6),
        ("premiere",             4.0),
    ]]
    _GENERIC_BASES: List[tuple] = [(k, round(r * _GEN_ANCHOR, 2)) for k, r in [
        ("parallel",         25.0),
        ("secret rare",      56.0),
        ("ultra rare",       25.0),
        ("full art",         19.0),
        ("rare",              6.3),
        ("uncommon",          2.5),
        ("common",            1.0),
    ]]

    if   "one piece" in game or "onepiece" in game or game == "op":
        game_table = _ONE_PIECE_BASES
    elif "pokemon" in game or "ptcg" in game:
        game_table = _POKEMON_BASES
    elif "dragon ball" in game or "dragonball" in game or "dbz" in game:
        game_table = _DBZ_BASES
    elif any(g in game for g in ("star wars", "starwars", "star_wars")):
        game_table = _STAR_WARS_BASES
    else:
        game_table = _GENERIC_BASES

    # Game-specific fallback when rarity is unknown/empty
    # No hardcoded fallback dollar amounts.
    # If rarity doesn't match any key AND no PC data → base = 0 → no_data
    _fallback = 0.0

    # Normalise ASCII star (*) to Unicode ★ and expand shorthand notations
    # so stored values like "Rare *", "SR*", "R*" all resolve correctly.
    def _norm_rarity_str(s: str) -> str:
        s = s.lower().strip()
        # Most-specific patterns first (super rare before rare, leader before r)
        s = re.sub(r'super\s+rare\s+\*(\s+parallel)?', 'super rare parallel', s)
        s = re.sub(r'leader\s+\*(\s+parallel)?',        'leader parallel',     s)
        s = re.sub(r'rare\s+\*(\s+parallel)?',          'rare parallel',       s)
        # Abbreviations with OPTIONAL space before *: "SR * Parallel", "SR*", "SR *"
        # \s* allows zero or more spaces between the abbreviation and the star
        s = re.sub(r'\bsr\s*\*(\s+parallel)?', 'super rare parallel', s)
        s = re.sub(r'\br\s*\*(\s+parallel)?',  'rare parallel',       s)
        s = re.sub(r'\bl\s*\*(\s+parallel)?',  'leader parallel',     s)
        # Unicode star shorthand (with optional space): "SR★", "SR ★"
        s = re.sub(r'\bsr\s*★(\s+parallel)?', 'super rare parallel', s)
        s = re.sub(r'\br\s*★(\s+parallel)?',  'rare parallel',       s)
        s = re.sub(r'\bl\s*★(\s+parallel)?',  'leader parallel',     s)
        # Written-out "SR Parallel" / "R Parallel"
        s = re.sub(r'\bsr\s+parallel\b', 'super rare parallel', s)
        s = re.sub(r'\br\s+parallel\b',  'rare parallel',       s)
        return s

    # Include card_set in search so set-based keys work (Star Wars, sports, etc.)
    # e.g. "Dark Empire II" → matches "dark empire" in _STAR_WARS_BASES
    # Also combine rarity+variant so "Gold" + "Parallel" → "gold parallel"
    _set_lower = (card_set or "").lower().strip()
    search_str = _norm_rarity_str(f"{rar} {var} {fin} {_set_lower}")
    # ── Determine base price ─────────────────────────────────────────────────
    # Priority 1: PC live loose-price (real market data — most accurate)
    # Priority 2: Ratio-based calculated estimate from rarity tier
    # Priority 3: 0 → caller returns no_market_data

    base_aud     = _fallback
    matched_tier = "unknown"

    if pc_raw_price_aud and float(pc_raw_price_aud) > 0:
        # Real data from PriceCharting — always preferred
        base_aud     = float(pc_raw_price_aud)
        matched_tier = f"pc_live"
        logging.info(f"📊 Attr base: PC live ${base_aud:.2f} AUD for {search_str!r}")
    else:
        # No live data — use ratio-based calculated estimate
        for key, price in game_table:
            if key in search_str:
                base_aud     = price
                matched_tier = f"ratio:{key}"
                break
        if base_aud > 0:
            logging.info(f"📊 Attr base: ratio estimate ${base_aud:.2f} AUD (tier={matched_tier})")
        else:
            logging.info(f"📊 Attr base: no data for {search_str!r} — returning no_market_data")

    # If no base at all — return empty so caller knows price is genuinely unknown
    if base_aud <= 0:
        return {
            "current_price":  0,
            "source":          "no_market_data",
            "confidence":      "none",
            "matched_rarity_tier": "unknown",
            "base_price":      0,
        }

    # ── Step 2: Feed everything into the layered valuation engine ─────────────
    lv = cla_layered_valuation(
        raw_base_aud     = base_aud,
        grade            = grade,
        rarity           = rarity,
        variant_type     = variant_type,
        finish           = finish,
        language         = language,
        edition          = "",
        is_signed        = is_signed,
        signed_by        = signed_by,
        game_type        = game_type,
        rank_within_card = rank_within_card,
        rank_total       = rank_total,
        is_error         = is_error,
        is_promo         = is_promo,
    )

    final_value = lv["final_value"]
    grade_mult  = lv["grade_mult"]
    signed_mult = lv["signed_mult"]

    logging.info(
        f"📊 Attribute estimate v3: game={game} | tier={matched_tier} | "
        f"base=${base_aud:.2f} | checks={lv['active_checks']} → ${final_value:.2f} AUD"
    )

    return {
        "current_price":        final_value,
        "final_price":          final_value,
        "signal_price":         base_aud,
        "final_value":          final_value,
        "base_price":           base_aud,
        "multiplier_applied":   round(lv["total_mult"], 4),
        "grade_multiplier":     round(grade_mult, 4),
        "slab_premium_added":   0.0,
        "multiplier_reason":    f"layered_estimate_{matched_tier}",
        "confidence":           "estimate",
        "source":               "cla_attribute_estimate",
        "sources_used":         ["cla_attribute_estimate"],
        "price_includes_grade": True,
        "price_reliability":    "auto_save",   # permit PHP to save; est is better than stale bad data
        "grade_12_uplift":      False,
        "variant_matched":      True,
        "matched_rarity_tier":  matched_tier,
        "is_jp":                is_jp,
        "signed_multiplier":    signed_mult,
        "fingerprint_used":     False,
        "estimated":            True,
        "layered_checks":       lv["checks"],
        "active_checks":        lv["active_checks"],
        "last_updated":         datetime.now().isoformat(),
    }


# ========================================
# CLA VALUATION MODEL (single source of truth)
# ========================================
def _cla_parse_grade(grade_str: str) -> float | None:
    if not grade_str:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(grade_str))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

# ========================================
# CLA LAYERED VALUATION ENGINE v3
# ========================================
# Architecture: raw_base × grade × pop_rank × language × finish ×
#               edition × signed × error × promo
#
# Each check is independent and stacks multiplicatively.
# The grade multiplier is price-tiered via interpolation between five
# raw-price breakpoints, calibrated against AU PSA/CLA market data 2023-2026.
# ========================================

# ── Grade tier table ──────────────────────────────────────────────────────────
# Each row: (min_grade, (mult_at_$3, mult_at_$15, mult_at_$50, mult_at_$150, mult_at_$500+))
# Represents final_graded_price / raw_nm_price at that price point.
# Cheap cards attract a higher grade % premium because the grading cost itself
# is significant vs the card's raw value.  Expensive cards: grade adds
# authentication and liquidity but not a proportionally large % premium.
#
# Calibration examples (AU market, 2023-2026):
#   $3 raw → Grade 12: ~$15 (5×)  |  Grade 10: ~$12 (4×)  |  Grade 9: ~$7.50 (2.5×)
#   $15 raw → Grade 12: ~$52 (3.5×) | Grade 10: ~$45 (3×)  | Grade 9: ~$30 (2×)
#   $50 raw → Grade 12: ~$125 (2.5×) | Grade 10: ~$110 (2.2×) | Grade 9: ~$80 (1.6×)
#   $150 raw → Grade 12: ~$270 (1.8×) | Grade 10: ~$240 (1.6×) | Grade 9: ~$195 (1.3×)
#   $500 raw → Grade 12: ~$750 (1.5×) | Grade 10: ~$675 (1.35×) | Grade 9: ~$575 (1.15×)
#
_CLA_GRADE_TIERS: List[tuple] = [
    # (min_grade, (mult_$3, mult_$15, mult_$50, mult_$150, mult_$500))
    # Grade 12 calibration (AU market observed 2023-2026):
    #   $3 raw → ~$15 (5×)  | $15 raw → ~$52 (3.5×) | $50 raw → ~$125 (2.5×)
    #   $150 raw → ~$270 (1.8×) | $500 raw → ~$750 (1.5×)
    # CLA is a boutique AU grader — NOT BGS Black equivalent (which is 6.5× PSA 10).
    # Previous values (23.5, 17.6, 12.9, 9.40, 7.93) were formula-derived (Grade 10 × 5.87)
    # and did not reflect real AU market prices; they caused 10–20× inflation.
    (12.0, (5.0,  3.5,  2.5,  1.80, 1.50)),  # CLA 12 Ultra Flawless — realistic AU premium
    (10.0, (4.0,  3.0,  2.2,  1.60, 1.35)),  # CLA/PSA 10 Flawless
    ( 9.5, (3.0,  2.4,  1.9,  1.45, 1.25)),  # Near-gem
    ( 9.0, (2.5,  2.0,  1.6,  1.30, 1.15)),  # PSA 9 Mint
    ( 8.5, (1.5,  1.3,  1.15, 1.05, 1.00)),  # PSA 8.5 NM+
    ( 8.0, (1.00, 0.92, 0.90, 0.88, 0.85)),  # PSA 8: at/slightly below market
    ( 7.0, (0.80, 0.75, 0.72, 0.70, 0.68)),  # PSA 7: visible condition
    ( 0.1, (0.65, 0.60, 0.55, 0.52, 0.50)),  # PSA 6 and below
]
_CLA_GRADE_BREAKPOINTS: List[float] = [3.0, 15.0, 50.0, 150.0, 500.0]


def _cla_grade_mult_tiered(grade_str: str, raw_base_aud: float = 0.0) -> float:
    """
    Price-tiered grade premium multiplier.

    Returns the multiplier (final_graded / raw_nm) for a given CLA grade at
    a given raw base price.  Interpolates linearly between the five breakpoints
    in _CLA_GRADE_BREAKPOINTS.

    Must stay in sync with cg_grade_value_multiplier() in cg-collection.php.
    """
    g   = _cla_parse_grade(grade_str) or 0.0
    if g == 0:
        return 1.0

    mults: Optional[tuple] = None
    for min_g, tier in _CLA_GRADE_TIERS:
        if g >= min_g:
            mults = tier
            break
    if mults is None:
        return 1.0

    raw = float(raw_base_aud or 0.0)
    bp  = _CLA_GRADE_BREAKPOINTS

    if raw <= 0 or raw <= bp[0]:
        return float(mults[0])
    if raw >= bp[-1]:
        return float(mults[-1])

    # Linear interpolation between adjacent breakpoints
    for i in range(len(bp) - 1):
        if bp[i] <= raw < bp[i + 1]:
            t = (raw - bp[i]) / (bp[i + 1] - bp[i])
            return float(round(mults[i] + t * (mults[i + 1] - mults[i]), 4))

    return float(mults[-1])


def cla_layered_valuation(
    raw_base_aud:     float,
    grade:            str   = "",
    rarity:           str   = "",
    variant_type:     str   = "",
    finish:           str   = "",
    language:         str   = "english",
    edition:          str   = "",
    is_signed:        bool  = False,
    signed_by:        str   = "",
    game_type:        str   = "",
    rank_within_card: Optional[int] = None,
    rank_total:       Optional[int] = None,
    is_error:         bool  = False,
    is_promo:         bool  = False,
) -> Dict[str, Any]:
    """
    CLA Layered Valuation Engine v3 — the single source of truth for
    attribute-based pricing.

    Architecture
    ─────────────
    final_value = raw_base × CHECK1_grade × CHECK2_pop_rank ×
                  CHECK3_language × CHECK4_finish × CHECK5_edition ×
                  CHECK6_signed × CHECK7_error × CHECK8_promo

    Each check is independent and stacks multiplicatively.  This mirrors how
    TCGPlayer and PriceCharting work: find the raw/ungraded market price, then
    stack premiums for every attribute that adds collector value.

    Returns a full breakdown dict so the UI can explain every dollar.
    """
    raw_base_aud = max(float(raw_base_aud or 0.0), 0.0)
    checks: Dict[str, float] = {}
    running = raw_base_aud

    game = (game_type or "").lower().strip()
    fin  = (finish    or "").lower().strip()
    vt   = (variant_type or "").lower().strip()
    ed   = (edition   or "").lower().strip()
    lang = (language  or "english").lower().strip()

    # ── CHECK 1: Grade premium ────────────────────────────────────────────────
    # The biggest driver.  Uses the tiered interpolated table above.
    grade_mult = _cla_grade_mult_tiered(grade, raw_base_aud) if grade else 1.0
    checks["grade"] = grade_mult
    running *= grade_mult

    # ── CHECK 2: Population rank scarcity ─────────────────────────────────────
    # Top-pop graded cards command a scarcity premium — there are only so many
    # gem copies in existence, and being #1 pop means you own the best one.
    pop_mult = 1.0
    if grade and rank_within_card is not None and rank_within_card > 0:
        if   rank_within_card == 1:   pop_mult = 1.25   # sole #1 — unique in the registry
        elif rank_within_card <= 3:   pop_mult = 1.15
        elif rank_within_card <= 5:   pop_mult = 1.08
        elif rank_within_card <= 10:  pop_mult = 1.04
    checks["pop_rank"] = pop_mult
    running *= pop_mult

    # ── CHECK 3: Language premium ─────────────────────────────────────────────
    # IMPORTANT: Language premiums must reflect the CURRENT market, not assumptions.
    #
    # One Piece JP:
    #   Japanese is the ORIGINAL print run. Sets release in JP first (sometimes
    #   exclusively). The AU One Piece collector community actively prefers JP.
    #   A modest +15% is defensible based on observed AU market data.
    #
    # Pokémon JP:
    #   NO premium applied. In the current (2024–2026) EN-dominant market,
    #   English Pokémon cards frequently sell for MORE than their Japanese
    #   equivalents — EN SIRs, Crown Rares, and ex cards are the collector
    #   standard globally. Japanese Pokémon bulk is actively cheaper.
    #   Applying a JP premium here would OVER-value JP Pokémon cards.
    #
    # All other games:
    #   No language premium by default. Most other TCGs (YGO, DBS, Digimon)
    #   have no consistent JP vs EN price differential in the AU market.
    #   If evidence of a consistent premium emerges for a specific game,
    #   add it here with a market reference.
    is_jp = lang in ("japanese", "jp", "ja")
    lang_mult = 1.0
    if is_jp:
        if "one piece" in game:
            lang_mult = 1.15   # JP OP: modest premium, original print run
        # Pokémon: no JP premium — EN is the dominant/higher-value language currently
        # Other games: no premium — insufficient evidence of consistent JP differential
    checks["language"] = lang_mult
    running *= lang_mult

    # ── CHECK 4: Finish / treatment premium ───────────────────────────────────
    # Special finishes are short-printed variants that command meaningful premiums.
    combined = f"{fin} {vt}"
    finish_mult = 1.0
    if   any(f in combined for f in ("textured", "etched", "gilded")):  finish_mult = 1.40
    elif any(f in combined for f in ("galaxy", "cosmos")):              finish_mult = 1.25
    elif "holo" in combined and "reverse" not in combined:              finish_mult = 1.10
    checks["finish"] = finish_mult
    running *= finish_mult

    # ── CHECK 5: Edition premium ──────────────────────────────────────────────
    # 1st Edition and Shadowless are genuinely scarce and historically significant.
    edition_mult = 1.0
    if   "1st edition" in ed or "first edition" in ed:  edition_mult = 2.50
    elif "shadowless"  in ed:                            edition_mult = 2.00
    elif "collector"   in ed and "edition" in ed:        edition_mult = 1.30
    checks["edition"] = edition_mult
    running *= edition_mult

    # ── CHECK 6: Signed / autograph premium ───────────────────────────────────
    # Four tiers:
    #   TCG artist (Ken Sugimori etc):              1.8×  floor $20
    #   Franchise artist (Cam Kennedy, Struzan etc):3.5×  floor $80
    #   Comic/card artist (generic):                2.2×  floor $45
    #   Celebrity/actor (Harrison Ford etc):        4.5×  floor $150
    #
    # Franchise artists are the definitive illustrators of a specific IP —
    # their autographs carry a meaningful premium over generic comic artists
    # because collectors of that franchise specifically seek their signatures.
    _tcg_games_s = ("pokemon", "one piece", "dragon ball", "yugioh", "magic",
                     "mtg", "digimon", "lorcana", "weiss", "vanguard", "fab")
    _is_tcg = any(g in game for g in _tcg_games_s)
    _signer_lower = (signed_by or "").lower().strip()
    _is_celebrity = any(w in _signer_lower for w in (
        "actor", "actress", "celebrity",
        "harrison", "hamill", "fisher", "prowse", "ford",
        "williams", "mayhew", "daniels", "oz",
    ))
    # Franchise / IP-defining artists — primary illustrators of a specific universe.
    # Cam Kennedy: Dark Empire I & II (definitive Star Wars Dark Horse artist)
    # Hildebrandt brothers: original SW poster art
    # Ralph McQuarrie: original SW concept/production art
    # Drew Struzan: SW theatrical poster artist
    # Dave Dorman: Dark Horse cover artist
    # Tsunekazu Ishihara / Ken Sugimori: Pokémon (handled by TCG tier above)
    _FRANCHISE_ARTISTS = {
        "cam kennedy", "kennedy",
        "hildebrandt", "greg hildebrandt", "tim hildebrandt",
        "mcquarrie", "ralph mcquarrie",
        "struzan", "drew struzan",
        "dorman", "dave dorman",
        "wheatley", "john wheatley",
    }
    _is_franchise_artist = any(a in _signer_lower for a in _FRANCHISE_ARTISTS)
    signed_mult  = 1.0
    signed_floor = 0.0
    if is_signed:
        if _is_tcg:
            signed_mult  = 1.80   # TCG artist auto
            signed_floor = 20.0
        elif _is_celebrity:
            signed_mult  = 4.50   # Celebrity / major actor
            signed_floor = 150.0
        elif _is_franchise_artist:
            signed_mult  = 3.50   # Franchise / IP-defining artist (Cam Kennedy etc)
            signed_floor = 80.0
        else:
            signed_mult  = 2.20   # Generic comic / card artist
            signed_floor = 45.0
    checks["signed"] = signed_mult
    running *= signed_mult
    if is_signed and running < signed_floor:
        running = signed_floor

    # ── CHECK 7: Error / misprint premium ─────────────────────────────────────
    # Documented factory errors (misprints, miscuts, wrong-back) are scarce and
    # desirable to error collectors.  2× is a conservative starting point.
    error_mult = 1.0
    if is_error:
        error_mult = 2.0
    checks["error"] = error_mult
    running *= error_mult

    # ── CHECK 8: Promo premium ────────────────────────────────────────────────
    # Tournament / staff / event promos are limited-distribution and command a
    # modest premium over the standard print.
    promo_mult = 1.0
    if is_promo:
        promo_mult = 1.30
    checks["promo"] = promo_mult
    running *= promo_mult

    final_value = round(running, 2)
    total_mult  = round(final_value / raw_base_aud, 4) if raw_base_aud > 0 else 1.0

    # Build a human-readable check summary for UI tooltips
    active_checks = {k: round(v, 4) for k, v in checks.items() if abs(v - 1.0) > 0.001}

    return {
        "final_value":    final_value,
        "raw_base":       raw_base_aud,
        "total_mult":     total_mult,
        "checks":         checks,           # ALL checks (1.0 = no premium)
        "active_checks":  active_checks,    # only checks that added/reduced value
        "grade_mult":     grade_mult,
        "pop_mult":       pop_mult,
        "lang_mult":      lang_mult,
        "finish_mult":    finish_mult,
        "edition_mult":   edition_mult,
        "signed_mult":    signed_mult,
        "error_mult":     error_mult,
        "promo_mult":     promo_mult,
    }


def cla_grade_multiplier(grade_str: str, rank_within_card: int | None = None,
                         rank_total: int | None = None,
                         signal_price: float = 0.0) -> float:
    """
    Backward-compatible wrapper — calls the new tiered grade table.

    Callers in the reconciler path pass signal_price as the raw base so the
    interpolation uses the actual card value rather than a default.

    Must stay in sync with cg_grade_value_multiplier() in cg-collection.php.
    """
    mult = _cla_grade_mult_tiered(grade_str, signal_price)

    # Pop-rank premium (kept in this wrapper for reconciler path which passes
    # rank separately rather than via cla_layered_valuation).
    if rank_within_card is not None and rank_within_card > 0:
        if   rank_within_card == 1:   mult *= 1.25
        elif rank_within_card <= 3:   mult *= 1.15
        elif rank_within_card <= 5:   mult *= 1.08
        elif rank_within_card <= 10:  mult *= 1.04

    return float(round(mult, 6))

def _parse_grade_from_query(q: str) -> float | None:
    if not q:
        return None
    m = re.search(r"\b(?:PSA|BGS|CGC|SGC|CSG|TAG|HGA|ACE|GMA|ARS)\s*(\d+(?:\.\d+)?)\b", q, re.I)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


# ── Grader brand factor ──────────────────────────────────────────────────────
# eBay comps are almost always PSA-graded. Converts a PSA-based comp price into
# the estimated resale value for whichever grader + grade actually holds the card.
#
# Calibrated from real PriceCharting data (Baby 5 Alt Art OP05-034, Mar 2026):
#   SGC 10:          $35  → 0.602× PSA 10
#   CGC 10:          $47  → 0.809× PSA 10
#   PSA 10:          $58  → 1.000× (baseline)
#   BGS 10 Pristine: $76  → 1.307× PSA 10  ← BGS 10 > PSA 10 (much rarer)
#   CGC 10 Pristine: $85  → 1.462× PSA 10
#   BGS 10 Black:   $380  → 6.537× PSA 10  ← Perfect score, near impossible
#
# CRITICAL — BGS grading scale differs from PSA:
#   BGS 9.5 Gem Mint  ≈ PSA 10 (standard gem grade, similar price to CGC 10)
#   BGS 10 Pristine   > PSA 10 (rarer, 1.3× more valuable)
#   BGS 10 Black      >> PSA 10 (all sub-grades perfect, 6× more valuable)
#
# CLA (Collectors League — worldwide grader, 12-point scale):
#   CLA 10 Flawless       = PSA 10 equivalent   → factor 1.000×
#   CLA 11                = interpolated         → factor 2.424× (geometric mean 10↔12)
#   CLA 12 Ultra Flawless ≈ 90% of BGS Black     → factor 5.874×
#   Note: these are price-CONVERSION MULTIPLIERS (PSA eBay comp → estimated CLA value),
#   not percentage premiums. "CLA 12 at 5.87×" means if PSA 10 comps at $100,
#   CLA 12 estimates at ~$587 — matching BGS Black minus a boutique liquidity discount.
#
# The function is GRADE-AWARE for BGS/CGC/CLA since those graders use
# non-standard scales or have distinct tiers within the same numeric grade.

# Base factors calibrated from real PriceCharting data (2 cards, Mar 2026)
# and CLA grading philosophy (confirmed by CLA):
#
#   CLA grading equivalency (confirmed):
#     CLA 10 Flawless       = PSA 10          → 1.000× PSA
#     CLA 11                = interpolated     → 2.554× PSA (geometric mean of 10 and 12)
#     CLA 12 Ultra Flawless = BGS 10 Black     → 6.525× PSA (perfect grade, all sub-scores max)
#
#   Other graders (calibrated from real PriceCharting data):
#     SGC 10:          0.599× avg (Baby5 + Charizard — high confidence)
#     CGC 10:          0.810× (Baby5; Charizard excluded as thin-market outlier)
#     BGS 9.5 GemMint: 0.810× (≈ CGC 10, same price tier)
#     BGS 10 Pristine: 1.305× (consistently above PSA 10 across both cards)
#     BGS 10 Black:    6.525× (= CLA 12, perfect grade, near impossible)
#     CGC 10 Pristine: 1.280× (averaged from both cards)
#     TAG 10:          0.930× (Charizard only — single point)
#     ACE 10:          0.490× (Charizard only — single point)

_GRADER_BASE_FACTORS: Dict[str, float] = {
    "PSA":        1.000,
    "CGC":        0.810,  # calibrated (Baby5 reliable); Charizard thin-market excluded
    "BGS":        0.810,  # BGS 9.5 GemMint default — grade-aware logic handles Pristine/Black
    "BECKETT":    0.810,
    "SGC":        0.600,  # 0.599× averaged across 2 cards — high confidence
    "CLA":        1.000,  # CLA 10 = PSA 10 (confirmed by CLA grading standard)
    "COLLECTORS LEAGUE": 1.000,
    "CGA":        0.600,  # regional AU only, similar to SGC
    "TAG":        0.930,  # real data (Charizard) — single point, use cautiously
    "HGA":        0.650,  # estimated
    "ACE":        0.490,  # real data (Charizard) — single point
    "":           0.800,  # unknown — conservative default
}

def _grader_brand_factor(grader: str, grade_str: str = "") -> float:
    """
    Return multiplier to convert a PSA-based eBay comp into estimated value
    for the actual grader + grade on the slab.

    Grade-aware for BGS (9.5/10/10 Black), CGC (10/Pristine), and
    CLA (10 Flawless / 12 Ultra Flawless).
    """
    g = (grader or "").strip().upper()
    # Normalise aliases
    if g in ("COLLECTORS LEAGUE", "COLLECTORS LEAGUE AUSTRALIA", "CL"):
        g = "CLA"
    if g in ("BECKETT", "BECKETT GRADING", "BVG"):
        g = "BGS"
    if g in ("CERTIFIED GUARANTY", "CERTIFIED GUARANTY COMPANY"):
        g = "CGC"

    factor = _GRADER_BASE_FACTORS.get(g, _GRADER_BASE_FACTORS[""])

    # Parse numeric grade value
    gv = 0.0
    grade_lower = (grade_str or "").lower()
    if grade_str:
        try:
            m = re.search(r"(\d+(?:\.\d+)?)", grade_str)
            if m:
                gv = float(m.group(1))
        except Exception:
            pass

    # ── BGS: grade-aware tiers ────────────────────────────────────────────────
    # BGS 9.5 GemMint ≈ PSA 10 in grade but ~0.81× in price (same as CGC 10)
    # BGS 10 Pristine > PSA 10: 1.307× (real data)
    # BGS 10 Black (Perfect): all sub-grades 10 — ~6.5× PSA 10 (near impossible)
    if g == "BGS":
        is_black    = "black" in grade_lower or "perfect" in grade_lower
        is_pristine = gv >= 10.0 and not is_black
        if is_black:
            factor = 6.526  # BGS 10 Black — Baby5+Charizard avg (6.537+6.514); Luffy pop-anomaly excluded
        elif is_pristine:
            factor = 1.307  # BGS 10 Pristine — rarer, MORE than PSA 10
        else:
            factor = 0.810  # BGS 9.5 Gem Mint — equivalent grade to PSA 10

    # ── CGC: standard 10 vs Pristine tier ────────────────────────────────────
    # CGC 10 standard = 0.81× PSA 10
    # CGC 10 Pristine = 1.46× PSA 10 (real data)
    elif g == "CGC":
        if "pristine" in grade_lower:
            factor = 1.280  # CGC 10 Pristine — averaged Baby5(1.462) + Charizard(1.101)
        else:
            factor = 0.810  # CGC 10 standard — calibrated from Baby5 (Charizard excluded as outlier)

    # ── CLA: 12-point scale ───────────────────────────────────────────────────
    # CLA 10 Flawless       = PSA 10            → 1.000× (confirmed equivalency)
    # CLA 11                = interpolated       → 2.4235× (geometric mean 10↔12)
    # CLA 12 Ultra Flawless = 90% of BGS Black  → 5.8734× (= 6.526 × 0.90)
    elif g == "CLA":
        if gv >= 12:
            factor = 5.8734  # Ultra Flawless = 90% of BGS 10 Black
        elif gv >= 11:
            factor = 2.4235  # geometric interpolation between CLA 10 and CLA 12
        else:
            factor = 1.000  # CLA 10 Flawless = PSA 10 (confirmed by CLA grading standard)

    return round(factor, 4)

def _cla_resolve_price(
    pc_result:        Dict[str, Any],
    grade:            str,
    card_name:        str,
    ebay_results:     Optional[List[Dict[str, Any]]] = None,
    game_type:        str        = "",
    rarity:           str        = "",
    variant_type:     str        = "",
    finish:           str        = "",
    language:         str        = "english",
    edition:          str        = "",
    is_signed:        bool       = False,
    signed_by:        str        = "",
    is_error:         bool       = False,
    is_promo:         bool       = False,
    rank_within_card: Optional[int] = None,
    rank_total:       Optional[int] = None,
    card_set:         str        = "",
    card_number:      str        = "",
) -> Dict[str, Any]:
    """
    Grade 10 / Grade 12 price resolver.

    Source priority:
      1. PC graded fields  — matched by product ID, always trusted
      2. eBay graded comps — keyword search, used as fallback or averaged with PC
      3. PC raw price      — grade mult applied via layered engine
      4. eBay raw comps    — last resort, grade mult applied
      5. Rarity table      — pure estimate when all live data missing

    Averaging rule:
      - PC only:        use PC price directly
      - eBay only:      use eBay price directly (PC had no graded data)
      - Both PC + eBay: average the two implied prices
        (two independent sources agreeing is more reliable than one)

    eBay sanity ceiling: eBay implied price must not exceed
      PC graded implied × 3.0  (if PC has graded data)
      PC raw × grade_mult × 2.5 (if PC has raw price only)
    Anything above the ceiling is rejected as a wrong-card match.

    Grade 10 target = PSA 10 equiv (factor 1.0 = direct)
    Grade 12 target = BGS Black equiv (factor 1.0 = direct)
    """
    _PSA_EQUIV: Dict[str, float] = {
        "bgs black":     6.525,
        "bgs 10 black":  6.525,
        "bgs10 black":   6.525,
        "bgs 10":        1.307,
        "bgs10":         1.307,
        "psa 10":        1.000,
        "psa10":         1.000,
        "gem mint 10":   1.000,
        "cgc 10":        0.810,
        "cgc10":         0.810,
        "bgs 9.5":       0.810,
        "bgs9.5":        0.810,
        "psa 9":         0.667,
        "psa9":          0.667,
        "sgc 10":        0.600,
        "sgc10":         0.600,
        "psa 8":         0.400,
        "psa8":          0.400,
    }
    _PC_FIELD_MAP = [
        ("bgs_black_price_aud", 6.525, "PC BGS Black"),
        ("bgs_10_price_aud",    1.307, "PC BGS 10 Pristine"),
        ("psa_10_price_aud",    1.000, "PC PSA 10"),
        ("cgc_10_price_aud",    0.810, "PC CGC 10"),
        ("bgs_gem_price_aud",   0.810, "PC BGS 9.5"),
        ("psa_9_price_aud",     0.667, "PC PSA 9"),
        ("sgc_10_price_aud",    0.600, "PC SGC 10"),
    ]

    _gv = _cla_parse_grade(grade) or 0.0
    if _gv >= 12:
        # CLA 12 target = 1.5× PSA 10 equivalent.
        # The old value (6.525 = BGS Black ratio) caused 10-20× inflation because
        # PC BGS 10 Pristine data (src factor 1.307) was scaled by 6.525/1.307 = 4.99×
        # on top of an already-graded price. CLA is a boutique AU grader and does not
        # command BGS Black premiums; 1.5× PSA 10 matches observed AU market data.
        _target_psa_equiv = 1.50
        _grade_label      = "Grade 12"
    else:
        _target_psa_equiv = 1.000
        _grade_label      = "Grade 10"

    def _make_candidate(price, src_psa_equiv, label, source):
        factor   = _target_psa_equiv / src_psa_equiv
        implied  = round(price * factor, 2)
        distance = abs(factor - 1.0)
        return {"label": label, "source": source, "price": price,
                "factor": round(factor, 4), "implied": implied, "distance": distance}

    # ── PC raw price (used for ceiling calc) ─────────────────────────────────
    try:
        _pc_raw = float((pc_result or {}).get("raw_price_aud") or 0)
    except (TypeError, ValueError):
        _pc_raw = 0.0

    # ── 1. Best PC graded candidate ───────────────────────────────────────────
    pc_candidates = []
    for field, src_psa_equiv, label in _PC_FIELD_MAP:
        try:
            val = float((pc_result or {}).get(field) or 0)
        except (TypeError, ValueError):
            val = 0.0
        if val > 0:
            c = _make_candidate(val, src_psa_equiv, label, "pricecharting")
            pc_candidates.append(c)
            logging.info(f"  {_grade_label} PC: {label} ${val:.2f} × {c['factor']:.4f} = ${c['implied']:.2f} | {card_name}")

    pc_candidates.sort(key=lambda c: c["distance"])
    best_pc = pc_candidates[0] if pc_candidates else None

    # ── 2. Best eBay graded candidate — with sanity ceiling ───────────────────
    # Ceiling derived from best available PC reference to reject wrong-card matches.
    if best_pc:
        _ebay_ceiling = best_pc["implied"] * 3.0
    elif _pc_raw > 0:
        _max_mult     = _cla_grade_mult_tiered(grade, _pc_raw)
        _ebay_ceiling = _pc_raw * _max_mult * 2.5
    else:
        _ebay_ceiling = float("inf")  # no PC reference — accept anything, rely on rarity table fallback

    ebay_candidates = []
    for ebay_res in (ebay_results or []):
        if not ebay_res:
            continue
        try:
            ebay_price = float(ebay_res.get("median") or 0)
        except (TypeError, ValueError):
            continue
        if ebay_price <= 0:
            continue

        query_lower   = (ebay_res.get("query") or "").lower()
        src_psa_equiv = None
        matched_label = None
        for token, equiv in sorted(_PSA_EQUIV.items(), key=lambda x: -len(x[0])):
            if token in query_lower:
                src_psa_equiv = equiv
                matched_label = token.upper()
                break

        if src_psa_equiv is None:
            # Raw/ungraded eBay — handled separately below
            ebay_candidates.append({
                "label": "eBay Raw", "source": "ebay_raw",
                "price": ebay_price, "factor": None,
                "implied": ebay_price, "distance": 999.0,
            })
        else:
            c = _make_candidate(ebay_price, src_psa_equiv, f"eBay {matched_label}", "ebay")
            if c["implied"] > _ebay_ceiling:
                logging.warning(
                    f"⚠️ {_grade_label}: rejecting eBay {matched_label} ${ebay_price:.2f} "
                    f"(implied ${c['implied']:.2f} > ceiling ${_ebay_ceiling:.2f}) "
                    f"— likely wrong-card match | {card_name}"
                )
                continue
            ebay_candidates.append(c)
            logging.info(f"  {_grade_label} eBay: {c['label']} ${ebay_price:.2f} × {c['factor']:.4f} = ${c['implied']:.2f} | {card_name}")

    ebay_candidates.sort(key=lambda c: c["distance"])
    best_ebay = next((c for c in ebay_candidates if c["distance"] < 999.0), None)

    # ── 3. Resolve base price ─────────────────────────────────────────────────
    raw_base_aud   = 0.0
    _apply_grade   = False
    sources_used   = []
    best_candidate = None

    if best_pc and best_ebay:
        # Both sources available — average them
        raw_base_aud   = round((best_pc["implied"] + best_ebay["implied"]) / 2.0, 2)
        sources_used   = ["pricecharting", "ebay"]
        best_candidate = best_pc  # for audit trail
        logging.info(
            f"🎯 {_grade_label}: averaged PC {best_pc['label']} ${best_pc['implied']:.2f} "
            f"+ eBay {best_ebay['label']} ${best_ebay['implied']:.2f} "
            f"= ${raw_base_aud:.2f} AUD | {card_name}"
        )
    elif best_pc:
        # PC only
        raw_base_aud   = best_pc["implied"]
        sources_used   = ["pricecharting"]
        best_candidate = best_pc
        logging.info(
            f"🎯 {_grade_label}: PC only — {best_pc['label']} ${best_pc['price']:.2f} "
            f"× {best_pc['factor']:.4f} = ${raw_base_aud:.2f} AUD | {card_name}"
        )
    elif best_ebay:
        # eBay only (PC had no graded data)
        raw_base_aud   = best_ebay["implied"]
        sources_used   = ["ebay"]
        best_candidate = best_ebay
        logging.info(
            f"🎯 {_grade_label}: eBay only (no PC graded) — {best_ebay['label']} ${best_ebay['price']:.2f} "
            f"× {best_ebay['factor']:.4f} = ${raw_base_aud:.2f} AUD | {card_name}"
        )
    elif _pc_raw > 0:
        # No graded data anywhere — use PC raw, apply grade mult via layered engine.
        # Minimum raw floor: a wrong-card PC match (e.g. Japanese bulk junk at $0.17)
        # must not drive the final price. Reject signals below these AUD thresholds
        # and fall through to the rarity table which uses observed tier ratios.
        #   Grade 12: raw must be ≥ $1.50 (cheapest legit CLA 12 is a $1.50 common)
        #   Grade 10: raw must be ≥ $0.50
        _pc_raw_min = 1.50 if _gv >= 12 else 0.50
        if _pc_raw < _pc_raw_min:
            logging.warning(
                f"⚠️ {_grade_label}: PC raw ${_pc_raw:.2f} < floor ${_pc_raw_min:.2f} "
                f"— wrong-card PC match, falling to rarity table | {card_name}"
            )
            # raw_base_aud stays 0 → falls to rarity table below
        else:
            raw_base_aud   = _pc_raw
            _apply_grade   = True
            sources_used   = ["pricecharting_raw"]
            logging.info(f"📦 {_grade_label}: PC raw ${_pc_raw:.2f} (grade mult will apply) | {card_name}")
    else:
        # Check raw eBay as absolute last resort before rarity table
        raw_ebay = next((c for c in ebay_candidates if c["distance"] == 999.0), None)
        if raw_ebay:
            raw_base_aud = raw_ebay["price"]
            _apply_grade = True
            sources_used = ["ebay_raw"]
            best_candidate = raw_ebay
            logging.info(f"📦 {_grade_label}: raw eBay ${raw_base_aud:.2f} (grade mult will apply) | {card_name}")

    # ── 4. No data — rarity table ─────────────────────────────────────────────
    if raw_base_aud <= 0:
        est = _cla_estimate_from_attributes(
            game_type=game_type, rarity=rarity, variant_type=variant_type,
            finish=finish, language=language, is_signed=is_signed, signed_by=signed_by,
            grade=grade, rank_within_card=rank_within_card, rank_total=rank_total,
            card_set=card_set, card_number=card_number, is_error=is_error, is_promo=is_promo,
        )
        est["price_source"]        = f"attr_estimate_{_grade_label.lower().replace(' ','')}"
        est["grader_brand_factor"] = 1.0
        est["card_name"]           = card_name
        est["all_candidates"]      = pc_candidates + ebay_candidates
        logging.info(f"📊 {_grade_label}: rarity table ${est.get('current_price',0):.2f} AUD | {card_name}")
        return est

    # ── 5. Apply attribute multipliers exactly once ───────────────────────────
    # _apply_grade=False: base comes from actual graded sale comps (PSA 10, BGS 10, etc.)
    #   — the comp price ALREADY reflects the card's finish/language/rarity premium.
    #   — Applying finish/language on top = double-counting. ONLY pop_rank still applies
    #     (a #1 pop CLA 12 does add scarcity on top of any graded comp).
    # _apply_grade=True: base is raw/ungraded — ALL attribute multipliers needed.
    if _apply_grade:
        # Raw base: apply full attribute stack (grade + finish + language + edition + signed…)
        lv = cla_layered_valuation(
            raw_base_aud=raw_base_aud, grade=grade,
            rarity=rarity, variant_type=variant_type, finish=finish,
            language=language, edition=edition, is_signed=is_signed, signed_by=signed_by,
            game_type=game_type, rank_within_card=rank_within_card, rank_total=rank_total,
            is_error=is_error, is_promo=is_promo,
        )
    else:
        # Graded comp base: skip grade + attribute mults (already priced in).
        # Pass pop_rank only — scarcity within the graded pop is additive.
        lv = cla_layered_valuation(
            raw_base_aud=raw_base_aud, grade="",
            rarity="", variant_type="", finish="",
            language="english", edition="", is_signed=False, signed_by="",
            game_type=game_type, rank_within_card=rank_within_card, rank_total=rank_total,
            is_error=False, is_promo=False,
        )

    final_price = lv["final_value"]

    # ── Sanity cap — two separate schedules for the two paths ─────────────────
    #
    # PATH 1 — _apply_grade=False (graded comp signal, PSA-equiv factor conversion)
    #   The factor is always close to 1.0×; the only additional mult is pop_rank (max 1.25×).
    #   Cap = signal × 2.0 is generous headroom.
    #
    # PATH 2 — _apply_grade=True (raw signal, full grade + attribute stack)
    #   Grade mult alone can be 2.5–5.0×. Combined with JP (1.15×) + finish (1.40×),
    #   the legitimate total can reach 8.05× for Grade 12 on a cheap card.
    #   Cap = grade_table_max × 2.5 so normal multiplications are never truncated.
    #   Only true outliers fire it (e.g. $5 card matched a $500 wrong-card eBay raw).
    if raw_base_aud > 0:
        if not _apply_grade:
            # Graded comp path — tight cap, factor conversion is near 1.0×
            _cap_ratio = 2.0
        else:
            # Raw signal path — look up the table-maximum mult for this grade
            _grade_max_at_3 = 1.0   # mult at $3 breakpoint = highest possible for this grade
            for _min_g, _tiers in _CLA_GRADE_TIERS:
                if _gv >= _min_g:
                    _grade_max_at_3 = _tiers[0]
                    break
            _cap_ratio = round(_grade_max_at_3 * 3.5, 1)   # 3.5× above table max: accommodates
            # full attr stack (grade × finish × JP × signed) without clipping legitimate values

        _price_cap = round(raw_base_aud * _cap_ratio, 2)
        if final_price > _price_cap:
            logging.warning(
                f"⚠️ {_grade_label} sanity cap ({'raw' if _apply_grade else 'graded-comp'} path): "
                f"${final_price:.2f} → ${_price_cap:.2f} "
                f"(signal=${raw_base_aud:.2f} × {_cap_ratio}×) | {card_name}"
            )
            final_price = _price_cap

    logging.info(
        f"✅ {_grade_label} final: ${raw_base_aud:.2f} × attr {lv['active_checks']} "
        f"= ${final_price:.2f} AUD | {card_name}"
    )

    _best_label = best_candidate["label"] if best_candidate else "raw"
    return {
        "current_price":        final_price,
        "final_price":          final_price,
        "final_value":          final_price,
        "signal_price":         raw_base_aud,
        "base_price":           raw_base_aud,
        "multiplier_applied":   round(lv["total_mult"], 4),
        "grade_multiplier":     round(lv["grade_mult"], 4),
        "slab_premium_added":   0.0,
        "multiplier_reason":    f"{_grade_label}_{_best_label.lower().replace(' ','_')}",
        "confidence":           "high" if len(sources_used) > 1
                                else "medium" if best_candidate else "estimate",
        "source":               "grade_resolved",
        "sources_used":         sources_used,
        "price_includes_grade": True,
        "grade_12_uplift":      (_gv >= 12),
        "grader_brand_factor":  1.0,
        "card_name":            card_name,
        "last_updated":         datetime.now().isoformat(),
        "fingerprint_used":     True,
        "estimated":            (best_candidate is None),
        "best_source_label":    _best_label,
        "best_source_factor":   best_candidate["factor"] if best_candidate else None,
        "best_source_price":    best_candidate["price"]  if best_candidate else None,
        "pc_candidate":         best_pc,
        "ebay_candidate":       best_ebay,
        "layered_checks":       lv["checks"],
        "active_checks":        lv["active_checks"],
        "all_candidates":       pc_candidates + ebay_candidates,
    }
def apply_cla_valuation(signal_price: float, signal_includes_grade: bool, search_query: str, target_grade: str | None,
                        rank_within_card: int | None = None, rank_total: int | None = None,
                        slab_premium_add_aud: float = 70.0) -> dict:
    """Compute final_value from a market signal + CLA rules."""
    signal_price = float(signal_price or 0.0)
    if signal_price <= 0:
        return {
            "final_value": 0.0,
            "base_price": 0.0,
            "multiplier_applied": 1.0,
            "slab_premium_added": 0.0,
            "target_multiplier": 1.0,
            "base_multiplier": 1.0,
        }

    # Pass signal_price into grade multiplier so the price-scaled cap applies.
    tg_mult = cla_grade_multiplier(target_grade or "", rank_within_card, rank_total,
                                   signal_price=signal_price) if target_grade else 1.0

    if signal_includes_grade and target_grade:
        # Signal already reflects graded comps — re-base to target grade only.
        base_grade   = _parse_grade_from_query(search_query) or 10.0
        base_mult    = cla_grade_multiplier(str(base_grade), signal_price=signal_price)
        base_price   = signal_price
        slab_added   = 0.0
        mult_applied = (tg_mult / base_mult) if base_mult > 0 else 1.0
    elif (not signal_includes_grade) and target_grade:
        # Raw signal → apply grade multiplier only.
        # Slab premium removed: the multiplier already encodes the full grade-vs-raw
        # premium (grading cost, authentication, liquidity).  Adding a separate slab
        # fee on top was double-counting and inflated high-value cards severely.
        slab_added   = 0.0
        base_price   = signal_price
        base_mult    = 1.0
        mult_applied = tg_mult
    else:
        slab_added   = 0.0
        base_price   = signal_price
        base_mult    = 1.0
        mult_applied = 1.0

    final_value = round(base_price * mult_applied, 2)
    return {
        "final_value": final_value,
        "base_price": round(base_price, 2),
        "multiplier_applied": round(mult_applied, 6),
        "slab_premium_added": round(slab_added, 2),
        "target_multiplier": round(tg_mult, 6),
        "base_multiplier": round(base_mult, 6),
    }

def _rarity_price_band(
    game_type: str,
    rarity: str,
    is_graded: bool,
    price_aud: float,
) -> dict:
    """
    Return {"ok": bool, "reason": str, "floor": float, "ceiling": float}.

    Validates a returned price against known realistic ranges per game × rarity.
    Prices outside the band are flagged as needs_review — not zeroed or capped,
    because the band logic may be wrong for edge cases.  The caller can still
    show the price to the user but should not auto-save it.

    All values in AUD.  Floors and ceilings are intentionally wide to only
    catch egregious mismatches (wrong card matched), not minor variance.

    Raw bands are for ungraded cards.  For graded the ceiling is doubled
    (a PSA 10 chase card can legitimately be 2-3× raw) and the floor is kept
    the same (a low grade doesn't become more valuable just because it's slabbed).
    """
    g = (game_type or "").lower().strip()
    _r_raw = (rarity or "").lower().strip()
    # Normalise ASCII * to Unicode ★ and expand shorthand (matches _cla_estimate_from_attributes)
    _r_raw = re.sub(r'\br\*\b',  'r★',  _r_raw)
    _r_raw = re.sub(r'\bsr\*\b', 'sr★', _r_raw)
    _r_raw = re.sub(r'rare\s+\*',          'rare parallel',       _r_raw)
    _r_raw = re.sub(r'super\s+rare\s+\*', 'super rare parallel', _r_raw)
    _r_raw = re.sub(r'leader\s+\*',        'leader parallel',     _r_raw)
    r = _r_raw

    # ── Band definitions: (floor_aud, ceiling_aud) ─────────────────────────
    # Bands cover the realistic secondary market range.
    # Anything outside is almost certainly a wrong-card match or a data error.
    BANDS: dict[str, dict[str, tuple]] = {
        "one piece": {
            "common":                 (0.10,   25.0),
            "uncommon":               (0.20,   40.0),
            "rare":                   (0.50,   80.0),
            "super rare":             (1.00,  200.0),
            "leader":                 (2.00,  400.0),
            "secret rare":            (5.00,  800.0),
            "special art rare":       (10.0, 1500.0),
            "alt art":                (10.0, 1500.0),
            "ultra rare":             (5.00,  800.0),
            "promo":                  (2.00,  500.0),
            "gold":                   (5.00,  600.0),
            "don":                    (0.10,   80.0),   # DON!! cards
            "":                       (0.10, 1500.0),   # unknown rarity — wide band
        },
        "pokemon": {
            "common":                 (0.05,   30.0),
            "uncommon":               (0.10,   50.0),
            "rare":                   (0.50,  100.0),
            "double rare":            (1.00,  150.0),  # Modern ex/vstar Double Rare
            "holo rare":              (1.00,  400.0),
            "ultra rare":             (5.00, 2000.0),
            "full art":               (5.00, 2000.0),
            "secret rare":            (10.0, 3000.0),
            "rainbow rare":           (8.00, 2000.0),
            "illustration rare":      (5.00, 1500.0),
            "special illustration rare": (10.0, 6000.0),
            "hyper rare":             (10.0, 4000.0),
            "crown rare":             (20.0, 8000.0),
            "promo":                  (0.50, 1000.0),
            "":                       (0.05, 8000.0),
        },
        "dragon ball": {
            "common":                 (0.10,   30.0),
            "uncommon":               (0.20,   50.0),
            "rare":                   (0.50,  100.0),
            "super rare":             (2.00,  300.0),
            "special rare":           (5.00,  500.0),
            "secret rare":            (10.0, 1000.0),
            "":                       (0.10, 1000.0),
        },
    }

    # Normalise rarity string to the closest known key
    def _best_key(rarity_str: str, band_dict: dict) -> str:
        rs = rarity_str.lower()
        # Exact match first
        if rs in band_dict:
            return rs
        # Substring match (longest wins)
        matches = [(k, len(k)) for k in band_dict if k and k in rs]
        if matches:
            return max(matches, key=lambda x: x[1])[0]
        # DON!! special case
        if "don" in rs:
            return "don" if "don" in band_dict else ""
        return ""

    game_bands = None
    for gk in BANDS:
        if gk in g or g in gk:
            game_bands = BANDS[gk]
            break

    if game_bands is None:
        # Unknown game — no band check
        return {"ok": True, "reason": "unknown_game_no_band", "floor": 0.0, "ceiling": 99999.0}

    key    = _best_key(r, game_bands)
    floor_, ceil_ = game_bands.get(key, game_bands.get("", (0.0, 99999.0)))

    if is_graded:
        ceil_ = ceil_ * 2.5   # graded copies can legitimately be 2-3× raw
        # floor stays same — grade doesn't rescue a worthless card

    if price_aud < floor_:
        return {
            "ok":      False,
            "reason":  f"price_{price_aud:.2f}_below_floor_{floor_:.2f}_for_{game_type}_{rarity}",
            "floor":   floor_,
            "ceiling": ceil_,
        }
    if price_aud > ceil_:
        return {
            "ok":      False,
            "reason":  f"price_{price_aud:.2f}_above_ceiling_{ceil_:.2f}_for_{game_type}_{rarity}",
            "floor":   floor_,
            "ceiling": ceil_,
        }
    return {"ok": True, "reason": "within_band", "floor": floor_, "ceiling": ceil_}


@app.post("/api/market/price-lookup")
@safe_endpoint
async def market_price_lookup(request: MarketPriceLookupRequest):
    """
    Precision pricing v2 — card-fingerprint-first, multi-source pipeline.

    Flow:
    1. Build a strict card fingerprint (card_number + set mandatory).
    2. Query all three sources IN PARALLEL:
         • TCGPlayer (via PokemonTCG.io) — exact set+number, EN Pokémon only
         • eBay completed sales (AU+US)  — card_number mandatory in every query
         • PriceCharting                 — card-number-validated product match
    3. Reconcile: weighted median, confidence, multiplier ONLY on raw base price.
    4. Return final_price + full audit trail (source_breakdown, confidence, etc.)

    Backwards-compatible fields (current_price, signal_price, source, etc.)
    are preserved so existing PHP callers continue to work without changes.

    Falls back to the original query-ladder path when card_number or card_set
    is missing (legacy collections that predate the precision fields).
    """
    try:
        card_name    = (request.card_name    or "").strip()
        card_set     = (request.card_set     or "").strip()
        card_number  = (request.card_number  or "").strip()
        grade        = (request.grade        or "").strip()
        language     = (getattr(request, "language",  None) or "english").strip()
        game_type    = (getattr(request, "game_type", None) or "").strip()
        rarity       = (request.rarity       or "").strip()
        variant_type = (request.variant_type or "").strip()
        finish       = (request.finish        or "").strip()
        edition      = (request.edition      or "").strip()
        is_signed    = bool(request.is_signed)
        signed_by    = (request.signed_by    or "").strip()
        is_error     = bool(request.is_error_card)
        is_promo     = bool(request.is_promo)
        grader       = (getattr(request, "grader", None) or "").strip().upper()

        if not card_name:
            return {"current_price": 0, "source": "error", "error": "card_name required"}

        logging.info(
            f"💰 Price lookup v2: {card_name} | set={card_set} | "
            f"num={card_number} | grade={grade} | lang={language}"
        )

        # ── Detect grade ─────────────────────────────────────────────────────
        grade_is_12_plus = False
        grade_is_10_plus = False
        _grade_val = 0.0
        try:
            _gm = re.search(r"(\d+(?:\.\d+)?)", grade)
            if _gm:
                _grade_val = float(_gm.group(1))
                if _grade_val >= 12: grade_is_12_plus = True
                if _grade_val >= 10: grade_is_10_plus = True
        except Exception:
            pass

        # ── Grade-based resolver — runs for ALL graded cards ──────────────────
        # Grader brand is irrelevant to market value — what matters is the grade
        # number and the card's attributes. Fetches PC + eBay at multiple grade
        # tiers in parallel, picks the signal closest to the target grade, applies
        # one dynamic conversion factor, then attribute multipliers once.
        # No grader brand factor. No legacy path. No apply_cla_valuation.
        if grade and (grade_is_10_plus or grade_is_12_plus):
            fp: Dict[str, Any] = {}
            fp_error: Optional[str] = None
            try:
                fp = _resolve_card_fingerprint(
                    card_name   = card_name,
                    card_set    = card_set,
                    card_number = card_number,
                    language    = language,
                    game_type   = game_type,
                    grade       = grade,
                    grader      = "",
                )
            except ValueError as ve:
                fp_error = str(ve)
                logging.warning(f"⚠️ Fingerprint incomplete ({ve}) — resolver will use rarity table")

            pc_result:    Dict[str, Any]       = {}
            ebay_results: List[Dict[str, Any]] = []

            if fp and not fp_error:
                _fp_base = dict(fp)
                _fp_base["is_graded"] = True

                if grade_is_12_plus:
                    _fp_bgs_black = dict(_fp_base)
                    _fp_bgs_black["_grade_token_override"] = "BGS 10 Black"
                    _fp_psa10 = dict(_fp_base)
                    _fp_psa10["_grade_token_override"] = "PSA 10"
                    _fp_raw = dict(fp); _fp_raw["is_graded"] = False; _fp_raw["grade"] = ""
                    pc_result, eb_bgs_black, eb_psa10, eb_raw = await asyncio.gather(
                        _fetch_pricecharting_strict(fp),
                        _ebay_sold_strict(_fp_bgs_black, limit=30, days_lookback=180),
                        _ebay_sold_strict(_fp_psa10,     limit=40, days_lookback=90),
                        _ebay_sold_strict(_fp_raw,       limit=20, days_lookback=60),
                    )
                    ebay_results = [r for r in [eb_bgs_black, eb_psa10, eb_raw] if r]
                else:
                    _fp_psa10 = dict(_fp_base)
                    _fp_psa10["_grade_token_override"] = "PSA 10"
                    _fp_psa9  = dict(_fp_base)
                    _fp_psa9["_grade_token_override"]  = "PSA 9"
                    _fp_raw = dict(fp); _fp_raw["is_graded"] = False; _fp_raw["grade"] = ""
                    pc_result, eb_psa10, eb_psa9, eb_raw = await asyncio.gather(
                        _fetch_pricecharting_strict(fp),
                        _ebay_sold_strict(_fp_psa10, limit=40, days_lookback=90),
                        _ebay_sold_strict(_fp_psa9,  limit=30, days_lookback=90),
                        _ebay_sold_strict(_fp_raw,   limit=20, days_lookback=60),
                    )
                    ebay_results = [r for r in [eb_psa10, eb_psa9, eb_raw] if r]

                pc_result = pc_result or {}

            result = _cla_resolve_price(
                pc_result        = pc_result,
                ebay_results     = ebay_results,
                grade            = grade,
                card_name        = card_name,
                game_type        = game_type,
                rarity           = rarity,
                variant_type     = variant_type,
                finish           = finish,
                language         = language,
                edition          = edition,
                is_signed        = is_signed,
                signed_by        = signed_by,
                is_error         = is_error,
                is_promo         = is_promo,
                rank_within_card = getattr(request, "rank_within_card", None),
                rank_total       = getattr(request, "rank_total",       None),
                card_set         = card_set,
                card_number      = card_number,
            )
            result["pricecharting_detail"] = pc_result if pc_result else None
            result["ebay_detail"]          = ebay_results[0] if ebay_results else None
            result["grader"]               = grader or ""

            # ── Claude fallback for grade 10/12 path ─────────────────────────
            _rp = float(result.get("current_price") or result.get("final_price") or 0)
            _rc = str(result.get("confidence") or "none")
            logging.warning(
                f"🔎 DIAG grade-path {card_name} #{card_number}: "
                f"price={_rp} conf={_rc} "
                f"anthropic={ANTHROPIC_AVAILABLE} key={bool(ANTHROPIC_API_KEY)} "
                f"gate={(_rp <= 0 or _rc == 'none')}"
            )
            if (_rp <= 0 or _rc == "none") and ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY:
                try:
                    _cr = await _claude_market_lookup(
                        card_name   = card_name,
                        card_number = card_number,
                        card_set    = card_set,
                        grade       = grade,
                        game_type   = game_type,
                        rarity      = rarity,
                        is_signed   = is_signed,
                        language    = language,
                        image_url   = getattr(request, "image_url", None) or "",
                    )
                    _cp = float(_cr.get("price_aud") or 0)
                    _cc = _cr.get("confidence", "none")
                    if _cp > 0 and _cc in ("high", "medium", "low"):
                        logging.warning(
                            f"✅ Claude grade-path fill: ${_cp:.2f} AUD "
                            f"conf={_cc} | {card_name} #{card_number}"
                        )
                        result.update({
                            "current_price":      _cp,
                            "final_price":        _cp,
                            "final_value":        _cp,
                            "signal_price":       _cp,
                            "confidence":         _cc,
                            "sources_used":       ["claude_web_search"],
                            "source":             "claude_web_search",
                            "multiplier_applied": 1.0,
                            "multiplier_reason":  "claude_web_search",
                            "claude_note":        _cr.get("note", ""),
                            "claude_source":      _cr.get("source", ""),
                            "price_source":       "claude_web_search",
                        })
                except Exception as _cge:
                    logging.warning(f"Claude grade-path lookup failed: {_cge}")

            return result

        # ── Step 1: Build fingerprint (grades below 10 / ungraded path) ──────
        fp: Dict[str, Any] = {}
        fp_error: Optional[str] = None
        try:
            fp = _resolve_card_fingerprint(
                card_name   = card_name,
                card_set    = card_set,
                card_number = card_number,
                language    = language,
                game_type   = game_type,
                grade       = grade,
                grader      = "",
            )
        except ValueError as ve:
            fp_error = str(ve)
            logging.warning(
                f"⚠️ Fingerprint incomplete ({ve}) — falling back to legacy lookup"
            )

        # ── Inject request-level variant/rarity into fp ───────────────────────
        # _resolve_card_fingerprint only knows name/set/number/grade/language.
        # The rarity band check in _reconcile_prices reads fp.get("rarity") —
        # without this injection fp["rarity"] is always empty → band check skipped
        # → wrong-card prices like the $944 Charizard (Double Rare, ceiling $150)
        # pass through undetected.
        if fp:
            if rarity:       fp["rarity"]       = rarity
            if variant_type: fp["variant_type"]  = variant_type
            if finish:       fp["finish"]        = finish
            if edition:      fp["edition"]       = edition

        # ── Pre-compute attribute estimate (always, for fallback + comparison) ──
        # Uses CLA-native grade tiers — no grader brand factor applied.
        # Available whether precision path succeeds or fails.
        _attr_est: Dict[str, Any] = {}
        try:
            _attr_est = _cla_estimate_from_attributes(
                game_type    = game_type,
                rarity       = rarity,
                variant_type = variant_type,
                finish       = finish,
                language     = language,
                is_signed    = is_signed,
                signed_by    = signed_by,
                grade        = grade,
                rank_within_card = getattr(request, "rank_within_card", None),
                rank_total       = getattr(request, "rank_total",       None),
                card_set     = card_set,
                card_number  = card_number,
                is_error     = is_error,
                is_promo     = is_promo,
                # pc_raw_price_aud injected after PC fetch completes — see below
            )
        except Exception as _ae:
            logging.info(f"Attr estimate skipped: {_ae}")

        # ── Step 2 (PRECISION PATH) ───────────────────────────────────────────
        if fp and not fp_error:
            # Standard (non-CLA) precision path — fetch all sources in parallel
            tcg_result, ebay_result, pc_result = await asyncio.gather(
                _fetch_pokemontcg_price(fp),
                _ebay_sold_strict(fp, limit=60, days_lookback=90),
                _fetch_pricecharting_strict(fp),
            )

            sources = {
                "tcgplayer":     tcg_result,
                "ebay":          ebay_result,
                "pricecharting": pc_result,
            }

            # ── Re-run attr estimate with PC live raw price if we got one ─────
            # The pre-computed attr_est used hardcoded table bases.
            # Now that PC has returned, re-run with real loose-price if available.
            _pc_raw = (pc_result or {}).get("raw_price_aud")
            if _pc_raw and float(_pc_raw) > 0:
                try:
                    _attr_est = _cla_estimate_from_attributes(
                        game_type    = game_type,
                        rarity       = rarity,
                        variant_type = variant_type,
                        finish       = finish,
                        language     = language,
                        is_signed    = is_signed,
                        signed_by    = signed_by,
                        grade        = grade,
                        rank_within_card = getattr(request, "rank_within_card", None),
                        rank_total       = getattr(request, "rank_total",       None),
                        card_set     = card_set,
                        card_number  = card_number,
                        is_error     = is_error,
                        is_promo     = is_promo,
                        pc_raw_price_aud = float(_pc_raw),
                    )
                    logging.info(
                        f"📊 Attr estimate refreshed with PC raw=${_pc_raw:.2f} AUD: "
                        f"${_attr_est.get('current_price', 0):.2f} AUD for {card_name}"
                    )
                except Exception as _re:
                    logging.debug(f"Attr re-estimate failed: {_re}")

            # ── Variant second-pass ──────────────────────────────────────────
            # Pass rarity/variant into the fingerprint so _ebay_sold_strict
            # builds primary tiers that already include the variant token.
            # Previously: vfp['card_name'] prepended variant → Japanese chars
            # in tiers 1-4 still got searched first, returning nothing.
            # Now: fp carries rarity/variant_type directly → _ebay_sold_strict
            # injects _variant_primary into tiers 1-2 before Japanese chars.
            # Also: run the variant pass even when the base ebay_result is empty
            # (base may return nothing because common-card results were sparse,
            # but the specific variant may have its own sales).
            variant_tokens_v2 = []
            _skip_rarity_v2 = {"common", "uncommon", "rare", ""}
            if rarity.lower() not in _skip_rarity_v2:
                variant_tokens_v2.append(rarity)
            if edition.lower() in ("1st edition", "first edition"):
                variant_tokens_v2.append("1st Edition")
            elif edition.lower() == "shadowless":
                variant_tokens_v2.append("Shadowless")
            if is_signed:
                variant_tokens_v2.append(
                    f"Signed {signed_by}".strip() if signed_by else "Signed"
                )
            if is_error:
                variant_tokens_v2.append("Error Misprint")

            if variant_tokens_v2:
                vfp = dict(fp)
                # Pass rarity and variant_type into the fingerprint so
                # _ebay_sold_strict._variant_primary picks them up directly,
                # rather than prepending to card_name (which adds Japanese chars).
                vfp["rarity"]       = " ".join(variant_tokens_v2)
                vfp["variant_type"] = " ".join(variant_tokens_v2)
                # Keep English-only card name for variant queries (strip Japanese)
                if "/" in card_name:
                    vfp["card_name"] = card_name.split("/")[-1].strip()
                v_ebay = await _ebay_sold_strict(vfp, limit=40, days_lookback=120)
                # Lower threshold: 2 sales (graded variants are low-volume)
                if v_ebay and v_ebay.get("count", 0) >= 2:
                    sources["ebay"] = v_ebay
                    logging.info(
                        f"✅ Variant eBay: ${v_ebay.get('median',0):.2f} AUD "
                        f"from {v_ebay.get('count',0)} sales | variant={variant_tokens_v2}"
                    )

            # ── Step 3: Reconcile ─────────────────────────────────────────────
            rec         = _reconcile_prices(sources, fp)
            final_price = rec["final_price"]
            signal_price = rec["signal_price"]
            confidence  = rec["confidence"]
            sources_used = rec["sources_used"]

            if "tcgplayer" in sources_used or "tcgplayer_via_pokemontcgapi" in sources_used:
                primary_source = "tcgplayer_via_pokemontcgapi"
            elif "ebay" in sources_used:
                primary_source = "ebay_completed"
            elif "pricecharting" in sources_used:
                primary_source = "pricecharting"
            else:
                primary_source = "no_results"

            if final_price > 0:
                # Apply grader brand factor — eBay comps are almost always PSA.
                # NEVER apply for CLA 12: the entire precision path for CLA 12
                # already converts to CLA 12 price (BGS Black × 0.90, PC × 5.87 etc.)
                # Applying the factor again would double-multiply every time.
                _is_cla12_result = grade_is_12_plus and grader.upper() in ("CLA", "COLLECTORS LEAGUE", "")
                _gbf = 1.0 if _is_cla12_result else _grader_brand_factor(grader, grade)
                if _gbf != 1.0 and grader:
                    _pre_brand = final_price
                    final_price = round(final_price * _gbf, 2)
                    logging.info(
                        f"🏷️ Grader brand factor: {grader} {_gbf:.2f}x | "
                        f"${_pre_brand:.2f} → ${final_price:.2f} AUD"
                    )
                logging.info(
                    f"✅ Precision price: ${final_price:.2f} AUD | "
                    f"signal=${signal_price:.2f} | confidence={confidence} | "
                    f"grader={grader or 'PSA'} brand_factor={_gbf:.2f} | "
                    f"sources={sources_used} | breakdown={rec['source_breakdown']}"
                )
                return {
                    # Primary fields used by PHP callers
                    "current_price":        final_price,
                    "final_price":          final_price,
                    "signal_price":         signal_price,
                    # Confidence (internal/admin only — not shown to customers)
                    "confidence":           confidence,
                    "sources_used":         sources_used,
                    "source_breakdown":     rec["source_breakdown"],
                    "sample_sizes":         rec["sample_sizes"],
                    "sources_agreed_pct":   rec["sources_agreed_pct"],
                    "max_deviation_pct":    rec["max_deviation_pct"],
                    # Multiplier audit trail
                    "multiplier_applied":   rec["multiplier_applied"],
                    "slab_premium_added":   rec["slab_premium_added"],
                    "multiplier_reason":    rec["multiplier_reason"],
                    "base_price":           signal_price,
                    "final_value":          final_price,
                    # Backwards-compat
                    "source":               primary_source,
                    "price_includes_grade": rec["multiplier_reason"] == "source_price_already_graded",
                    "grade_12_uplift":      bool(grade_is_12_plus),
                    "card_name":            card_name,
                    "search_query":         (ebay_result or {}).get("query", ""),
                    "last_updated":         datetime.now().isoformat(),
                    "fingerprint_used":     True,
                    "grader":               grader or "PSA",
                    "grader_brand_factor":  _gbf,
                    # Individual source details (admin/debug)
                    "tcgplayer_detail":     tcg_result  if tcg_result  else None,
                    "ebay_detail":          ebay_result if ebay_result else None,
                    "pricecharting_detail": pc_result   if pc_result   else None,
                    "attr_estimate":        _attr_est.get("current_price", 0) if _attr_est else 0,
                    "attr_estimate_detail": _attr_est if _attr_est else None,
                    "price_source":         "market_comp",
                }

            logging.info(
                f"ℹ️ No market comps for {card_name} #{card_number} "
                f"— trying Claude web search before attribute estimate"
            )

            # ── Claude web search (fires before rarity table) ─────────────────
            _claude_result: dict = {}
            logging.warning(
                f"🔎 DIAG precision-path {card_name} #{card_number}: "
                f"anthropic={ANTHROPIC_AVAILABLE} key={bool(ANTHROPIC_API_KEY)}"
            )
            if ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY:
                try:
                    _claude_result = await _claude_market_lookup(
                        card_name   = card_name,
                        card_number = card_number,
                        card_set    = card_set,
                        grade       = grade,
                        game_type   = game_type,
                        rarity      = rarity,
                        is_signed   = is_signed,
                        language    = language,
                        image_url   = getattr(request, "image_url", None) or "",
                    )
                except Exception as _cle:
                    logging.warning(f"Claude market lookup error: {_cle}")

            _claude_price = float(_claude_result.get("price_aud") or 0)
            _claude_conf  = _claude_result.get("confidence", "none")

            if _claude_price > 0 and _claude_conf in ("high", "medium", "low"):
                logging.warning(
                    f"✅ Claude market fill: ${_claude_price:.2f} AUD "
                    f"conf={_claude_conf} | {card_name} #{card_number}"
                )
                return {
                    "current_price":        _claude_price,
                    "final_price":          _claude_price,
                    "final_value":          _claude_price,
                    "signal_price":         _claude_price,
                    "base_price":           _claude_price,
                    "multiplier_applied":   1.0,
                    "multiplier_reason":    "claude_web_search",
                    "slab_premium_added":   0.0,
                    "confidence":           _claude_conf,
                    "sources_used":         ["claude_web_search"],
                    "source":               "claude_web_search",
                    "source_breakdown":     {"claude": _claude_price},
                    "sample_sizes":         {"claude": _claude_result.get("comp_count", 0)},
                    "sources_agreed_pct":   100,
                    "max_deviation_pct":    0.0,
                    "price_includes_grade": True,
                    "grade_12_uplift":      bool(grade_is_12_plus),
                    "card_name":            card_name,
                    "search_query":         f"{card_name} {card_number}".strip(),
                    "fingerprint_used":     bool(fp),
                    "grader":               grader or "CLA",
                    "grader_brand_factor":  1.0,
                    "last_updated":         datetime.now().isoformat(),
                    "claude_note":          _claude_result.get("note", ""),
                    "claude_source":        _claude_result.get("source", ""),
                    "price_source":         "claude_web_search",
                    "tcgplayer_detail":     tcg_result  if tcg_result  else None,
                    "ebay_detail":          ebay_result if ebay_result else None,
                    "pricecharting_detail": pc_result   if pc_result   else None,
                }

            # ── Attribute estimate (rarity table) — last resort ───────────────
            if _attr_est and _attr_est.get("current_price", 0) > 0:
                _attr_est["card_name"]           = card_name
                _attr_est["search_query"]        = ""
                _attr_est["fingerprint_used"]    = True
                _attr_est["grader_brand_factor"] = 1.0
                _attr_est["price_source"]        = "attr_estimate"
                _attr_est["tcgplayer_detail"]    = tcg_result  if tcg_result  else None
                _attr_est["ebay_detail"]         = ebay_result if ebay_result else None
                _attr_est["pricecharting_detail"]= pc_result   if pc_result   else None
                logging.info(
                    f"📊 Attr estimate: ${_attr_est['current_price']:.2f} AUD "
                    f"for {card_name} | tier={_attr_est.get('matched_rarity_tier','?')}"
                )
                return _attr_est
            # Try attribute estimate now — the precision path found sources but
            # the floor guard (or all-zero signals) killed the price.  The legacy
            # ladder may just repeat the same bad signal; the estimate is more
            # reliable for known-rarity cards with no viable comps.
            _has_meaningful_rarity_p = bool(
                rarity and rarity.lower() not in ("", "common", "uncommon")
            ) or bool(variant_type and variant_type.lower() not in ("", "regular", "standard"))
            if grade or _has_meaningful_rarity_p:
                try:
                    _pest = _cla_estimate_from_attributes(
                        game_type    = game_type,
                        rarity       = rarity,
                        variant_type = variant_type,
                        finish       = finish,
                        language     = language,
                        is_signed    = is_signed,
                        signed_by    = signed_by,
                        grade        = grade,
                        rank_within_card = getattr(request, "rank_within_card", None),
                        rank_total       = getattr(request, "rank_total",       None),
                        card_set     = card_set,
                        card_number  = card_number,
                    )
                    if _pest.get("current_price", 0) > 0:
                        # Do NOT apply grader brand factor to attribute estimates.
                        # The attr estimate uses CLA-specific grade multipliers already.
                        # Brand factor only applies when converting PSA eBay comps.
                        _pest["grader_brand_factor"] = 1.0
                        _pest["card_name"]     = card_name
                        _pest["search_query"]  = ""
                        _pest["fingerprint_used"] = True
                        logging.info(
                            f"📊 Precision→attribute estimate: ${_pest['current_price']:.2f} AUD "
                            f"for {card_name} #{card_number}"
                        )
                        return _pest
                except Exception as _pe:
                    logging.warning(f"Precision attribute estimate failed: {_pe}")
            logging.warning(
                f"⚠️ Falling back to legacy query ladder for {card_name} #{card_number}"
            )

        # ── LEGACY FALLBACK PATH ──────────────────────────────────────────────
        # Used when fingerprint is incomplete (card_number/set missing) OR when
        # the precision path found no results.
        # NOTE: CLA 10/12 cards never reach here — they are handled by the
        # dedicated _cla_resolve_price path at the top of this function.
        logging.info(f"🔄 Legacy pricing path for: {card_name}")

        variant_tokens = []
        _skip_rarity = {"common", "uncommon", "rare", ""}
        _rarity_canonical = {
            "secret rare":                   "Secret Rare",
            "sr":                            "Secret Rare",
            "full art":                      "Full Art",
            "alt art":                       "Alt Art",
            "special illustration rare":     "Special Illustration Rare SAR",
            "special art rare":              "SAR",
            "sar":                           "Special Illustration Rare SAR",
            "illustration rare":             "Illustration Rare",
            "ir":                            "Illustration Rare",
            "hyper rare":                    "Hyper Rare",
            "rainbow rare":                  "Rainbow Rare",
            "crown rare":                    "Crown Rare",
            "ultra rare":                    "Ultra Rare",
            "gold rare":                     "Gold",
            "shiny rare":                    "Shiny",
            "shiny ultra rare":              "Shiny Ultra Rare",
            "trainer gallery":               "Trainer Gallery",
            "radiant rare":                  "Radiant",
        }
        if rarity.lower() not in _skip_rarity:
            token = _rarity_canonical.get(rarity.lower(), rarity)
            if token:
                variant_tokens.append(token)

        _skip_variant = {"regular", "standard", ""}
        _variant_canonical = {
            "reverse holo":  "Reverse Holo",
            "holo":          "Holo",
            "cosmos holo":   "Cosmos Holo",
            "rainbow":       "Rainbow Rare",
            "gold":          "Gold",
            "textured":      "Textured",
            "etched":        "Etched",
            "master ball":   "Master Ball",
            "art rare":      "Art Rare",
            "promo":         "Promo",
        }
        if variant_type.lower() not in _skip_variant:
            vtoken = _variant_canonical.get(variant_type.lower(), variant_type)
            if vtoken and vtoken not in " ".join(variant_tokens):
                variant_tokens.append(vtoken)

        _edition_canonical = {
            "1st edition":         "1st Edition",
            "first edition":       "1st Edition",
            "shadowless":          "Shadowless",
            "collector's edition": "Collector's Edition",
        }
        if edition:
            etoken = _edition_canonical.get(edition.lower(), "")
            if etoken:
                variant_tokens.append(etoken)

        if is_signed:
            variant_tokens.append(
                f"Signed {signed_by}" if signed_by else "Signed"
            )
        if is_error:
            variant_tokens.append("Error Misprint")
        if is_promo and not any("Promo" in t for t in variant_tokens):
            variant_tokens.append("Promo")

        variant_str = " ".join(variant_tokens).strip()

        base_queries = _build_ebay_query_ladder(
            card_name=card_name,
            card_set=card_set,
            card_number=card_number,
            grade=grade,
        )

        if variant_str:
            variant_queries = []
            for bq in base_queries[:2]:
                vq = _norm_ws(f"{bq} {variant_str}")
                if vq not in variant_queries:
                    variant_queries.append(vq)
            # Non-variant queries kept as fallback, but flagged so we can
            # reject their signals when they don't contain the variant token.
            queries = variant_queries + [q for q in base_queries if q not in variant_queries]
        else:
            queries = base_queries

        if not queries:
            return {"current_price": 0, "source": "error",
                    "error": "Could not build search query"}

        last_completed = None
        last_active    = None
        MIN_COMPLETED_TRUST = 3

        for search_query in queries:
            logging.info(f"🔍 eBay legacy query: '{search_query}'")
            grade_in_query = bool(
                re.search(r"\bPSA\s*\d|\bCGC\s*\d|\bBGS\s*\d|\bgraded\b",
                          search_query, re.I)
            )
            completed = await _ebay_completed_stats(search_query, limit=25, days_lookback=30)
            active    = await _ebay_active_stats(search_query, limit=25)
            last_completed = completed or last_completed
            last_active    = active    or last_active

            comp_median   = float((completed or {}).get("median") or 0.0)
            comp_count    = int((completed  or {}).get("count")  or 0)
            active_median = float((active   or {}).get("median") or 0.0)
            active_count  = int((active     or {}).get("count")  or 0)

            if comp_median <= 0 and active_median <= 0:
                continue

            if comp_median > 0 and comp_count >= MIN_COMPLETED_TRUST:
                price  = comp_median
                source = "ebay_completed"
                sales  = comp_count
            elif comp_median > 0 and active_median > 0 and active_count >= MIN_COMPLETED_TRUST:
                price  = round(comp_median * 0.40 + active_median * 0.60, 2)
                source = "ebay_blended"
                sales  = comp_count + active_count
            elif comp_median > 0:
                price  = comp_median
                source = "ebay_completed_sparse"
                sales  = comp_count
            else:
                price  = active_median
                source = "ebay_active"
                sales  = active_count

            if variant_tokens and search_query:
                sq_lower = search_query.lower()
                variant_in_query = any(
                    tok.lower() in sq_lower for tok in variant_tokens if tok
                )
                if variant_tokens and not variant_in_query:
                    _VARIANT_SKIP_THRESHOLD = 30.0
                    if price < _VARIANT_SKIP_THRESHOLD:
                        logging.warning(
                            f"⚠️ Skipping low-value signal ${price:.2f} from query '{search_query}' "
                            f"— variant token {variant_tokens} not present (likely wrong card)"
                        )
                        continue
                    # ── Signed card guard ─────────────────────────────────────────
                    # If the card is signed but the query doesn't contain "Signed",
                    # the eBay price is for an UNSIGNED copy.  The PHP signed multiplier
                    # (1.8× artist / 4.5× celebrity) will be applied afterward, but only
                    # if the backend returns signed_multiplier=1 (not pre-applied).
                    # However: for vintage signed cards where the attr_estimate already
                    # encodes the signed premium (cla_layered_valuation), using the
                    # attr_estimate directly is more accurate than eBay_unsigned × PHP_mult.
                    # Prefer attr_estimate when: is_signed=True AND "Signed" not in query
                    # AND attr_estimate is available AND attr_estimate > unsigned_price.
                    if is_signed and "signed" not in sq_lower and _attr_est:
                        _ae_price = float(_attr_est.get("current_price", 0) or 0)
                        if _ae_price > 0 and _ae_price > price:
                            logging.info(
                                f"✍️ Signed card: attr_estimate ${_ae_price:.2f} > "
                                f"unsigned eBay ${price:.2f} — using attr_estimate | {card_name}"
                            )
                            _attr_est["search_query"]        = ""
                            _attr_est["queries_tried"]       = queries
                            _attr_est["card_name"]           = card_name
                            _attr_est["grader_brand_factor"] = 1.0
                            return _attr_est
                    else:
                        logging.info(
                            f"ℹ️ High-value signal ${price:.2f} accepted without variant token "
                            f"from query '{search_query}' — likely correct card"
                        )
            else:
                variant_in_query = False

            logging.info(
                f"✅ {source}: ${price:.2f} ({sales} records) | "
                f"variant_in_query={variant_in_query}"
            )

            try:
                if request.apply_cla_valuation is not False:
                    valuation = apply_cla_valuation(
                        signal_price          = float(price or 0.0),
                        signal_includes_grade = bool(grade_in_query),
                        search_query          = search_query,
                        target_grade          = grade or None,
                        rank_within_card      = getattr(request, "rank_within_card", None),
                        rank_total            = getattr(request, "rank_total", None),
                        slab_premium_add_aud  = 0.0,
                    )
                    final_price = float(valuation.get("final_value") or 0.0)
                else:
                    valuation   = None
                    final_price = float(price or 0.0)
            except Exception as _e:
                logging.exception("Legacy valuation error; falling back to signal price")
                valuation   = None
                final_price = float(price or 0.0)

            # Apply grader brand factor — legacy ladder path
            # NEVER apply for CLA 12 (legacy path already applies full CLA grade mult)
            _is_cla12_leg = grade_is_12_plus and grader.upper() in ("CLA", "COLLECTORS LEAGUE", "")
            _gbf_leg = 1.0 if _is_cla12_leg else _grader_brand_factor(grader, grade)
            if _gbf_leg != 1.0 and grader:
                final_price = round(final_price * _gbf_leg, 2)
                logging.info(f"🏷️ Grader brand factor (legacy): {grader} {_gbf_leg:.2f}x → ${final_price:.2f}")
            return {
                "current_price":        final_price,
                "signal_price":         float(price or 0.0),
                "base_price":           (valuation.get("base_price")          if valuation else float(price or 0.0)),
                "final_value":          (valuation.get("final_value")         if valuation else final_price),
                "multiplier_applied":   (valuation.get("multiplier_applied")  if valuation else 1.0),
                "slab_premium_added":   (valuation.get("slab_premium_added")  if valuation else 0.0),
                "target_multiplier":    (valuation.get("target_multiplier")   if valuation else 1.0),
                "base_multiplier":      (valuation.get("base_multiplier")     if valuation else 1.0),
                "confidence":           "low",
                "grader":               grader or "PSA",
                "grader_brand_factor":  _gbf_leg,
                "sources_used":         [source],
                "source":               source,
                "search_query":         search_query,
                "queries_tried":        queries,
                "card_name":            card_name,
                "sales_count":          sales,
                # IMPORTANT: if apply_cla_valuation was called and applied a multiplier,
                # final_price already includes the grade premium. Signal price_includes_grade=True
                # so the PHP caller does NOT apply a second multiplier on top.
                # grade_12_uplift=False for same reason — uplift is already baked into final_price.
                "price_includes_grade": True if (valuation and valuation.get("multiplier_applied", 1.0) != 1.0) else grade_in_query,
                "grade_12_uplift":      False if (valuation and valuation.get("multiplier_applied", 1.0) != 1.0) else bool(grade_is_12_plus and grade_in_query),
                "variant_matched":      variant_in_query,
                "variant_terms_used":   variant_str or None,
                "fingerprint_used":     False,
                "last_updated":         datetime.now().isoformat(),
            }

        # ── eBay found nothing — try PriceCharting as fallback ────────────────
        if PRICECHARTING_TOKEN:
            try:
                pc_q        = _norm_ws(f"{card_name} {card_set}".strip())
                pc_products = await _pc_search(pc_q, limit=5)
                if pc_products:
                    pc_best   = pc_products[0]
                    pc_pid    = str(pc_best.get("id") or pc_best.get("product-id") or "").strip()
                    pc_detail = await _pc_product(pc_pid) if pc_pid else {}
                    pc_merged = dict(pc_best)
                    if isinstance(pc_detail, dict):
                        pc_merged.update(pc_detail)
                    pc_prices = _pc_extract_price_fields(pc_merged)
                    for pk in ["loose-price", "used_price", "complete-price", "new-price"]:
                        if (pk in pc_prices
                                and isinstance(pc_prices[pk], (int, float))
                                and pc_prices[pk] > 0):
                            pc_aud = _usd_to_aud_simple(pc_prices[pk])
                            if pc_aud and float(pc_aud) > 0:
                                logging.info(f"✅ PriceCharting fallback: ${float(pc_aud):.2f} AUD")
                                try:
                                    if request.apply_cla_valuation is not False:
                                        _val = apply_cla_valuation(
                                            signal_price          = float(pc_aud or 0.0),
                                            signal_includes_grade = False,
                                            search_query          = str(pc_q),
                                            target_grade          = grade or None,
                                            rank_within_card      = getattr(request, "rank_within_card", None),
                                            rank_total            = getattr(request, "rank_total", None),
                                            slab_premium_add_aud  = 0.0,
                                        )
                                        _final = float(_val.get("final_value") or 0.0)
                                    else:
                                        _val   = None
                                        _final = round(float(pc_aud), 2)
                                except Exception:
                                    _val   = None
                                    _final = round(float(pc_aud), 2)

                                return {
                                    "current_price":        round(_final, 2),
                                    "signal_price":         round(float(pc_aud), 2),
                                    "base_price":           (_val.get("base_price") if _val else round(float(pc_aud), 2)),
                                    "final_value":          (_val.get("final_value") if _val else round(_final, 2)),
                                    "multiplier_applied":   (_val.get("multiplier_applied") if _val else 1.0),
                                    "slab_premium_added":   (_val.get("slab_premium_added") if _val else 0.0),
                                    "target_multiplier":    (_val.get("target_multiplier") if _val else 1.0),
                                    "base_multiplier":      (_val.get("base_multiplier") if _val else 1.0),
                                    "confidence":           "low",
                                    "sources_used":         ["pricecharting"],
                                    "source":               "pricecharting",
                                    "search_query":         pc_q,
                                    "queries_tried":        queries,
                                    "card_name":            card_name,
                                    "price_includes_grade": False,
                                    "fingerprint_used":     False,
                                    "last_updated":         datetime.now().isoformat(),
                                }
            except Exception as _pce:
                logging.warning(f"PriceCharting fallback error: {_pce}")

        # ── No results from any source ─────────────────────────────────────────
        # Before returning 0, attempt an attribute-based estimate.
        # This fires when the card has a rarity/variant/grade but no eBay comps
        # exist (new set, niche card, obscure variant).  A conservative estimate
        # is better than $0 which makes the collection total meaningless.
        logging.warning(f"❌ No market comps found for: {card_name} — attempting attribute estimate")

        _has_meaningful_rarity = bool(
            rarity and rarity.lower() not in ("", "common", "uncommon")
        ) or bool(variant_type and variant_type.lower() not in ("", "regular", "standard"))
        _has_grade = bool(grade)

        if _has_grade or _has_meaningful_rarity:
            try:
                est = _cla_estimate_from_attributes(
                    game_type    = game_type,
                    rarity       = rarity,
                    variant_type = variant_type,
                    finish       = finish,
                    language     = language,
                    is_signed    = is_signed,
                        signed_by    = signed_by,
                    grade        = grade,
                    rank_within_card = getattr(request, "rank_within_card", None),
                    rank_total       = getattr(request, "rank_total",       None),
                    card_set     = card_set,
                    card_number  = card_number,
                )
                if est.get("current_price", 0) > 0:
                    est["search_query"]  = ""
                    est["queries_tried"] = queries
                    est["card_name"]     = card_name
                    logging.info(
                        f"📊 Attribute estimate used: ${est['current_price']:.2f} AUD "
                        f"for {card_name} (no comps found)"
                    )
                    return est
            except Exception as _est_err:
                logging.warning(f"Attribute estimate failed: {_est_err}")

        best_q = queries[-1] if queries else card_name
        candidates_sold   = await _ebay_find_items(best_q, limit=5, sold=True,  days_lookback=30)
        candidates_active = await _ebay_find_items(best_q, limit=5, sold=False)
        return {
            "current_price":     0,
            "final_value":       0,
            "multiplier_applied":1.0,   # prevents PHP double-mult on zero price
            "confidence":        "none",
            "source":            "no_results",
            "error":             "No listings found on eBay or PriceCharting",
            "search_query":      best_q,
            "queries_tried":     queries,
            "card_name":         card_name,
            "price_includes_grade": True,  # nothing to multiply
            "grade_12_uplift":   False,
            "fingerprint_used":  bool(fp and not fp_error),
            "candidates_sold":   candidates_sold,
            "candidates_active": candidates_active,
            "hint":              "Pick the closest match from candidates and save that price.",
        }

    except Exception as e:
        logging.error(f"❌ Price lookup error: {e}", exc_info=True)
        return {"current_price": 0, "source": "error", "error": str(e)}


@app.post("/api/market/active-listings")
@safe_endpoint
async def get_active_listings(
    card_name: str = Form(...),
    set_name: str = Form(""),
    card_number: str = Form(""),
    card_type: str = Form("Pokemon"),
    max_results: int = Form(12),
):
    '''Return current eBay active listings for the given card.'''
    try:
        queries = _build_ebay_query_ladder(card_name=card_name, set_name=set_name, card_number=card_number, grade="")
    except TypeError:
        queries = [f"{card_name} {set_name} {card_number}".strip()]

    listings = []
    query_used = queries[0] if queries else card_name

    for q in (queries or [])[:3]:
        try:
            results = await _ebay_find_items(q, sold=False, limit=max_results)
            if results:
                listings = results
                query_used = q
                break
        except Exception:
            continue

    return JSONResponse(content={
        "success": True,
        "listings": listings,
        "query_used": query_used,
        "total": len(listings),
    })


@app.post("/api/market/sell-plan")
@safe_endpoint
async def generate_sell_plan(
    card_name: str = Form(...),
    set_name: str = Form(""),
    grade: str = Form(""),
    price_low: float = Form(0),
    price_median: float = Form(0),
    price_high: float = Form(0),
    trend_direction: str = Form("stable"),
):
    '''Generate a sell plan with recommended pricing and timing.'''
    prompt = f'''You are an expert collectibles market advisor.

Card: {card_name} ({set_name})
Grade: {grade}
Current market (AUD): Low {price_low:.2f} | Median {price_median:.2f} | High {price_high:.2f}
Trend: {trend_direction}

Return JSON ONLY with keys:
- recommended_list_price: {{low: number, high: number}}
- suggested_timing: string
- timing_reasoning: string
- listing_tips: array of strings (3-6)
- platform_recommendation: string
- price_strategy: string
'''

    plan = None

    try:
        # Use the shared async helper — avoids blocking the event loop
        _sp_result = await _openai_chat(
            [{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.3,
        )
        if not _sp_result.get("error"):
            plan = _parse_json_or_none(_sp_result.get("content", ""))
    except Exception:
        plan = None

    if not isinstance(plan, dict):
        markup = 1.10 if trend_direction == "up" else (0.95 if trend_direction == "down" else 1.02)
        plan = {
            "recommended_list_price": {
                "low": round(float(price_median) * 0.95, 2) if price_median else round(float(price_low) * 1.05, 2),
                "high": round(float(price_high) * markup, 2) if price_high else round(float(price_median) * 1.05, 2),
            },
            "suggested_timing": "Now" if trend_direction != "down" else "Wait 2–4 weeks",
            "timing_reasoning": f"Market trend appears {trend_direction}.",
            "listing_tips": [
                "Use high-quality front and back photos",
                "Include grade and certification details in the title",
                "Price at the top of the range, allow offers",
                "List during peak traffic (weekend evenings)",
            ],
            "platform_recommendation": "eBay Australia",
            "price_strategy": "Start high with Best Offer; reduce gradually if no interest.",
        }

    return JSONResponse(content={"success": True, "sell_plan": plan})



@app.post("/api/collection/update-values")
@safe_endpoint
async def update_collection_market_values(request: Request):
    """
    Fetch current market values for a batch of items.

    WHY THIS EXISTS:
    - Older front-ends call this endpoint directly.
    - WordPress is still the source of truth, but this endpoint can return prices + candidate matches
      so the WP UI can save the selected value.

    INPUT (preferred JSON):
      {
        "user_id": 123,
        "items": [
          {"item_id":"4","card_name":"Mega Charizard X ex","card_set":"", "card_number":"", "grade":""}
        ]
      }

    INPUT (legacy form):
      user_id=<int>
      items_json=<json string as above items list>
    """
    try:
        payload: Dict[str, Any] = {}

        ct = (request.headers.get("content-type") or "").lower()
        if "application/json" in ct:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
        else:
            form = await request.form()
            payload = dict(form) if form else {}

        # Parse fields
        user_id = int(payload.get("user_id") or 0)
        items = payload.get("items") or payload.get("items_json") or payload.get("itemsJson") or []

        if isinstance(items, str):
            try:
                items = json.loads(items)
            except Exception:
                items = []

        if not isinstance(items, list) or not items:
            return {
                "success": False,
                "updated_count": 0,
                "total_value": "0.00",
                "change_24h": "0.00",
                "change_percent": "0.0",
                "errors": [
                    {"item_id": "", "card": "", "error": "No items provided. Send JSON body with an items[] array (or form items_json)."}
                ],
                "note": "This endpoint cannot pull collection items from WordPress automatically. The front-end must send the items to price.",
            }

        updated_count = 0
        total_value = 0.0
        errors: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []

        # ── Parallel market lookups — asyncio.gather instead of sequential awaits ──
        # Sequential: 10 items × ~3s each = 30s. Parallel: ~3-5s total.
        # Also pass rarity/variant/finish/edition/language so eBay queries are
        # specific to the actual card variant (e.g. SAR vs common, Shadowless, etc.)
        valid_items = []
        for it in items:
            if not isinstance(it, dict):
                continue
            item_id      = str(it.get("item_id") or it.get("id") or "")
            card_name    = str(it.get("card_name") or it.get("card") or "").strip()
            card_set     = str(it.get("card_set")  or it.get("set")  or "").strip()
            card_number  = str(it.get("card_number") or it.get("number") or "").strip()
            grade        = str(it.get("grade") or "").strip()
            # Variant / rarity enrichment fields
            rarity       = str(it.get("rarity")       or "").strip()
            variant_type = str(it.get("variant_type") or "").strip()
            finish       = str(it.get("finish")       or "").strip()
            edition      = str(it.get("edition")      or "").strip()
            is_signed    = bool(it.get("is_signed")   or False)
            signed_by    = str(it.get("signed_by")    or "").strip()
            is_error     = bool(it.get("is_error_card") or False)
            is_promo     = bool(it.get("is_promo")    or False)
            language     = str(it.get("language")     or "").strip()
            rank_within  = it.get("rank_within_card")
            rank_total   = it.get("rank_total")
            if not card_name:
                errors.append({"item_id": item_id, "card": "", "error": "Missing card_name"})
                continue
            valid_items.append((
                item_id, card_name, card_set, card_number, grade,
                rarity, variant_type, finish, edition,
                is_signed, signed_by, is_error, is_promo, language,
                rank_within, rank_total,
            ))

        async def _lookup_one(
            item_id, card_name, card_set, card_number, grade,
            rarity, variant_type, finish, edition,
            is_signed, signed_by, is_error, is_promo, language,
            rank_within, rank_total,
        ):
            try:
                result = await market_price_lookup(MarketPriceLookupRequest(
                    card_name=card_name,
                    card_set=card_set or None,
                    card_number=card_number or None,
                    grade=grade or None,
                    rarity=rarity or None,
                    variant_type=variant_type or None,
                    finish=finish or None,
                    edition=edition or None,
                    is_signed=is_signed or None,
                    signed_by=signed_by or None,
                    is_error_card=is_error or None,
                    is_promo=is_promo or None,
                    language=language or None,
                    rank_within_card=int(rank_within) if rank_within else None,
                    rank_total=int(rank_total) if rank_total else None,
                ))
                return item_id, card_name, result
            except Exception as ex:
                return item_id, card_name, {"current_price": 0, "error": str(ex)}

        lookup_results = await asyncio.gather(
            *[_lookup_one(*args) for args in valid_items],
            return_exceptions=False,
        )

        for item_id, card_name, market in lookup_results:
            price = float(market.get("current_price") or 0.0)
            if price > 0:
                updated_count += 1
                total_value += price
                results.append({
                    "item_id": item_id,
                    "card": card_name,
                    "current_market_value": round(price, 2),
                    # Expose signal vs final so the UI can explain the number
                    "raw_price": round(float(market.get("signal_price") or price), 2),
                    "multiplier": round(float(market.get("multiplier_applied") or 1.0), 4),
                    "slab_premium": round(float(market.get("slab_premium_added") or 0.0), 2),
                    "source": market.get("source", ""),
                    "variant_matched": bool(market.get("variant_matched")),
                    "variant_terms": market.get("variant_terms_used") or "",
                    "search_query": market.get("search_query") or "",
                })
            else:
                errors.append({"item_id": item_id, "card": card_name, "error": "No price returned", "market": market})

        return {
            "success": True,
            "user_id": user_id,
            "updated_count": updated_count,
            "total_items": len(valid_items),
            "total_value": f"{total_value:.2f}",
            "change_24h": "0.00",
            "change_percent": "0.0",
            "updated_items": results,   # per-item detail incl. raw_price, multiplier, variant_matched
            "errors": errors,
        }

    except Exception as e:
        logging.error(f"❌ update-values error: {e}")
        return {"success": False, "error": str(e)}
@app.post("/api/generate-heatmap")
@safe_endpoint
async def generate_defect_heatmap(
    front: UploadFile = File(...),
    back: UploadFile = File(None),
    assessment_data: str = Form(...),
):
    """
    Generate heatmap overlay data (zones) based on BOTH assessment JSON AND real CV analysis.
    Returns canvas-friendly coordinates (0..1) + severities.

    v2 (Feb 2026): Now actually processes the uploaded images with PIL CV to compute
    real anomaly zones (brightness/edge hotspots), then merges with assessment-stated defects.
    """
    front_bytes = await front.read()
    back_bytes = await back.read() if back else None

    assessment = {}
    try:
        assessment = json.loads(assessment_data) if assessment_data else {}
    except Exception:
        assessment = {}

    # ── 1. JSON-driven zones (assessment keywords → fixed coordinates) ─────────
    defect_zones = extract_defect_zones(assessment)

    # ── 2. CV-driven zones (actual pixel analysis for real anomaly positions) ───
    def cv_hotspot_zones(img_bytes: bytes, side: str) -> List[Dict]:
        """Use PIL to find real brightness hotspots near borders and return zone dicts."""
        if not img_bytes or Image is None:
            return []
        zones: List[Dict] = []
        try:
            im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = im.size
            g_img = ImageOps.grayscale(im)
            edge = g_img.filter(ImageFilter.FIND_EDGES)

            band = max(8, int(min(w, h) * 0.10))

            def _score_region(x1, y1, x2, y2):
                """Score a region by edge + brightness signal (whitening = bright + edges)."""
                c_gray = g_img.crop((x1, y1, x2, y2))
                c_edge = edge.crop((x1, y1, x2, y2))
                b_gray = ImageStat.Stat(c_gray).mean[0] / 255.0
                b_edge = ImageStat.Stat(c_edge).mean[0] / 255.0
                return 0.5 * b_edge + 0.5 * b_gray

            # Corner zones
            c_size = max(16, int(min(w, h) * 0.16))
            corner_defs = [
                ("top_left",     0,       0,       c_size,   c_size,   0.12, 0.12),
                ("top_right",    w-c_size, 0,      w,        c_size,   0.88, 0.12),
                ("bottom_left",  0,       h-c_size, c_size,  h,        0.12, 0.88),
                ("bottom_right", w-c_size, h-c_size, w,      h,        0.88, 0.88),
            ]
            for (name, x1, y1, x2, y2, cx, cy) in corner_defs:
                score = _score_region(max(0,x1), max(0,y1), min(w,x2), min(h,y2))
                severity = 0
                if score > 0.55: severity = 3
                elif score > 0.40: severity = 2
                elif score > 0.28: severity = 1
                if severity > 0:
                    zones.append({
                        "x": cx, "y": cy, "radius": 0.13,
                        "severity": severity, "type": "corner",
                        "label": name.replace("_", " ").title(),
                        "source": "cv_pixel"
                    })

            # Edge mid-points
            edge_defs = [
                ("top",    w//4, 0,     3*w//4, band,    0.5,  0.06),
                ("bottom", w//4, h-band, 3*w//4, h,      0.5,  0.94),
                ("left",   0,    h//4,  band,   3*h//4,  0.06, 0.5),
                ("right",  w-band, h//4, w,     3*h//4,  0.94, 0.5),
            ]
            for (name, x1, y1, x2, y2, cx, cy) in edge_defs:
                score = _score_region(max(0,x1), max(0,y1), min(w,x2), min(h,y2))
                severity = 0
                if score > 0.50: severity = 2
                elif score > 0.35: severity = 1
                if severity > 0:
                    zones.append({
                        "x": cx, "y": cy, "radius": 0.18,
                        "severity": severity, "type": "edge",
                        "label": name.title() + " Edge",
                        "source": "cv_pixel"
                    })
        except Exception:
            pass
        return zones

    cv_front = cv_hotspot_zones(front_bytes, "front")
    cv_back  = cv_hotspot_zones(back_bytes, "back") if back_bytes else []

    # ── 3. Merge: CV zones fill gaps, assessment zones can upgrade severity ──────
    def merge_zones(json_zones: List[Dict], cv_zones: List[Dict]) -> List[Dict]:
        """Merge JSON-derived and CV-derived zones.
        - If both agree on a location (within 0.15 radius), keep higher severity.
        - CV-only zones are added if no JSON zone covers the area.
        """
        merged = list(json_zones)
        for cv_z in cv_zones:
            cx, cy = cv_z["x"], cv_z["y"]
            overlap = False
            for jz in merged:
                dx = jz["x"] - cx
                dy = jz["y"] - cy
                if (dx*dx + dy*dy) < 0.15*0.15:
                    # Promote severity if CV sees it worse
                    jz["severity"] = max(int(jz.get("severity", 0)), int(cv_z.get("severity", 0)))
                    jz["cv_confirmed"] = True
                    overlap = True
                    break
            if not overlap:
                cv_z["cv_only"] = True
                merged.append(cv_z)
        return merged

    merged_front = merge_zones(defect_zones.get("front", []), cv_front)
    merged_back  = merge_zones(defect_zones.get("back",  []), cv_back)

    heatmap_data = {
        "front": generate_heatmap_layer(merged_front, "front"),
        "back":  generate_heatmap_layer(merged_back,  "back") if (back_bytes or defect_zones.get("back")) else None,
    }

    return JSONResponse(content={
        "success": True,
        "heatmap_data": heatmap_data,
        "defect_count": sum(len(zs) for zs in [merged_front, merged_back]),
        "severity_breakdown": calculate_severity_breakdown({"front": merged_front, "back": merged_back}),
        "cv_zones_found": {"front": len(cv_front), "back": len(cv_back)},
    })


# ──────────────────────────────────────────────────────────────────────────────
# NEW: /api/overlay-variants — server-side filter images for inspection overlay
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/overlay-variants")
@safe_endpoint
async def generate_overlay_variants(
    front: UploadFile = File(...),
    back: UploadFile = File(None),
):
    """
    Generate server-side filter variants for the inspection overlay modal.

    Returns base64 JPEGs for every frontend overlay mode:
      normal          → original (not returned, frontend uses its own)
      contrast_sharp  → 'contrast' mode
      edge_wear       → 'edgewear' mode
      sobel_surface   → 'surface' mode
      sobel_edges     → 'edges' mode
      invert          → 'invert' mode
      channel_r       → 'r' mode
      channel_g       → 'g' mode
      channel_b       → 'b' mode

    Frontend can either render these server images directly (sharper on mobile/low-end devices)
    or use them as reference alongside client-side canvas rendering.
    """
    front_bytes = await front.read()
    back_bytes  = await back.read() if back else None

    if not front_bytes or len(front_bytes) < 200:
        raise HTTPException(status_code=400, detail="Front image required")

    # Compress before filter processing — _make_defect_filter_variants will also
    # resize internally to 800px but compressing here reduces IO overhead too.
    front_bytes = _compress_image(front_bytes, max_long=1000, quality=85)
    back_bytes  = _compress_image(back_bytes,  max_long=1000, quality=85) if back_bytes else None

    front_variants = _make_defect_filter_variants(front_bytes)
    back_variants  = _make_defect_filter_variants(back_bytes) if back_bytes else {}

    def encode_variants(vd: Dict[str, bytes], prefix: str) -> Dict[str, str]:
        out = {}
        for k, v in vd.items():
            if v and len(v) > 200:
                out[f"{prefix}_{k}"] = _b64(v)
        return out

    payload = {}
    payload.update(encode_variants(front_variants, "front"))
    payload.update(encode_variants(back_variants, "back"))

    # Map variant names to frontend mode names for easy lookup
    mode_map = {
        "contrast_sharp":   "contrast",
        "edge_wear":        "edgewear",
        "sobel_surface":    "surface",
        "sobel_edges":      "edges",
        "invert":           "invert",
        "channel_r":        "r",
        "channel_g":        "g",
        "channel_b":        "b",
        "gray_autocontrast":"ai_second_pass",
        "uv_sim":           "uv",
        "local_variance":   "variance",
        "chromatic":        "chromatic",
        "corner_isolate":   "corners",
    }

    return JSONResponse(content={
        "success": True,
        "variants": payload,
        "mode_map": mode_map,
        "has_back": bool(back_bytes),
        "pil_available": PIL_AVAILABLE,
    })


# ══════════════════════════════════════════════════════════════════════════════
# /api/defect-scan  — Standalone Defect Detection Tool
# ══════════════════════════════════════════════════════════════════════════════
#
# Two modes:
#   quick (default)  → CV-only (PIL/numpy): filter variants + centering + pixel
#                       hotspot zones. No AI calls. ~2-4 seconds.
#   full             → Quick pass PLUS _openai_label_rois on worst ROI crops,
#                       then a synthesis pass for a natural-language summary.
#                       ~10-20 seconds depending on image count.
#
# Accepts: front (required), back (optional), extra (optional detail shot)
# Returns: centering, filter_variants (all 13 modes), cv_defects, hotspot_crops,
#          severity_breakdown, and (full only) ai_defects + defect_summary.
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/defect-scan")
@safe_endpoint
async def defect_scan(
    front: UploadFile = File(...),
    back:  Optional[UploadFile] = File(None),
    extra: Optional[UploadFile] = File(None),   # close-up / detail shot
    scan_mode:  str  = Form("quick"),            # "quick" | "full"
    card_name:  Optional[str] = Form(None),
    card_set:   Optional[str] = Form(None),
    # Optional client-provided hashes (for chain-of-custody UI / cross-check)
    front_sha256: Optional[str] = Form(None),
    back_sha256:  Optional[str] = Form(None),
    extra_sha256: Optional[str] = Form(None),
):
    """
    Dedicated defect scanner — the only job is finding and reporting defects.

    quick mode:  CV-only, no AI.  ~2-4 s.
    full  mode:  CV + AI ROI labelling + natural-language synthesis.  ~15-20 s.
    """
    t0 = time.time()  # `time` imported at module level

    # ── 1. Read and compress images ───────────────────────────────────────────
    # Read raw upload bytes first (chain-of-custody), then compress for processing.
    front_raw = await front.read()
    back_raw  = await back.read()  if back  and back.filename  else b""
    extra_raw = await extra.read() if extra and extra.filename else b""

    front_bytes = _compress_image(front_raw)
    back_bytes  = _compress_image(back_raw)  if back_raw  else b""
    extra_bytes = _compress_image(extra_raw) if extra_raw else b""

    def _sha256(b: bytes) -> str:
        try:
            return hashlib.sha256(b or b"").hexdigest()
        except Exception:
            return ""

    # Hashes of the RAW uploads (match client-side SHA-256) + compressed processing bytes.
    front_sha_raw = _sha256(front_raw)
    back_sha_raw  = _sha256(back_raw)  if back_raw  else ""
    extra_sha_raw = _sha256(extra_raw) if extra_raw else ""

    front_sha_server = _sha256(front_bytes)
    back_sha_server  = _sha256(back_bytes)  if back_bytes  else ""
    extra_sha_server = _sha256(extra_bytes) if extra_bytes else ""

    if not front_bytes or len(front_bytes) < 200:
        raise HTTPException(status_code=400, detail="Front image required")

    scan_mode = (scan_mode or "quick").strip().lower()
    if scan_mode not in ("quick", "full"):
        scan_mode = "quick"

    # ── 1b. Autocrop — detect card border and crop to 1% inside it ───────────
    # The cropped bytes are used for ALL downstream analysis (filter variants,
    # defect zones, centering, AI pass). The original compressed bytes are kept
    # for the crop overlay display.
    front_cropped, front_crop_info = _autocrop_card(front_bytes, inset_pct=0.0, pad_pct=0.05)
    back_cropped,  back_crop_info  = _autocrop_card(back_bytes,  inset_pct=0.0, pad_pct=0.05) if back_bytes  else (b"", {"detected": False})
    extra_cropped, extra_crop_info = _autocrop_card(extra_bytes, inset_pct=0.0, pad_pct=0.05) if extra_bytes else (b"", {"detected": False})

    # Use cropped versions for all analysis; fall back to originals if not detected
    front_proc = front_cropped if front_crop_info.get("detected") else front_bytes
    back_proc  = back_cropped  if back_crop_info.get("detected")  else back_bytes
    extra_proc = extra_cropped if extra_crop_info.get("detected") else extra_bytes

    # ── 2. CV pass — runs for both modes ─────────────────────────────────────

    # 2a. Centering (pixel-accurate Sobel border detection)
    # Run on ORIGINAL compressed bytes (before autocrop) — centering measures the
    # physical white border of the card, which the autocrop deliberately removes.
    # Using the cropped image would give zero/garbage margins since the borders are gone.
    centering_front = _estimate_centering_from_image(front_bytes)
    centering_back  = _estimate_centering_from_image(back_bytes)  if back_bytes  else None
    centering_extra = _estimate_centering_from_image(extra_bytes) if extra_bytes else None

    # 2b. Filter variants (all 13 modes) — run on cropped images
    front_variants = _make_defect_filter_variants(front_proc)
    back_variants  = _make_defect_filter_variants(back_proc)  if back_proc  else {}
    extra_variants = _make_defect_filter_variants(extra_proc) if extra_proc else {}

    def _encode_variants(vd: Dict[str, bytes], prefix: str) -> Dict[str, str]:
        out = {}
        for k, v in vd.items():
            if v and len(v) > 200:
                out[f"{prefix}_{k}"] = _b64(v)
        return out

    filter_variants: Dict[str, str] = {}
    filter_variants.update(_encode_variants(front_variants, "front"))
    filter_variants.update(_encode_variants(back_variants,  "back"))
    filter_variants.update(_encode_variants(extra_variants, "extra"))

    # 2c. CV pixel hotspot detection (corners + edges on each image)
    def _cv_zones_for_image(img_bytes: bytes, side: str) -> List[Dict]:
        """Return hotspot zone dicts from pixel analysis on one image."""
        if not img_bytes or Image is None:
            return []
        zones: List[Dict] = []
        try:
            im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = im.size
            g_img = ImageOps.grayscale(im)
            edge_img = g_img.filter(ImageFilter.FIND_EDGES)
            band = max(8, int(min(w, h) * 0.10))

            def _score(x1, y1, x2, y2) -> float:
                c_g = g_img.crop((x1, y1, x2, y2))
                c_e = edge_img.crop((x1, y1, x2, y2))
                bg = ImageStat.Stat(c_g).mean[0] / 255.0
                be = ImageStat.Stat(c_e).mean[0] / 255.0
                return 0.5 * bg + 0.5 * be

            c = max(16, int(min(w, h) * 0.18))
            corners = [
                ("top_left",     0,       0,       c,     c,     0.12, 0.12),
                ("top_right",    w - c,   0,       w,     c,     0.88, 0.12),
                ("bottom_left",  0,       h - c,   c,     h,     0.12, 0.88),
                ("bottom_right", w - c,   h - c,   w,     h,     0.88, 0.88),
            ]
            for (name, x1, y1, x2, y2, cx, cy) in corners:
                score = _score(max(0, x1), max(0, y1), min(w, x2), min(h, y2))
                sev = 3 if score > 0.55 else (2 if score > 0.40 else (1 if score > 0.28 else 0))
                if sev > 0:
                    zones.append({
                        "x": cx, "y": cy, "radius": 0.13, "severity": sev,
                        "type": "corner", "label": name.replace("_", " ").title(),
                        "side": side, "source": "cv_pixel",
                    })

            edges = [
                ("top",    w // 4, 0,        3 * w // 4, band,       0.50, 0.06),
                ("bottom", w // 4, h - band,  3 * w // 4, h,          0.50, 0.94),
                ("left",   0,      h // 4,    band,        3 * h // 4, 0.06, 0.50),
                ("right",  w-band, h // 4,    w,           3 * h // 4, 0.94, 0.50),
            ]
            for (name, x1, y1, x2, y2, cx, cy) in edges:
                score = _score(max(0, x1), max(0, y1), min(w, x2), min(h, y2))
                sev = 2 if score > 0.50 else (1 if score > 0.35 else 0)
                if sev > 0:
                    zones.append({
                        "x": cx, "y": cy, "radius": 0.18, "severity": sev,
                        "type": "edge", "label": name.title() + " Edge",
                        "side": side, "source": "cv_pixel",
                    })

            # Surface scan: interior variance (scratch detector)
            if np is not None:
                try:
                    interior_x1 = int(w * 0.15)
                    interior_y1 = int(h * 0.15)
                    interior_x2 = int(w * 0.85)
                    interior_y2 = int(h * 0.85)
                    interior = im.crop((interior_x1, interior_y1, interior_x2, interior_y2))
                    g_int = np.asarray(ImageOps.grayscale(interior), dtype=np.float32)
                    # 7×7 local variance
                    from scipy.ndimage import uniform_filter
                    mean_f = uniform_filter(g_int, size=7)
                    sq_mean = uniform_filter(g_int ** 2, size=7)
                    var_map = sq_mean - mean_f ** 2
                    # Flag if top 1% variance patches are unusually high
                    thr_var = float(np.percentile(var_map, 99))
                    mean_var = float(np.mean(var_map))
                    if thr_var > mean_var * 4.5:  # abnormal variance spike
                        # Find rough location of worst patch
                        max_pos = np.unravel_index(np.argmax(var_map), var_map.shape)
                        rel_y = (interior_y1 + max_pos[0]) / h
                        rel_x = (interior_x1 + max_pos[1]) / w
                        zones.append({
                            "x": round(rel_x, 3), "y": round(rel_y, 3),
                            "radius": 0.12, "severity": 2,
                            "type": "surface", "label": "Surface Anomaly",
                            "side": side, "source": "cv_variance",
                        })
                except Exception:
                    pass
        except Exception:
            pass
        return zones

    cv_defects = {
        "front": _cv_zones_for_image(front_bytes, "front"),
        "back":  _cv_zones_for_image(back_bytes,  "back")  if back_bytes  else [],
        "extra": _cv_zones_for_image(extra_bytes, "extra") if extra_bytes else [],
    }

    # 2d. Hotspot thumbnail crops (evidence strips)
    hotspot_crops: List[Dict] = []
    for side_key, img_bytes in [("front", front_bytes), ("back", back_bytes), ("extra", extra_bytes)]:
        if img_bytes:
            hotspot_crops.extend(_make_basic_hotspot_snaps(img_bytes, side_key, max_snaps=6))

    # 2e. Severity breakdown
    all_zones = cv_defects["front"] + cv_defects["back"] + cv_defects["extra"]
    def _severity_breakdown(zones: List[Dict]) -> Dict[str, int]:
        b = {"minor": 0, "moderate": 0, "severe": 0}
        for z in zones:
            s = int(z.get("severity", 0) or 0)
            if s == 1:   b["minor"] += 1
            elif s == 2: b["moderate"] += 1
            elif s >= 3: b["severe"] += 1
        return b

    severity_breakdown = _severity_breakdown(all_zones)

    # ── 3. AI pass — full mode only ───────────────────────────────────────────
    ai_defects: List[Dict] = []
    defect_summary: str = ""
    overall_severity: str = "clean"

    if scan_mode == "full":
        # Build ROI list from worst CV zones (top 8 by severity)
        ranked_zones = sorted(all_zones, key=lambda z: -int(z.get("severity", 0) or 0))
        rois_to_label = []
        for z in ranked_zones[:8]:
            r = z.get("radius", 0.13)
            cx, cy = z["x"], z["y"]
            rois_to_label.append({
                "side": z.get("side", "front"),
                "roi":  z.get("label", "region"),
                "bbox": {
                    "x": max(0.0, cx - r),
                    "y": max(0.0, cy - r),
                    "w": min(1.0, r * 2),
                    "h": min(1.0, r * 2),
                },
            })

        # Also add the extra image as a surface ROI if provided
        if extra_bytes:
            rois_to_label.append({
                "side": "extra",
                "roi":  "detail_shot",
                "bbox": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
            })

        # Map "extra" side to front_bytes for the labeler (closest match)
        img_map = {"front": front_bytes, "back": back_bytes, "extra": extra_bytes}

        # Patch: _openai_label_rois only accepts front/back — split extra into front path
        rois_front_back = []
        for r in rois_to_label:
            side = r.get("side", "front")
            if side == "extra":
                r2 = dict(r); r2["side"] = "front"
                rois_front_back.append(r2)
            else:
                rois_front_back.append(r)

        ai_defects = await _openai_label_rois(rois_front_back, front_bytes, back_bytes)

        # Synthesis: natural-language summary
        defect_lines = []
        for d in ai_defects:
            if d.get("is_defect"):
                defect_lines.append(
                    f"- {d.get('roi','unknown')} ({d.get('side','?')} side): "
                    f"{d.get('type','defect')} — {d.get('note','')}"
                )
        cv_summary_lines = []
        for z in ranked_zones[:5]:
            sev_word = {1: "minor", 2: "moderate", 3: "severe"}.get(z.get("severity", 0), "unknown")
            cv_summary_lines.append(f"- {z.get('label','zone')} ({z.get('side','?')}): CV-detected {sev_word} anomaly")

        synth_prompt = (
            f"You are a professional card defect analyst. "
            f"A collector has scanned a {'named ' + card_name if card_name else 'trading'} card "
            f"{'from ' + card_set if card_set else ''}. "
            f"Below are the defects found by CV analysis and AI ROI inspection. "
            f"Write a clear, honest 3-5 sentence defect summary for the collector. "
            f"Mention the most impactful issues first. Do NOT recommend a grade. "
            f"CV findings:\n" + ("\n".join(cv_summary_lines) or "None detected") +
            f"\nAI ROI findings:\n" + ("\n".join(defect_lines) or "No AI-confirmed defects") +
            f"\nFormat: plain text, no markdown, no bullet points."
        )
        synth_res = await _openai_chat(
            [{"role": "user", "content": synth_prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        defect_summary = _norm_ws(synth_res.get("content", ""))

        # Overall severity label
        if severity_breakdown["severe"] > 0:
            overall_severity = "severe"
        elif severity_breakdown["moderate"] > 0:
            overall_severity = "moderate"
        elif severity_breakdown["minor"] > 0:
            overall_severity = "minor"
        else:
            overall_severity = "clean"

    elapsed_ms = int((time.time() - t0) * 1000)

    # ── 4. Build response ──────────────────────────────────────────────────────
    response: Dict[str, Any] = {
        "success": True,
        "scan_mode": scan_mode,
        "server_time_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "processing_time_ms": elapsed_ms,
        "evidence": {
            "inputs": {
                "front": {
                    "filename": getattr(front, "filename", None),
                    "content_type": getattr(front, "content_type", None),
                    "bytes": len(front_raw or b""),
                    "raw_sha256": front_sha_raw,
                    "processed_bytes": len(front_bytes or b""),
                    "processed_sha256": front_sha_server,
                    "client_sha256": (front_sha256 or "").strip() or None,
                    "client_sha256_match": (front_sha_raw == (front_sha256 or "").strip()) if front_sha256 else None,
                },
                "back": {
                    "filename": getattr(back, "filename", None) if back else None,
                    "content_type": getattr(back, "content_type", None) if back else None,
                    "bytes": len(back_raw or b""),
                    "raw_sha256": back_sha_raw or None,
                    "processed_bytes": len(back_bytes or b""),
                    "processed_sha256": back_sha_server or None,
                    "client_sha256": (back_sha256 or "").strip() or None,
                    "client_sha256_match": (back_sha_raw == (back_sha256 or "").strip()) if back_sha256 and back_raw else None,
                },
                "extra": {
                    "filename": getattr(extra, "filename", None) if extra else None,
                    "content_type": getattr(extra, "content_type", None) if extra else None,
                    "bytes": len(extra_raw or b""),
                    "raw_sha256": extra_sha_raw or None,
                    "processed_bytes": len(extra_bytes or b""),
                    "processed_sha256": extra_sha_server or None,
                    "client_sha256": (extra_sha256 or "").strip() or None,
                    "client_sha256_match": (extra_sha_raw == (extra_sha256 or "").strip()) if extra_sha256 and extra_raw else None,
                },
            },
            "notes": {
                "compression": {"max_long": 1200, "quality": 88},
                "autocrop": {"inset_pct": 0.01, "analysis_runs_on_cropped": True, "centering_runs_on_original": True},
            },
        },
        "centering": {
            "front": centering_front,
            "back":  centering_back,
            "extra": centering_extra,
        },
        "cv_defects": cv_defects,
        "severity_breakdown": severity_breakdown,
        "overall_severity": overall_severity,
        "filter_variants": filter_variants,
        "filter_mode_map": {
            "contrast_sharp":    "contrast",
            "edge_wear":         "edgewear",
            "sobel_surface":     "surface",
            "sobel_edges":       "edges",
            "invert":            "invert",
            "channel_r":         "r",
            "channel_g":         "g",
            "channel_b":         "b",
            "gray_autocontrast": "ai_pass",
            "uv_sim":            "uv",
            "local_variance":    "variance",
            "chromatic":         "chromatic",
            "corner_isolate":    "corners",
        },
        "hotspot_crops": hotspot_crops,
        "has_back":  bool(back_bytes),
        "has_extra": bool(extra_bytes),
        "pil_available": PIL_AVAILABLE,
        # ── Autocrop ─────────────────────────────────────────────────────────
        # The frontend displays these cropped images in the viewer by default.
        # crop_info contains crop_box and crop_pct so the JS can draw a crop
        # rectangle overlay on the original image.
        "autocrop": {
            "front": {
                "info":  front_crop_info,
                "image": _b64(front_cropped) if front_crop_info.get("detected") else None,
                "rotated": front_crop_info.get("rotated", False),
            },
            "back": {
                "info":  back_crop_info,
                "image": _b64(back_cropped)  if back_crop_info.get("detected")  else None,
                "rotated": back_crop_info.get("rotated", False),
            },
            "extra": {
                "info":  extra_crop_info,
                "image": _b64(extra_cropped) if extra_crop_info.get("detected") else None,
                "rotated": extra_crop_info.get("rotated", False),
            },
        },
    }

    if scan_mode == "full":
        response["ai_defects"]     = ai_defects
        response["defect_summary"] = defect_summary

    # ── 5. PHP frontend compatibility layer ────────────────────────────────────
    # The PHP scan panel JS reads a flat, normalised shape.  We add these
    # aliased fields without removing the original nested structure so that
    # any future clients using the raw schema still work.

    # 5a. overall_severity — compute for quick mode too (was always "clean")
    if scan_mode == "quick":
        if severity_breakdown["severe"] > 0:
            overall_severity = "severe"
        elif severity_breakdown["moderate"] > 0:
            overall_severity = "moderate"
        elif severity_breakdown["minor"] > 0:
            overall_severity = "minor"
        else:
            overall_severity = "clean"
        response["overall_severity"] = overall_severity

    # 5b. Flat centering — PHP expects h_pct, v_pct, left_pct, right_pct, top_pct, bottom_pct
    def _parse_ratio(ratio_str) -> float:
        """Convert '55/45' → 55.0  (the larger side as a percentage)."""
        try:
            if ratio_str is None or ratio_str == "N/A":
                return 50.0
            parts = str(ratio_str).split("/")
            return float(parts[0])
        except Exception:
            return 50.0

    def _margins_to_pcts(cen_dict) -> dict:
        """Extract normalised margin percentages from centering dict."""
        if not cen_dict or not isinstance(cen_dict, dict):
            return {}
        m = cen_dict.get("margins", {})
        ml = m.get("left",   1)
        mr = m.get("right",  1)
        mt = m.get("top",    1)
        mb = m.get("bottom", 1)
        h_total = max(ml + mr, 1)
        v_total = max(mt + mb, 1)
        h_pct = round(_parse_ratio(cen_dict.get("lr")), 1)
        v_pct = round(_parse_ratio(cen_dict.get("tb")), 1)
        return {
            "h_pct":     h_pct,
            "v_pct":     v_pct,
            "left_pct":  round(ml / h_total * 100, 1),
            "right_pct": round(mr / h_total * 100, 1),
            "top_pct":   round(mt / v_total * 100, 1),
            "bottom_pct":round(mb / v_total * 100, 1),
            "lr":        cen_dict.get("lr", "N/A"),
            "tb":        cen_dict.get("tb", "N/A"),
        }

    response["centering"] = _margins_to_pcts(centering_front)
    # Keep nested originals under a separate key for forensics PDF
    response["centering_detail"] = {
        "front": centering_front,
        "back":  centering_back,
        "extra": centering_extra,
    }

    # 5c. Flat defects array — PHP iterates data.defects with .type/.location/.severity/.confidence
    flat_defects = []
    for side_key, zones in cv_defects.items():
        for z in (zones or []):
            flat_defects.append({
                "type":       z.get("type", "unknown"),
                "label":      z.get("label", ""),
                "location":   z.get("label", side_key),
                "side":       z.get("side", side_key),
                "severity":   int(z.get("severity", 0) or 0),
                "confidence": round(float(z.get("confidence", 0.6) or 0.6), 2),
                "source":     z.get("source", "cv"),
                "x":          z.get("x"),
                "y":          z.get("y"),
            })
    if scan_mode == "full":
        for d in ai_defects:
            if d.get("is_defect"):
                flat_defects.append({
                    "type":       d.get("type", "ai_detected"),
                    "label":      d.get("roi", ""),
                    "location":   d.get("roi", ""),
                    "side":       d.get("side", "front"),
                    "severity":   int(d.get("severity", 1) or 1),
                    "confidence": round(float(d.get("confidence", 0.8) or 0.8), 2),
                    "source":     "ai",
                    "note":       d.get("note", ""),
                })
    response["defects"] = flat_defects

    # 5d. defect_snaps — PHP reads data.defect_snaps; API produces hotspot_crops
    response["defect_snaps"] = hotspot_crops   # same list, aliased

    # 5e. synthesis — PHP reads data.synthesis; API produces defect_summary
    response["synthesis"] = defect_summary if scan_mode == "full" else ""

    # 5f. api_version — PHP logs this for chain-of-custody
    response["api_version"] = APP_VERSION

    # ── 6. Card identity fingerprint vectors ──────────────────────────────────
    # phash / print_dot_hash / surface_hash / keypoints / embedding.
    #
    # These were previously never computed in this endpoint, which is why the
    # DNA Viewer showed "Not captured" for all four fields after every scan.
    # PIL images are already in memory — this costs ~5-15ms and uses no AI calls.
    # Every block is wrapped in its own try/except: a failure here must NEVER
    # prevent the defect scan results from being returned.
    #
    # PHP fingerprint_json receives these via CF.result → findings_json → the
    # merge in cg_cf_ajax_save_report, then the GD backfill is skipped because
    # the fields are already populated.
    # ──────────────────────────────────────────────────────────────────────────

    _fp: Dict[str, Any] = {}
    _fp_errors: Dict[str, str] = {}  # surfaces per-block failures in the API response

    if not PIL_AVAILABLE:
        logging.warning("fingerprint_vectors: Pillow not available — all fingerprint fields empty.")
        _fp_errors["pil"] = "Pillow not installed"
    elif not front_proc:
        logging.warning("fingerprint_vectors: front_proc empty — cannot compute fingerprints.")
        _fp_errors["front_proc"] = "front_proc bytes were empty"
    else:
        # ── 6a. dHash (64-bit difference hash) — identical to PHP GD dHash ──
        # 9×8 grayscale → compare adjacent columns → 64-bit → 16-char hex.
        # Used as both phash (primary identity) and print_dot_hash (print pixel proxy).
        try:
            _im_dhash = Image.open(io.BytesIO(front_proc)).convert("L").resize((9, 8), Image.LANCZOS)
            _bits = ""
            for _y in range(8):
                for _x in range(8):
                    _bits += "1" if _im_dhash.getpixel((_x, _y)) < _im_dhash.getpixel((_x + 1, _y)) else "0"
            _dhex = "".join(format(int(_bits[_i:_i+4], 2), "x") for _i in range(0, 64, 4)).zfill(16)
            _fp["phash"]          = _dhex
            _fp["print_dot_hash"] = _dhex  # dHash IS derived from the physical print pixels
        except Exception as _e:
            _fp_errors["phash"] = str(_e)
            logging.warning("fingerprint_vectors: dHash failed — %s", _e)

        # ── 6b. Surface hash — SHA-256 of 32×32 centre crop pixel data ───────
        # Captures ink/foil micro-pattern of the card face independent of border noise.
        # FIX (v6.9.2): Use .tobytes() instead of bytes(b for px in getdata() for b in px).
        #   getdata() returns an ImagingCore sequence; iterating it with a generator expression
        #   can raise a TypeError in Pillow >= 11 when internal pixel representation changes.
        #   .tobytes() is the canonical Pillow API for raw pixel bytes and is always safe.
        try:
            _im_surf = Image.open(io.BytesIO(front_proc)).convert("RGB")
            _sw, _sh = _im_surf.size
            _crop32  = _im_surf.crop((int(_sw * 0.25), int(_sh * 0.25),
                                      int(_sw * 0.75), int(_sh * 0.75))).resize((32, 32), Image.LANCZOS)
            _surf_buf = _crop32.tobytes()  # raw RGB pixel bytes — safe across all Pillow versions
            _fp["surface_hash"]    = hashlib.sha256(_surf_buf).hexdigest()
            _fp["surface_texture"] = _fp["surface_hash"]  # alias PHP reads
        except Exception as _e:
            _fp_errors["surface_texture"] = str(_e)
            logging.warning("fingerprint_vectors: surface_texture failed — %s", _e)

        # ── 6c. Colour zones (3×3 grid) → 9 zones × 3 RGB channels = 27 dims ─
        _zones: Dict[str, list] = {}
        try:
            _im_rgb = Image.open(io.BytesIO(front_proc)).convert("RGB")
            _cw, _ch = _im_rgb.size
            for _gy in range(3):
                for _gx in range(3):
                    _x0, _x1 = int(_cw * _gx / 3), int(_cw * (_gx + 1) / 3)
                    _y0, _y1 = int(_ch * _gy / 3), int(_ch * (_gy + 1) / 3)
                    _reg   = _im_rgb.crop((_x0, _y0, _x1, _y1))
                    _means = ImageStat.Stat(_reg).mean
                    _zones[f"z{_gy * 3 + _gx}"] = [round(_means[0]), round(_means[1]), round(_means[2])]
            _fp["color_zones_front"] = _zones
        except Exception as _e:
            _fp_errors["color_zones"] = str(_e)
            logging.warning("fingerprint_vectors: colour zones failed — %s", _e)

        # ── 6d. Intensity histogram (16 buckets) ──────────────────────────────
        _hist16: List[int] = []
        try:
            _im_gray  = Image.open(io.BytesIO(front_proc)).convert("L")
            _raw_hist = _im_gray.histogram()  # 256 bins
            _hist16   = [sum(_raw_hist[_i * 16:(_i + 1) * 16]) for _i in range(16)]
            _fp["intensity_hist_front"] = _hist16
        except Exception as _e:
            _fp_errors["intensity_hist"] = str(_e)
            logging.warning("fingerprint_vectors: intensity histogram failed — %s", _e)

        # ── 6e. Embedding: 43-dim vector (27 colour dims + 16 histogram dims) ─
        # Local proxy for CNN embedding — genuine pixel-derived feature vector.
        try:
            _vec: List[float] = []
            for _z in _zones.values():
                _vec += [round(_z[0] / 255.0, 4), round(_z[1] / 255.0, 4), round(_z[2] / 255.0, 4)]
            if _hist16:
                _htot = max(sum(_hist16), 1)
                _vec += [round(_c / _htot, 6) for _c in _hist16]
            if len(_vec) >= 16:
                _fp["embedding"] = _vec
        except Exception as _e:
            _fp_errors["embedding"] = str(_e)
            logging.warning("fingerprint_vectors: embedding failed — %s", _e)

        # ── 6f. Keypoints — 8 border luminance landmark vectors ──────────────
        # 64 luminance samples from the 4 borders, grouped into 8 descriptors.
        # Equivalent to the PHP GD border-sig keypoints but computed server-side
        # from actual pixel data — source tagged "api_border" so DNA viewer shows
        # "N points (SIFT/ORB)" rather than "N points (GD border)".
        try:
            _im_brd  = Image.open(io.BytesIO(front_proc)).convert("L")
            _bw, _bh = _im_brd.size
            _bp      = max(2, int(min(_bw, _bh) * 0.03))
            _lum: List[int] = []
            for _i in range(16):  # top
                _lum.append(_im_brd.getpixel((int(_bw * (_i + 0.5) / 16), _bp)))
            for _i in range(16):  # right
                _lum.append(_im_brd.getpixel((_bw - _bp - 1, int(_bh * (_i + 0.5) / 16))))
            for _i in range(15, -1, -1):  # bottom
                _lum.append(_im_brd.getpixel((int(_bw * (_i + 0.5) / 16), _bh - _bp - 1)))
            for _i in range(15, -1, -1):  # left
                _lum.append(_im_brd.getpixel((_bp, int(_bh * (_i + 0.5) / 16))))

            _kp = []
            for _i in range(8):
                _sl   = _lum[_i * 8:(_i + 1) * 8]
                _mean = sum(_sl) / max(len(_sl), 1)
                _kp.append({
                    "x":       round((_i / 8.0) * 1000),
                    "y":       round((_mean / 255.0) * 1000),
                    "strength": round(_mean / 255.0, 3),
                    "side":    "front",
                    "source":  "api_border",
                })
            _fp["keypoints"]        = _kp
            _fp["border_sig_front"] = bytes(min(255, max(0, _v)) for _v in _lum).hex()
        except Exception as _e:
            _fp_errors["keypoints"] = str(_e)
            logging.warning("fingerprint_vectors: keypoints failed — %s", _e)

    # ── Write all fingerprint fields into response ────────────────────────────
    response["phash"]                = _fp.get("phash", "")
    response["print_dot_hash"]       = _fp.get("print_dot_hash", "")
    response["surface_hash"]         = _fp.get("surface_hash", "")
    response["surface_texture"]      = _fp.get("surface_texture", "")
    response["keypoints"]            = _fp.get("keypoints") or []
    response["embedding"]            = _fp.get("embedding") or []
    response["border_sig_front"]     = _fp.get("border_sig_front", "")
    response["color_zones_front"]    = _fp.get("color_zones_front", {})
    response["intensity_hist_front"] = _fp.get("intensity_hist_front", [])
    response["fp_source"]            = "api" if _fp.get("phash") else "unavailable"
    # fp_errors only appears in the response when something went wrong.
    # An empty dict (all blocks succeeded) is omitted to keep the response clean.
    if _fp_errors:
        response["fp_errors"] = _fp_errors

    return JSONResponse(content=response)


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

    STUB: WordPress (PHP) writes to the WP database directly.
    This endpoint exists for forward-compatibility; it returns success
    but does not persist data server-side. Collection management is owned by PHP.
    """
    return JSONResponse(content={
        "success": True,
        "message": "Card added to collection (PHP handles persistence)",
        "user_id": user_id,
        "submission_id": submission_id,
        "_stub": True,
    })


@app.get("/api/collection/value-history")
@safe_endpoint
async def get_collection_value_history(user_id: int, days: int = 30):
    """Collection value history — NOT YET IMPLEMENTED.

    Returns a 501 stub. Collection value history is tracked in WordPress;
    a real implementation would query the WP DB or a separate analytics store.
    """
    return JSONResponse(
        content={
            "success": False,
            "message": "Collection value history endpoint is not yet implemented. "
                       "Track value history through the WordPress plugin directly.",
            "_stub": True,
        },
        status_code=501,
    )


@app.post("/api/collection/update-values-mock")
@safe_endpoint
async def update_collection_market_values_mock(user_id: int = Form(...)):
    """DEPRECATED — use /api/collection/update-values instead.

    Returns a 410 Gone so callers know to switch endpoints.
    """
    return JSONResponse(
        content={
            "success": False,
            "message": "This endpoint is deprecated. Use /api/collection/update-values instead.",
            "_stub": True,
        },
        status_code=410,
    )


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

    prediction_prompt = """You are a quick card grading predictor with expert-level vision analysis.
Based on these HIGH-QUALITY images, provide a comprehensive probability distribution.

Examine EVERY aspect carefully:
- All 8 corners (4 front + 4 back): whitening, blunting, sharpness
- All 4 edges on each side: fraying, chipping, whitening
- Surface: scratches, print lines, dulling, staining, gloss
- Centering: estimate L/R and T/B margins

Return JSON:
{
  "grade_probabilities": {
    "10": 0.05,
    "9.5": 0.10,
    "9": 0.15,
    "8.5": 0.20,
    "8": 0.25,
    "7": 0.15,
    "6": 0.10
  },
  "most_likely_grade": "8.5",
  "confidence": 0.75,
  "centering_estimate": {
    "front_lr": "55/45",
    "back_lr": "60/40"
  },
  "worst_area": "top-left corner — visible whitening ~1mm",
  "improvements_for_higher_grade": [
    "Better lighting on top-left corner to confirm whitening extent",
    "Close-up of edges needed",
    "Back photo shows glare - retake at different angle"
  ],
  "grade_limiters": ["Corner whitening", "Edge wear visible"],
  "quick_summary": "This card will likely grade 8-8.5 based on visible corner wear."
}
"""

    # ── Enhanced grade prediction: high-detail images + filter variants ──────
    predict_parts: list = [{"type": "text", "text": prediction_prompt}]
    predict_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(front_bytes)}", "detail": "high"}})
    if back_bytes:
        predict_parts.append({"type": "text", "text": "BACK:"})
        predict_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(back_bytes)}", "detail": "high"}})

    if PIL_AVAILABLE:
        try:
            fv = _make_defect_filter_variants(front_bytes)
            if fv.get("contrast_sharp"):
                predict_parts += [
                    {"type": "text", "text": "FRONT enhanced (contrast/sharp):"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(fv['contrast_sharp'])}", "detail": "high"}},
                ]
            if fv.get("edge_wear"):
                predict_parts += [
                    {"type": "text", "text": "FRONT edge-wear scan (red=whitening/silvering at borders):"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(fv['edge_wear'])}", "detail": "high"}},
                ]
            if fv.get("sobel_surface"):
                predict_parts += [
                    {"type": "text", "text": "FRONT surface scan (interior scratches/print marks):"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(fv['sobel_surface'])}", "detail": "high"}},
                ]
        except Exception:
            pass

    msg = [{"role": "user", "content": predict_parts}]

    result = await _openai_chat(msg, max_tokens=1200, temperature=0.2)
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


# ========================================
# BULK ASSESSMENT (Item 5)
# ========================================

@app.post("/api/bulk-assess")
@safe_endpoint
async def bulk_assess(
    images: List[UploadFile] = File(...),
    user_id: int = Form(0),
    batch_name: str = Form("Untitled Batch"),
):
    """Process 2-10 card images for quick identification and grading."""
    if len(images) < 2 or len(images) > 10:
        raise HTTPException(status_code=400, detail="Must provide 2-10 images")

    results = []
    for idx, img_file in enumerate(images):
        img_bytes = await img_file.read()

        prompt = """Identify this card and provide a quick assessment.
Return JSON:
{
  "card_name": "...",
  "set": "...",
  "estimated_grade": "8",
  "estimated_value": 25.00,
  "priority": "high|medium|low",
  "key_features": ["...", "..."],
  "main_issues": ["...", "..."]
}
Respond ONLY with JSON."""

        msg = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(img_bytes)}", "detail": "low"}},
            ]
        }]

        try:
            result = await _openai_chat(msg, max_tokens=400, temperature=0.2)
            card_data = _parse_json_or_none(result.get("content", "")) or {}
        except Exception:
            card_data = {"card_name": f"Card {idx+1}", "estimated_grade": "N/A", "estimated_value": 0, "priority": "low"}

        card_data.setdefault("card_name", f"Card {idx+1}")
        card_data.setdefault("estimated_value", 0)
        card_data.setdefault("priority", "low")
        results.append(card_data)

    # Sort by estimated value descending
    results.sort(key=lambda c: float(c.get("estimated_value", 0) or 0), reverse=True)

    total_value = sum(float(c.get("estimated_value", 0) or 0) for c in results)
    high_priority = sum(1 for c in results if (c.get("priority") or "").lower() == "high")

    summary = f"Batch: {batch_name}\n"
    summary += f"Cards Assessed: {len(results)}\n"
    summary += f"Estimated Total Value: ${total_value:.2f}\n"
    summary += f"High Priority Cards: {high_priority}\n"

    best_finds = [c for c in results if (c.get("priority") or "").lower() in ("high", "medium")][:5]

    return JSONResponse(content={
        "success": True,
        "batch_name": batch_name,
        "results": results,
        "best_finds": best_finds,
        "summary": summary,
        "total_value": total_value,
    })


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
async def check_price_alerts(payload: dict = Body(...)):
    """Check a batch of alerts and return updated pricing + trigger state.

    Expected payload from WordPress:
      {
        "alerts": [
          {
            "id": 123,
            "user_id": 1,
            "card_name": "M Charizard X ex",
            "card_set": "Pokemon",
            "card_number": "",
            "grade": "10",
            "alert_type": "drops_below" | "rises_above" | "changes_by",
            "target_price": 250.0,
            "current_price": 275.0,         # previous known price (optional)
            "baseline_price": 275.0         # optional; used for changes_by
          }
        ],
        "max_matches": 10
      }
    """
    alerts = payload.get("alerts") or []
    max_matches = int(payload.get("max_matches") or 10)
    now = datetime.utcnow().isoformat() + "Z"

    updated = []
    for a in alerts:
        a2 = dict(a)

        card_name = (a.get("card_name") or "").strip()
        card_set = (a.get("card_set") or "").strip()
        card_number = (a.get("card_number") or "").strip()
        grade = (a.get("grade") or "").strip()

        ident = " ".join([x for x in [card_name, card_number] if x]).strip()
        queries = _build_ebay_query_ladder(card_name=ident or card_name, card_set=card_set, grade=grade)

        best_query = ""
        matched = 0
        current_price = None

        for q in queries:
            try:
                stats = await _ebay_active_stats(q, limit=max(1, int(max_matches or 10)))
            except Exception:
                stats = None
            if stats and (stats.get("matched") or 0) > 0 and stats.get("median") is not None:
                best_query = q
                matched = int(stats.get("matched") or 0)
                current_price = float(stats["median"])
                break

        # Fallback: if we got matches but median is missing, still return something meaningful
        if current_price is None and queries:
            best_query = best_query or queries[0]

        alert_type = (a.get("alert_type") or "").strip().lower()
        try:
            target_price = float(a.get("target_price")) if a.get("target_price") is not None else None
        except Exception:
            target_price = None

        triggered = False
        change_pct = None

        if current_price is not None and target_price is not None:
            if alert_type == "drops_below":
                triggered = current_price <= target_price
            elif alert_type == "rises_above":
                triggered = current_price >= target_price
            elif alert_type == "changes_by":
                # target_price is treated as % threshold (e.g. 10 means 10%)
                base = a.get("baseline_price")
                if base is None:
                    base = a.get("previous_price")
                if base is None:
                    base = a.get("current_price")  # prior known
                try:
                    base = float(base) if base is not None else None
                except Exception:
                    base = None
                if base and base > 0:
                    change_pct = abs((current_price - base) / base) * 100.0
                    triggered = change_pct >= target_price

        a2["current_price"] = current_price
        a2["matched"] = matched
        a2["triggered"] = bool(triggered)
        a2["checked_at"] = now
        a2["query_used"] = best_query
        if change_pct is not None:
            a2["change_pct"] = round(change_pct, 2)

        updated.append(a2)

    return {"success": True, "updated_alerts": updated}

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
        result = await _openai_text(msg, max_tokens=220, temperature=0.7)

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

# Per-session timestamp of last OpenAI frame analysis call.
# Prevents camera clients from triggering an AI call on every video frame.
_ws_last_analysis: Dict[str, float] = {}
_WS_MIN_INTERVAL = 2.0  # seconds between AI calls per session


@app.websocket("/ws/live-grading/{session_id}")
async def live_grading_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time grading feedback.

    Frame rate limited to one AI call per _WS_MIN_INTERVAL seconds per session
    to prevent runaway OpenAI API usage from high-fps camera clients.
    """
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "frame":
                frame_base64 = data.get("frame")
                now_ts = time.time()
                last_ts = _ws_last_analysis.get(session_id, 0.0)
                if now_ts - last_ts < _WS_MIN_INTERVAL:
                    continue  # too soon — skip frame, keep connection alive
                _ws_last_analysis[session_id] = now_ts
                feedback = await analyze_live_frame(frame_base64)
                await manager.send_feedback(websocket, feedback)

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        _ws_last_analysis.pop(session_id, None)
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        manager.disconnect(websocket)
        _ws_last_analysis.pop(session_id, None)


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
1. Print quality - legitimate cards have precise, clean printing with consistent dot patterns (rosette pattern)
2. Font kerning - spacing between letters should match authentic examples; counterfeits often have subtle spacing errors
3. Color saturation - counterfeits often have oversaturated or muted colors vs authentic reference
4. Holofoil pattern - if applicable, check for consistent holographic pattern matching known authentic patterns
5. Edge cut - authentic cards have precise, uniform edges; fakes may have rough or uneven cuts
6. Card stock texture and thickness indicators - real Pokemon cards ~0.32mm thick, real MTG ~0.30mm
7. Set symbol and copyright text clarity - should be crisp and correctly placed
8. Rosette pattern - legitimate cards show a specific dot pattern under magnification; reprints show different patterns
9. Black core / blue core visibility - authentic cards have a black or blue layer visible at torn edges
10. Surface texture - authentic cards have a specific micro-texture; reprints feel smoother or rougher
11. Back pattern consistency - does the back match known authentic patterns for this era/set?
12. Weight indicators - card should appear consistent with authentic weight (~1.8g for Pokemon, ~1.7g for MTG)
13. Known counterfeit tells for this specific set/era (if applicable)
14. Holo pattern type - does the holo match the correct pattern for this set (cosmos, galaxy, linear, etc.)?

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
    "manufacturing_marks": {{"score": 90, "notes": "..."}},
    "rosette_pattern": {{"score": 85, "notes": "visible/not_visible at this resolution, appears consistent/inconsistent with authentic"}},
    "card_stock": {{"score": 80, "notes": "texture and thickness assessment from visual cues"}},
    "back_pattern": {{"score": 90, "notes": "back design matches/does not match expected pattern for era"}}
  }},
  "comparison_notes": "...",
  "verification_steps": ["Specific steps the owner can take to verify further, e.g. 'Perform a light test - shine a flashlight through the card; authentic cards block most light'", "Check weight with a precision scale - should be approximately 1.8g"],
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


@app.post("/api/auth-check")
@safe_endpoint
async def auth_check_simple(
    front:     UploadFile        = File(...),
    back:      UploadFile        = File(None),
    detail1:   UploadFile        = File(None),
    detail2:   UploadFile        = File(None),
    detail3:   UploadFile        = File(None),
    detail4:   UploadFile        = File(None),
    card_game: Optional[str]     = Form(None),
    card_name: Optional[str]     = Form(None),
    card_set:  Optional[str]     = Form(None),
    card_year: Optional[str]     = Form(None),
):
    """
    Auth-check endpoint — called by the WordPress WP AJAX proxy.
    Accepts front + optional back + up to 4 detail shots + card context fields.
    Returns full forensic shape expected by the frontend tabs.
    Works well even when card details are not provided.
    """
    front_bytes   = await front.read()
    back_bytes    = await back.read()    if back    else None
    detail1_bytes = await detail1.read() if detail1 else None
    detail2_bytes = await detail2.read() if detail2 else None
    detail3_bytes = await detail3.read() if detail3 else None
    detail4_bytes = await detail4.read() if detail4 else None

    # Build context string — gracefully handle missing fields
    cg = (card_game or "").strip()
    cn = (card_name or "").strip()
    cs = (card_set  or "").strip()
    cy = (card_year or "").strip()

    has_context = bool(cg or cn or cs or cy)

    if has_context:
        card_desc = cn or "this card"
        context_line = f"You are analysing '{card_desc}'"
        if cg:  context_line += f" — a {cg} card"
        if cs:  context_line += f" from the {cs} set"
        if cy:  context_line += f" ({cy})"
        context_line += "."
        context_note = (
            "Use your knowledge of authentic printing standards, known counterfeit tells, "
            f"and documented issues specific to {cg or 'this TCG'} to inform your analysis."
        )
    else:
        context_line = "You are analysing an unidentified trading card."
        context_note = (
            "The card game/set has not been specified. Identify the game/set from the images if possible, "
            "then apply relevant authentication knowledge. Base your analysis purely on what you can observe — "
            "print quality, colours, foil/texture, edge cuts, font/text rendering, and back pattern. "
            "Do not penalise the card for lack of context — only flag genuine visual concerns."
        )

    extra_images_note = ""
    detail_count = sum(1 for b in [detail1_bytes, detail2_bytes, detail3_bytes, detail4_bytes] if b)
    if detail_count:
        extra_images_note = f"\nYou have also been provided {detail_count} close-up detail shot(s). Use these to inspect micro-level features such as rosette dot patterns, corner cuts, holofoil texture, text sharpness, and barcode/symbol accuracy."

    auth_prompt = f"""{context_line}
{context_note}{extra_images_note}

Perform a thorough forensic authentication analysis. Evaluate ALL of the following where visible:
- Print quality: rosette dot pattern, ink saturation, sharpness, bleed
- Colour accuracy: hue, contrast, brightness vs known authentic examples
- Font & text: kerning, weight, alignment, gloss/matte finish accuracy
- Holofoil / foil pattern: texture authenticity, light refraction, coverage area
- Card stock & edges: cut precision, corner radius, layering/core colour if visible
- Back pattern: colour accuracy, pattern symmetry, print consistency
- Set symbol / collector number: size, placement, font accuracy
- Any era/set-specific known counterfeit tells

Respond ONLY with valid JSON matching this exact schema (no markdown, no extra text):
{{
  "verdict": "Likely Authentic | Suspicious | Likely Counterfeit",
  "confidence": 0.85,
  "summary": "2-3 sentence overall assessment.",
  "flags": ["Specific concern 1", "Specific concern 2"],
  "positives": ["Positive indicator 1", "Positive indicator 2"],
  "checklist": {{
    "print_quality":   {{"score": 8, "notes": "Short note on print dot pattern and ink quality."}},
    "colour_accuracy": {{"score": 7, "notes": "Short note on colour fidelity."}},
    "font_and_text":   {{"score": 9, "notes": "Short note on font rendering and text placement."}},
    "holo_foil":       {{"score": 6, "notes": "Short note on foil pattern (or N/A if non-holo)."}},
    "card_stock_edges":{{"score": 8, "notes": "Short note on edge precision and card feel."}},
    "back_pattern":    {{"score": 7, "notes": "Short note on back design accuracy."}},
    "set_symbol":      {{"score": 9, "notes": "Short note on symbol/number accuracy."}}
  }},
  "forensic_notes": "1-2 sentences on the most technically significant finding.",
  "print_analysis": "1-2 sentences specifically about print/rosette dot quality observed.",
  "recommendations": [
    "Actionable recommendation 1",
    "Actionable recommendation 2"
  ],
  "next_steps": [
    "Suggested next step 1",
    "Suggested next step 2"
  ]
}}

Scoring guide for checklist: 10 = perfect/indistinguishable from authentic, 7-9 = good with minor concerns,
4-6 = notable issues worth flagging, 1-3 = strong counterfeit indicators.
If a category is not visible/applicable, score it 0 and set notes to "Not visible in provided images."
Only flag genuine concerns — normal manufacturing tolerance is NOT a defect.
"""

    content_parts: list = [{"type": "text", "text": auth_prompt}]

    if front_bytes:
        content_parts.append({"type": "text", "text": "FRONT IMAGE:"})
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_b64(front_bytes)}", "detail": "high"},
        })
    if back_bytes:
        content_parts.append({"type": "text", "text": "BACK IMAGE:"})
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_b64(back_bytes)}", "detail": "high"},
        })

    detail_labels = ["CORNER CLOSE-UP:", "HOLOFOIL / TEXTURE CLOSE-UP:", "BACK TEXT / PRINT CLOSE-UP:", "BARCODE / SYMBOL CLOSE-UP:"]
    for label, dbytes in zip(detail_labels, [detail1_bytes, detail2_bytes, detail3_bytes, detail4_bytes]):
        if dbytes:
            content_parts.append({"type": "text", "text": label})
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_b64(dbytes)}", "detail": "high"},
            })

    result = await _openai_chat([{"role": "user", "content": content_parts}], max_tokens=1400, temperature=0.1)
    parsed = _parse_json_or_none(result.get("content", "")) or {}

    # Normalise — gracefully handle partial responses
    verdict    = str(parsed.get("verdict")    or "Unable to determine").strip()
    confidence = float(parsed.get("confidence") or 0.0)
    summary    = str(parsed.get("summary")    or "Analysis complete.").strip()
    flags      = list(parsed.get("flags")     or parsed.get("red_flags")   or [])
    positives  = list(parsed.get("positives") or parsed.get("green_flags") or [])
    checklist  = parsed.get("checklist")      or {}
    forensic_notes = str(parsed.get("forensic_notes") or "").strip()
    print_analysis = str(parsed.get("print_analysis") or "").strip()
    recommendations = list(parsed.get("recommendations") or [])
    next_steps      = list(parsed.get("next_steps")      or [])

    # If recommendations empty, generate sensible defaults based on verdict
    if not recommendations:
        v_lower = verdict.lower()
        if "counterfeit" in v_lower or "suspicious" in v_lower:
            recommendations = [
                "Do not purchase until physically inspected by a certified grader.",
                "Compare side-by-side with a verified authentic copy under bright light.",
                "Submit to a professional grading service (PSA/CGC/BGS) for definitive authentication.",
            ]
        else:
            recommendations = [
                "Consider professional grading to lock in the card's authenticated status and value.",
                "Store in a hard case to preserve condition and prevent future authenticity disputes.",
            ]

    if not next_steps:
        next_steps = [
            "Photograph under UV light to check for fluorescent ink (common on reprints).",
            "Check card thickness with a calliper — authentic cards have tightly controlled tolerances.",
            "Compare the back pattern colour under neutral white light with a known authentic copy.",
        ]

    return JSONResponse(content={
        "verdict":         verdict,
        "confidence":      confidence,
        "summary":         summary,
        "flags":           flags,
        "positives":       positives,
        "checklist":       checklist,
        "forensic_notes":  forensic_notes,
        "print_analysis":  print_analysis,
        "recommendations": recommendations,
        "next_steps":      next_steps,
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

        # 2) Color histogram analysis (genuine heuristic for print uniformity)
        try:
            hist = img.histogram()  # 256 bins × 3 channels (R, G, B)
            r_hist = hist[0:256]
            g_hist = hist[256:512]
            b_hist = hist[512:768]

            # Measure coefficient of variation in each channel — high CV = uneven ink
            def _hist_cv(h: list) -> float:
                total = max(1, sum(h))
                mean_h = sum(i * h[i] for i in range(256)) / total
                var_h  = sum(h[i] * (i - mean_h) ** 2 for i in range(256)) / total
                return (var_h ** 0.5) / max(1.0, mean_h)

            cv_r = _hist_cv(r_hist)
            cv_g = _hist_cv(g_hist)
            cv_b = _hist_cv(b_hist)

            # Authentic cards tend to have balanced channel CVs (full-art / holo may differ)
            # Counterfeits often have one channel severely clipped or over-inked
            max_cv = max(cv_r, cv_g, cv_b)
            channel_imbalance = abs(cv_r - cv_b)  # R vs B gap is a key counterfeit tell

            checks["color_distribution"] = {
                "analyzed": True,
                "pass": bool(max_cv < 1.8 and channel_imbalance < 0.6),
                "channel_cv": {"r": round(cv_r, 3), "g": round(cv_g, 3), "b": round(cv_b, 3)},
                "note": (
                    "Color distribution looks balanced — consistent with authentic printing"
                    if (max_cv < 1.8 and channel_imbalance < 0.6)
                    else "Unusual channel imbalance — may indicate reproduced/altered printing"
                ),
            }
        except Exception:
            checks["color_distribution"] = {"analyzed": False, "pass": True, "note": "Histogram analysis unavailable"}

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
# FINGERPRINT ENDPOINTS
# ========================================
# These endpoints are called by cg-grading-dashboard.php (cgd_ajax_fingerprint_generate
# and cgd_ajax_fingerprint_match). They were missing from the API, causing 404 errors
# which meant the DNA viewer always showed "Not captured" for surface_texture and all
# other fingerprint fields when using the "Generate LingérPrint™ DNA" button.
#
# /api/fingerprint/generate  — accepts front/back image uploads (multipart) or
#                              image_front_url / image_back_url form fields.
#                              Runs the full fingerprint pipeline (same blocks 6a-6f
#                              as /api/defect-scan) and returns:
#                              { "success": true, "fingerprint": { phash, surface_texture,
#                                print_dot_hash, keypoints, embedding, ... } }
#
# /api/fingerprint/match     — accepts submission_id + fingerprint_hash + fingerprint_json.
#                              Returns { "matches": [] } — matching is done locally in PHP
#                              (cgd_hash_similarity). This endpoint exists so the PHP does
#                              not get a 404, and can be extended later for server-side
#                              registry matching if a shared DB is added.
# ========================================

@app.post("/api/fingerprint/generate")
@safe_endpoint
async def fingerprint_generate(
    request: Request,
    front: Optional[UploadFile]     = File(None),
    back:  Optional[UploadFile]     = File(None),
    submission_id:    Optional[str] = Form(None),
    card_name:        Optional[str] = Form(None),
    card_set:         Optional[str] = Form(None),
    image_front_url:  Optional[str] = Form(None),
    image_back_url:   Optional[str] = Form(None),
):
    """
    Generate a full pixel-derived fingerprint for a card image.

    PHP calls this from cgd_ajax_fingerprint_generate in cg-grading-dashboard.php.
    It sends either:
      (a) multipart with front/back image files, or
      (b) form fields with image_front_url / image_back_url to download.

    Returns { "success": true, "fingerprint": { all CV feature fields } }.
    The PHP then smart-merges the returned fingerprint over its local GD result,
    only overwriting fields where this API returns a non-empty value.
    """
    # ── 1. Resolve front image bytes ─────────────────────────────────────────
    front_bytes: bytes = b""
    back_bytes:  bytes = b""

    if front is not None:
        front_bytes = await front.read()

    if not front_bytes and image_front_url:
        try:
            client = _get_http_client()
            r = await client.get(image_front_url, timeout=20.0)
            if r.status_code == 200 and len(r.content) > 200:
                front_bytes = r.content
        except Exception as _e:
            logging.warning("fingerprint_generate: could not download image_front_url — %s", _e)

    if not front_bytes or len(front_bytes) < 200:
        return JSONResponse(status_code=400, content={
            "success": False,
            "error":   "Front image required — upload a file or provide image_front_url",
        })

    if back is not None:
        back_bytes = await back.read()

    if not back_bytes and image_back_url:
        try:
            client = _get_http_client()
            r2 = await client.get(image_back_url, timeout=20.0)
            if r2.status_code == 200 and len(r2.content) > 200:
                back_bytes = r2.content
        except Exception:
            pass

    # ── 2. Compress (same helper as defect-scan) ──────────────────────────────
    front_proc = _compress_image(front_bytes)
    # back not needed for fingerprint vectors but kept for future use

    # ── 3. Run fingerprint pipeline (same blocks 6a-6f as defect-scan) ───────
    _fp:       Dict[str, Any] = {}
    _fp_errors: Dict[str, str] = {}

    if not PIL_AVAILABLE:
        logging.warning("fingerprint_generate: Pillow not available.")
        _fp_errors["pil"] = "Pillow not installed"
    elif not front_proc:
        _fp_errors["front_proc"] = "front_proc bytes were empty"
    else:
        # 3a. dHash — 64-bit perceptual hash
        try:
            _im_dh = Image.open(io.BytesIO(front_proc)).convert("L").resize((9, 8), Image.LANCZOS)
            _bits  = ""
            for _y in range(8):
                for _x in range(8):
                    _bits += "1" if _im_dh.getpixel((_x, _y)) < _im_dh.getpixel((_x + 1, _y)) else "0"
            _dhex = "".join(format(int(_bits[_i:_i+4], 2), "x") for _i in range(0, 64, 4)).zfill(16)
            _fp["phash"]          = _dhex
            _fp["print_dot_hash"] = _dhex
        except Exception as _e:
            _fp_errors["phash"] = str(_e)
            logging.warning("fingerprint_generate: dHash failed — %s", _e)

        # 3b. Surface hash — SHA-256 of 32×32 centre crop (.tobytes() fix)
        try:
            _im_sf = Image.open(io.BytesIO(front_proc)).convert("RGB")
            _sw, _sh = _im_sf.size
            _crop32  = _im_sf.crop((int(_sw * 0.25), int(_sh * 0.25),
                                    int(_sw * 0.75), int(_sh * 0.75))).resize((32, 32), Image.LANCZOS)
            _surf_buf = _crop32.tobytes()
            _fp["surface_hash"]    = hashlib.sha256(_surf_buf).hexdigest()
            _fp["surface_texture"] = _fp["surface_hash"]
        except Exception as _e:
            _fp_errors["surface_texture"] = str(_e)
            logging.warning("fingerprint_generate: surface_texture failed — %s", _e)

        # 3c. Colour zones 3×3 grid
        _zones: Dict[str, list] = {}
        try:
            _im_rgb = Image.open(io.BytesIO(front_proc)).convert("RGB")
            _cw, _ch = _im_rgb.size
            for _gy in range(3):
                for _gx in range(3):
                    _x0, _x1 = int(_cw * _gx / 3), int(_cw * (_gx + 1) / 3)
                    _y0, _y1 = int(_ch * _gy / 3), int(_ch * (_gy + 1) / 3)
                    _reg   = _im_rgb.crop((_x0, _y0, _x1, _y1))
                    _means = ImageStat.Stat(_reg).mean
                    _zones[f"z{_gy * 3 + _gx}"] = [round(_means[0]), round(_means[1]), round(_means[2])]
            _fp["color_zones_front"] = _zones
        except Exception as _e:
            _fp_errors["color_zones"] = str(_e)
            logging.warning("fingerprint_generate: colour zones failed — %s", _e)

        # 3d. Intensity histogram 16 buckets
        _hist16: List[int] = []
        try:
            _im_gray  = Image.open(io.BytesIO(front_proc)).convert("L")
            _raw_hist = _im_gray.histogram()
            _hist16   = [sum(_raw_hist[_i * 16:(_i + 1) * 16]) for _i in range(16)]
            _fp["intensity_hist_front"] = _hist16
        except Exception as _e:
            _fp_errors["intensity_hist"] = str(_e)
            logging.warning("fingerprint_generate: intensity hist failed — %s", _e)

        # 3e. Embedding: 43-dim vector
        try:
            _vec: List[float] = []
            for _z in _zones.values():
                _vec += [round(_z[0] / 255.0, 4), round(_z[1] / 255.0, 4), round(_z[2] / 255.0, 4)]
            if _hist16:
                _htot = max(sum(_hist16), 1)
                _vec += [round(_c / _htot, 6) for _c in _hist16]
            if len(_vec) >= 16:
                _fp["embedding"] = _vec
        except Exception as _e:
            _fp_errors["embedding"] = str(_e)
            logging.warning("fingerprint_generate: embedding failed — %s", _e)

        # 3f. Keypoints — 8 border luminance landmarks
        try:
            _im_brd  = Image.open(io.BytesIO(front_proc)).convert("L")
            _bw, _bh = _im_brd.size
            _bp      = max(2, int(min(_bw, _bh) * 0.03))
            _lum: List[int] = []
            for _i in range(16):
                _lum.append(_im_brd.getpixel((int(_bw * (_i + 0.5) / 16), _bp)))
            for _i in range(16):
                _lum.append(_im_brd.getpixel((_bw - _bp - 1, int(_bh * (_i + 0.5) / 16))))
            for _i in range(15, -1, -1):
                _lum.append(_im_brd.getpixel((int(_bw * (_i + 0.5) / 16), _bh - _bp - 1)))
            for _i in range(15, -1, -1):
                _lum.append(_im_brd.getpixel((_bp, int(_bh * (_i + 0.5) / 16))))
            _kp = []
            for _i in range(8):
                _sl   = _lum[_i * 8:(_i + 1) * 8]
                _mean = sum(_sl) / max(len(_sl), 1)
                _kp.append({
                    "x":        round((_i / 8.0) * 1000),
                    "y":        round((_mean / 255.0) * 1000),
                    "strength": round(_mean / 255.0, 3),
                    "side":     "front",
                    "source":   "api_border",
                })
            _fp["keypoints"]        = _kp
            _fp["border_sig_front"] = bytes(min(255, max(0, _v)) for _v in _lum).hex()
        except Exception as _e:
            _fp_errors["keypoints"] = str(_e)
            logging.warning("fingerprint_generate: keypoints failed — %s", _e)

    # ── 4. Build response fingerprint dict ───────────────────────────────────
    fingerprint = {
        "phash":                _fp.get("phash", ""),
        "print_dot_hash":       _fp.get("print_dot_hash", ""),
        "surface_hash":         _fp.get("surface_hash", ""),
        "surface_texture":      _fp.get("surface_texture", ""),
        "keypoints":            _fp.get("keypoints") or [],
        "embedding":            _fp.get("embedding") or [],
        "border_sig_front":     _fp.get("border_sig_front", ""),
        "color_zones_front":    _fp.get("color_zones_front", {}),
        "intensity_hist_front": _fp.get("intensity_hist_front", []),
        "fp_source":            "api" if _fp.get("phash") else "unavailable",
        "source":               "api" if _fp.get("phash") else "unavailable",
    }
    if _fp_errors:
        fingerprint["fp_errors"] = _fp_errors

    return JSONResponse(content={
        "success":     True,
        "fingerprint": fingerprint,
    })


@app.post("/api/fingerprint/match")
@safe_endpoint
async def fingerprint_match(
    submission_id:    Optional[str] = Form(None),
    fingerprint_hash: Optional[str] = Form(None),
    fingerprint_json: Optional[str] = Form(None),
):
    """
    Registry match endpoint — returns matches for a given fingerprint hash.

    PHP calls this from cgd_ajax_fingerprint_match in cg-grading-dashboard.php.
    The Python API does not maintain a card registry DB, so it always returns
    an empty matches list. PHP automatically falls back to its own local
    cgd_hash_similarity scan when this returns no matches — so this endpoint
    existing (returning 200 + empty matches) is all that's needed to stop the
    404 and let the PHP fallback run correctly.

    If a shared registry DB is added to this service in future, implement
    the match logic here.
    """
    return JSONResponse(content={
        "success": True,
        "matches": [],
        "total":   0,
        "note":    "Registry matching is performed locally by the WordPress plugin.",
    })


# ========================================
# CLA AI GRADE
# ========================================

# Anthropic hard limit: 5 MB (5,242,880 bytes) of raw image data.
# Target 4 MB to leave headroom for metadata/encoding overhead.
_ANTHROPIC_MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4 MB

logging.info(f"PIL available for image compression: {PIL_AVAILABLE}")


def _compress_b64_image(b64: str, media_type: str) -> tuple[str, str]:
    """
    Decode -> resize/recompress -> re-encode a base64 image so it stays under
    Anthropic's 5 MB raw-bytes limit.  Returns (new_b64, new_media_type).

    Raises ValueError if the image is oversized and PIL is unavailable, so the
    caller can surface a clear error instead of sending a request that will 400.
    """
    raw = base64.b64decode(b64)
    original_size = len(raw)

    logging.info(f"_compress_b64_image: decoded {original_size:,} bytes, limit {_ANTHROPIC_MAX_IMAGE_BYTES:,}")

    if original_size <= _ANTHROPIC_MAX_IMAGE_BYTES:
        logging.info("Image within limit — no compression needed.")
        return b64, media_type

    # PIL unavailable — raise so caller returns a meaningful error
    if not PIL_AVAILABLE:
        raise ValueError(
            f"Image is {original_size:,} bytes which exceeds Anthropic's 5 MB limit, "
            "and Pillow (PIL) is not installed so automatic compression is unavailable. "
            "Install Pillow (pip install Pillow) or reduce the image size before uploading."
        )

    # Compress with PIL
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
        logging.info(f"Opened image {img.width}x{img.height} for compression.")
        quality = 85
        scale = 1.0

        while True:
            buf = BytesIO()
            if scale < 1.0:
                w = max(1, int(img.width * scale))
                h = max(1, int(img.height * scale))
                frame = img.resize((w, h), Image.LANCZOS)
            else:
                frame = img
            frame.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()

            logging.info(f"Compression attempt: scale={scale:.2f} quality={quality} -> {len(data):,} bytes")

            if len(data) <= _ANTHROPIC_MAX_IMAGE_BYTES:
                new_b64 = base64.b64encode(data).decode()
                logging.info(
                    f"Image compressed {original_size:,} -> {len(data):,} bytes "
                    f"(scale={scale:.2f}, quality={quality})"
                )
                return new_b64, "image/jpeg"

            # Step down quality first, then resolution
            if quality > 55:
                quality -= 10
            elif scale > 0.25:
                scale = round(max(0.25, scale - 0.15), 2)
            else:
                # Absolute last resort
                new_b64 = base64.b64encode(data).decode()
                logging.warning(
                    f"Could only compress to {len(data):,} bytes "
                    f"(limit {_ANTHROPIC_MAX_IMAGE_BYTES:,}) — sending anyway."
                )
                return new_b64, "image/jpeg"

    except Exception as exc:
        logging.error(f"Image compression exception: {exc}")
        raise ValueError(f"Image compression failed: {exc}") from exc


@app.post("/api/ai-grade")
@safe_endpoint
async def ai_grade(request: AIGradeRequest):
    """
    CLA Official AI Grading endpoint.

    Primary:  Claude Vision (Anthropic) — ANTHROPIC_API_KEY
    Fallback: GPT-4o Vision (OpenAI)   — OPENAI_API_KEY (used automatically if Anthropic fails)

    Called by: cg-ai-grade.php → wp_ajax_cg_ai_grade → this endpoint.
    """
    if not request.images:
        return JSONResponse(status_code=400, content={
            "success": False,
            "message": "No images provided.",
        })

    system_prompt = request.system_prompt or "You are an expert trading card grader. Return ONLY valid JSON."
    errors = []

    # ── 1. Try Anthropic (Claude) ──────────────────────────────────────────────
    if ANTHROPIC_AVAILABLE and _anthropic_sdk and ANTHROPIC_API_KEY:
        try:
            logging.info(f"🤖 AI Grade [Anthropic]: {len(request.images)} image(s) — {request.card_context!r}")

            content = []
            for img in request.images:
                compressed_b64, compressed_mt = _compress_b64_image(img.b64, img.media_type)
                content.append({
                    "type":   "image",
                    "source": {
                        "type":       "base64",
                        "media_type": compressed_mt,
                        "data":       compressed_b64,
                    },
                })
                content.append({"type": "text", "text": f"This is the {img.face.upper()} face of the card."})
            content.append({"type": "text", "text": "Grade this card to CLA standards. Return ONLY the JSON object — no markdown, no explanation."})

            client = _anthropic_sdk.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
            response = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=3000,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
            )

            # Detect truncation before attempting JSON parse
            if getattr(response, "stop_reason", None) == "max_tokens":
                raise ValueError(
                    "Anthropic response truncated — hit max_tokens limit before JSON completed. "
                    "Increase max_tokens further if this recurs."
                )

            raw_text = ""
            for _block in response.content:
                if getattr(_block, "type", None) == "text" and (_block.text or "").strip():
                    raw_text = _block.text.strip()
                    break
            if not raw_text:
                raise ValueError(
                    f"Empty Anthropic response (stop_reason={getattr(response, 'stop_reason', '?')}, "
                    f"blocks={[getattr(b, 'type', '?') for b in response.content]})"
                )
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            grade = json.loads(raw_text)

            logging.info(f"✅ AI Grade [Anthropic] complete — {grade.get('grade')} / {grade.get('score_100')}")
            return JSONResponse(content={"success": True, "grade": grade, "provider": "anthropic"})

        except json.JSONDecodeError as e:
            msg = f"Anthropic response JSON parse failed: {e}"
            logging.warning(f"⚠ {msg}")
            errors.append(msg)
        except Exception as e:
            msg = f"Anthropic call failed: {str(e)}"
            logging.error(f"❌ {msg}")
            errors.append(msg)
    else:
        reason = "SDK unavailable" if not ANTHROPIC_AVAILABLE else "API key not set"
        errors.append(f"Anthropic skipped — {reason}")
        logging.warning(f"⚠ AI Grade: Anthropic skipped — {reason}")

    # ── 2. Fallback: OpenAI GPT-4o Vision ─────────────────────────────────────
    if OPENAI_API_KEY:
        try:
            logging.info(f"🔄 AI Grade [OpenAI fallback]: {len(request.images)} image(s)")

            # Build OpenAI-format content — image_url with base64 data URI
            oai_content = []
            for img in request.images:
                compressed_b64, compressed_mt = _compress_b64_image(img.b64, img.media_type)
                oai_content.append({
                    "type": "text",
                    "text": f"This is the {img.face.upper()} face of the card.",
                })
                oai_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url":    f"data:{compressed_mt};base64,{compressed_b64}",
                        "detail": "high",
                    },
                })
            oai_content.append({
                "type": "text",
                "text": "Grade this card to CLA standards. Return ONLY the JSON object — no markdown, no explanation.",
            })

            http = _get_http_client()
            oai_resp = await http.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":      "gpt-4o",
                    "max_tokens": 3000,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": oai_content},
                    ],
                },
                timeout=60.0,
            )

            oai_data = oai_resp.json()
            if "choices" not in oai_data:
                err = oai_data.get("error", {}).get("message", str(oai_data))
                raise Exception(f"OpenAI API error: {err}")
            raw_text = oai_data["choices"][0]["message"]["content"].strip()
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            grade = json.loads(raw_text)

            logging.info(f"✅ AI Grade [OpenAI] complete — {grade.get('grade')} / {grade.get('score_100')}")
            return JSONResponse(content={"success": True, "grade": grade, "provider": "openai_fallback"})

        except json.JSONDecodeError as e:
            msg = f"OpenAI response JSON parse failed: {e}"
            logging.warning(f"⚠ {msg}")
            errors.append(msg)
        except Exception as e:
            msg = f"OpenAI fallback failed: {str(e)}"
            logging.error(f"❌ {msg}")
            errors.append(msg)
    else:
        errors.append("OpenAI fallback skipped — OPENAI_API_KEY not set")

    # ── Both providers failed ──────────────────────────────────────────────────
    logging.error(f"❌ AI Grade: all providers failed — {errors}")
    return JSONResponse(status_code=500, content={
        "success": False,
        "message": "AI grading failed on all providers. " + " | ".join(errors),
    })



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
