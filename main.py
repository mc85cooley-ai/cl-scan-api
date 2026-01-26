from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import cv2
import httpx
import base64
import os
import time
import uuid
import re

app = FastAPI(title="Collectors League Scan API")

# Bump this value when you redeploy if you want a visible version tag.
APP_VERSION = os.getenv("CL_SCAN_VERSION", "2026-01-26a")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
POKEMONTCG_API_KEY = os.getenv("POKEMONTCG_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")

# Optional enforcement: set REQUIRE_X_CLIS_KEY=1 and CLIS_X_KEY=...
REQUIRE_X_CLIS_KEY = os.getenv("REQUIRE_X_CLIS_KEY", "0").strip()
CLIS_X_KEY = os.getenv("CLIS_X_KEY", "").strip()

# --- Identify token store (in-memory TTL) ---
IDENT_TTL_SECONDS = 20 * 60
_ident_store: Dict[str, Dict[str, Any]] = {}

def _now() -> float:
    return time.time()

def _prune_store():
    t = _now()
    dead = [k for k, v in _ident_store.items() if (t - v.get("ts", 0)) > IDENT_TTL_SECONDS]
    for k in dead:
        _ident_store.pop(k, None)

def _require_key_if_enabled(request: Request):
    if REQUIRE_X_CLIS_KEY == "1":
        key = request.headers.get("X-CLIS-KEY", "")
        if not key or key != CLIS_X_KEY:
            raise HTTPException(status_code=403, detail="Access denied.")


@app.get("/")
def root():
    return {"status": "ok", "service": "cl-scan-api", "version": APP_VERSION}


@app.get("/health")
def health():
    _prune_store()
    return {
        "ok": True,
        "service": "cl-scan-api",
        "version": APP_VERSION,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_pokemontcg_key": bool(POKEMONTCG_API_KEY),
        "model": OPENAI_MODEL,
    }


def decode_image(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def blur_score(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def overexposed_ratio(gray):
    return float((gray >= 245).mean())


def underexposed_ratio(gray):
    return float((gray <= 10).mean())


def find_card_quad(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

    h, w = img_bgr.shape[:2]
    img_area = h * w

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.08:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    return None


def order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_card(img, quad, out_w=900, out_h=1260):
    pts = order_points(quad)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


def whitening_risk_edge(gray_warped):
    h, w = gray_warped.shape[:2]
    band = 10
    top = gray_warped[0:band, :]
    bot = gray_warped[h - band:h, :]
    left = gray_warped[:, 0:band]
    right = gray_warped[:, w - band:w]
    ring = np.concatenate([top.flatten(), bot.flatten(), left.flatten(), right.flatten()])
    return float((ring >= 235).mean())


def surface_line_risk(gray_angle_warped):
    blur = cv2.GaussianBlur(gray_angle_warped, (0, 0), 2.0)
    high = cv2.absdiff(gray_angle_warped, blur)
    denom = max(1.0, float(gray_angle_warped.mean()))
    return float(high.mean() / denom)


def b64_image_from_cv2(img_bgr, fmt=".jpg", quality=95):
    encode_params = []
    if fmt.lower() in [".jpg", ".jpeg"]:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(fmt, img_bgr, encode_params)
    if not ok:
        raise ValueError("Could not encode image for base64")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


async def openai_autoid_pokemon(front_bgr):
    """
    Returns either:
    - a normal dict: {name, number, set_name, confidence, notes}
    - OR an error dict: {error: True, ...}
    """
    if not OPENAI_API_KEY:
        return {"error": True, "reason": "missing_openai_key"}

    img_b64 = b64_image_from_cv2(front_bgr, fmt=".jpg", quality=95)

    instructions = (
        "Identify this Pokémon trading card from the photo.\n"
        "Return ONLY valid JSON.\n"
        "Schema:\n"
        "{"
        "\"name\": string|null,"
        "\"number\": string|null,"          # allow 199/165
        "\"set_name\": string|null,"
        "\"confidence\": number,"
        "\"notes\": string|null"
        "}\n"
        "Rules:\n"
        "- Prefer reading card name and card number.\n"
        "- If uncertain, set that field null and lower confidence.\n"
        "- Confidence must be 0..1.\n"
    )

    payload = {
        "model": OPENAI_MODEL,
        "instructions": instructions,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Identify this Pokémon card (front)."},
                {"type": "input_image", "image_base64": img_b64}
            ]
        }],
        "response_format": {"type": "json_object"}
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)

        if r.status_code < 200 or r.status_code >= 300:
            return {"error": True, "status": r.status_code, "body": r.text[:500]}

        data = r.json()
        text = None

        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") in ("output_text", "text"):
                        text = c.get("text")
                        break
            if text:
                break

        if not text:
            return {"error": True, "reason": "no_output_text", "raw": str(data)[:500]}

        try:
            import json
            return json.loads(text)
        except Exception:
            return {"error": True, "reason": "json_parse_failed", "text": str(text)[:500]}


async def autoid_best(front_raw, front_warped=None):
    """
    Try both warped + raw (warped usually better for text; raw sometimes better if warp is wrong).
    Returns best non-error result by confidence, or None if none succeeded.
    """
    best = None
    for img in [front_warped, front_raw]:
        if img is None:
            continue
        res = await openai_autoid_pokemon(img)
        if isinstance(res, dict) and not res.get("error"):
            try:
                conf = float(res.get("confidence") or 0)
            except Exception:
                conf = 0.0
            if best is None:
                best = res
            else:
                try:
                    best_conf = float(best.get("confidence") or 0)
                except Exception:
                    best_conf = 0.0
                if conf > best_conf:
                    best = res
    return best


def normalize_number(num):
    if not num:
        return None
    s = str(num).strip().replace(" ", "").replace("\\", "/")
    # Try to keep 199/165 if present; if it's just "199" keep it too
    return s


async def pokemontcg_lookup(name, number, set_name):
    if not name:
        return None

    q_parts = []
    safe_name = name.replace('"', '\\"')
    q_parts.append(f'name:"{safe_name}"')

    # PokémonTCG API stores number as left part (e.g., "199"), so if we got "199/165" use "199"
    num_left = None
    if number:
        s = str(number).strip()
        m = re.match(r"^(\d{1,3})\s*/\s*(\d{1,3})$", s)
        num_left = m.group(1) if m else s
    if num_left:
        safe_num = str(num_left).replace('"', '\\"')
        q_parts.append(f'number:"{safe_num}"')

    if set_name:
        safe_set = set_name.replace('"', '\\"')
        q_parts.append(f'set.name:"{safe_set}"')

    q = " ".join(q_parts)

    headers = {}
    if POKEMONTCG_API_KEY:
        headers["X-Api-Key"] = POKEMONTCG_API_KEY

    url = "https://api.pokemontcg.io/v2/cards"
    params = {"q": q, "pageSize": 3}

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, headers=headers, params=params)
        if r.status_code != 200:
            return None

        js = r.json()
        cards = js.get("data", [])
        if not cards:
            params = {"q": f'name:"{safe_name}"', "pageSize": 3}
            r2 = await client.get(url, headers=headers, params=params)
            if r2.status_code != 200:
                return None
            cards = r2.json().get("data", [])
            if not cards:
                return None

        c0 = cards[0]
        set_obj = c0.get("set", {}) or {}
        release = set_obj.get("releaseDate")
        year = release[:4] if isinstance(release, str) and len(release) >= 4 and release[:4].isdigit() else None

        # value estimate (hidden)
        value_est = None
        try:
            tcg = c0.get("tcgplayer", {}) or {}
            prices = tcg.get("prices", {}) or {}
            best = None
            for variant, pdata in prices.items():
                if not isinstance(pdata, dict):
                    continue
                for k in ["market", "mid", "high", "low"]:
                    v = pdata.get(k)
                    if isinstance(v, (int, float)):
                        best = v if best is None else max(best, v)
            if best is not None:
                value_est = float(best)
        except Exception:
            value_est = None

        return {
            "name": c0.get("name"),
            "number": c0.get("number"),
            "set_name": set_obj.get("name"),
            "series": set_obj.get("series"),
            "releaseDate": release,
            "year": year,
            "value_est": value_est
        }


def quality_warnings(img, label) -> Tuple[List[str], float]:
    """
    Returns (warnings, confidence_delta) where delta is negative if quality is poor.
    NEVER blocks processing.
    """
    warnings: List[str] = []
    delta = 0.0

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if min(h, w) < 900:
        warnings.append(f"{label}: Low resolution (move closer / higher quality)")
        delta -= 0.10

    bs = blur_score(gray)
    if bs < 70:
        warnings.append(f"{label}: Possible blur / focus issue")
        delta -= 0.08

    over = overexposed_ratio(gray)
    if over > 0.06:
        warnings.append(f"{label}: Glare/overexposure risk (reduce reflections)")
        delta -= 0.06

    under = underexposed_ratio(gray)
    if under > 0.15:
        warnings.append(f"{label}: Too dark (increase lighting)")
        delta -= 0.05

    return warnings, delta


def value_weighted_preapproval(preapproval: str, value_est: Optional[float]) -> str:
    """
    Hidden rule: high value cards get bumped up a step.
    Not disclosed to user.
    """
    if value_est is None:
        return preapproval

    low = preapproval.lower()
    if value_est >= 250:
        if "not pre-approved" in low or "declined" in low:
            return "Review Required — high value item (internal priority)"
        if "review" in low:
            return "Pre-Approved — proceed (internal priority)"
    if value_est >= 100:
        if "not pre-approved" in low or "declined" in low:
            return "Review Required — internal priority"
    return preapproval


# -------------------- NEW: Check 1 /api/identify --------------------
@app.post("/api/identify")
async def identify(request: Request, front: UploadFile = File(...)):
    _require_key_if_enabled(request)
    _prune_store()

    fb = await front.read()
    if len(fb) < 1500:
        raise HTTPException(status_code=400, detail="Front image looks empty/corrupt.")

    front_img = decode_image(fb)
    if front_img is None:
        raise HTTPException(status_code=400, detail="Could not decode Front image.")

    # Warnings only
    warnings, delta = quality_warnings(front_img, "Front")

    # Warp for ID attempt (better text)
    fq = find_card_quad(front_img)
    front_warp_id = warp_card(front_img, fq, out_w=1200, out_h=1680) if fq is not None else None

    auto_best = await autoid_best(front_img, front_warp_id)

    debug_openai = None
    if auto_best is None:
        tmp = await openai_autoid_pokemon(front_warp_id if front_warp_id is not None else front_img)
        if isinstance(tmp, dict) and tmp.get("error"):
            debug_openai = tmp

    card_name = None
    card_number = None
    set_name = None
    id_conf = 0.35

    if isinstance(auto_best, dict):
        card_name = auto_best.get("name")
        card_number = normalize_number(auto_best.get("number"))
        set_name = auto_best.get("set_name")
        try:
            id_conf = float(auto_best.get("confidence")) if auto_best.get("confidence") is not None else id_conf
        except Exception:
            id_conf = id_conf

    tcg = await pokemontcg_lookup(card_name, card_number, set_name) if card_name else None

    series = "Unknown"
    year = "Unknown"
    name = "Unknown item/card"
    number_out = card_number or ""

    value_est = None

    if tcg:
        name = tcg.get("name") or name
        year = tcg.get("year") or year
        series = tcg.get("series") or tcg.get("set_name") or series
        value_est = tcg.get("value_est")
        # if we didn't get full 199/165, keep user-visible number as what we have
        if not number_out and tcg.get("number"):
            number_out = str(tcg.get("number"))
        id_conf = max(id_conf, 0.60)
    else:
        if card_name:
            name = card_name
        if set_name:
            series = set_name

    # Generate token for stage 2
    token = str(uuid.uuid4())
    _ident_store[token] = {
        "ts": _now(),
        "identity": {"name": name, "series": series, "year": year, "number": number_out, "confidence": float(max(0.0, min(1.0, id_conf)))},
        "value_est": value_est,
        "debug_openai": debug_openai,
    }

    return JSONResponse(content={
        "ok": True,
        "identify_token": token,
        "name": name,
        "series": series,
        "year": year,
        "number": number_out,
        "confidence": float(max(0.0, min(1.0, id_conf + delta))),
        "warnings": warnings,
        # keep debug optional; you can remove later
        "debug_openai": debug_openai,
        "version": APP_VERSION
    })


# -------------------- UPDATED: /api/verify (Check 2) --------------------
@app.post("/api/verify")
async def verify(
    request: Request,
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    angle: Optional[UploadFile] = File(None),
    identify_token: Optional[str] = Form(None),  # ✅ stage-1 token (optional)
):
    _require_key_if_enabled(request)
    _prune_store()

    fb = await front.read()
    bb = await back.read()
    ab = await angle.read() if angle is not None else None

    if min(len(fb), len(bb)) < 1500:
        raise HTTPException(status_code=400, detail="Front or Back image looks empty/corrupt.")

    front_img = decode_image(fb)
    back_img = decode_image(bb)
    angle_img = decode_image(ab) if ab else None

    if front_img is None or back_img is None:
        raise HTTPException(status_code=400, detail="Could not decode Front or Back image.")
    if ab and angle_img is None:
        angle_img = None

    defects: List[str] = []
    warnings: List[str] = []
    confidence = 0.70

    # ------------- Identity: prefer identify_token -------------
    series = "Unknown"
    year = "Unknown"
    name = "Unknown item/card"
    value_est = None
    debug_openai = None

    if identify_token and identify_token in _ident_store:
        cached = _ident_store.get(identify_token) or {}
        ident = cached.get("identity") or {}
        name = ident.get("name") or name
        series = ident.get("series") or series
        year = ident.get("year") or year
        debug_openai = cached.get("debug_openai")
        value_est = cached.get("value_est")
    else:
        # Fallback: best-effort auto-id again (but do NOT block on it)
        fq_tmp = find_card_quad(front_img)
        front_warp_id = warp_card(front_img, fq_tmp, out_w=1200, out_h=1680) if fq_tmp is not None else None
        auto_best = await autoid_best(front_img, front_warp_id)

        if auto_best is None:
            tmp = await openai_autoid_pokemon(front_warp_id if front_warp_id is not None else front_img)
            if isinstance(tmp, dict) and tmp.get("error"):
                debug_openai = tmp

        card_name = None
        card_number = None
        set_name = None
        id_conf = None

        if isinstance(auto_best, dict):
            card_name = auto_best.get("name")
            card_number = normalize_number(auto_best.get("number"))
            set_name = auto_best.get("set_name")
            try:
                id_conf = float(auto_best.get("confidence")) if auto_best.get("confidence") is not None else None
            except Exception:
                id_conf = None

        tcg = await pokemontcg_lookup(card_name, card_number, set_name) if card_name else None
        if tcg:
            name = tcg.get("name") or name
            year = tcg.get("year") or year
            series = tcg.get("series") or tcg.get("set_name") or series
            value_est = tcg.get("value_est")
        else:
            if card_name:
                name = card_name
            if set_name:
                series = set_name

        if id_conf is not None and id_conf < 0.45:
            warnings.append("Auto-ID: Low confidence (text/set not clear in photo)")
            confidence -= 0.04

    # ------------- Quality warnings (never hard fail) -------------
    w1, d1 = quality_warnings(front_img, "Front"); warnings += w1; confidence += d1
    w2, d2 = quality_warnings(back_img, "Back");  warnings += w2; confidence += d2
    if angle_img is not None:
        w3, d3 = quality_warnings(angle_img, "Angled"); warnings += w3; confidence += d3
    else:
        warnings.append("Angled: Not provided (recommended for surface/foil assessment)")

    # ------------- Card boundary detection (assessment) -------------
    fq = find_card_quad(front_img)
    bq = find_card_quad(back_img)
    aq = find_card_quad(angle_img) if angle_img is not None else None

    subgrades: Dict[str, Any] = {
        "photo_quality": "Warn" if warnings else "Pass",
        "card_detected": "Yes" if (fq is not None and bq is not None) else "No",
        "version": APP_VERSION
    }

    # If we can't detect card boundaries, DO NOT stop. Return softer assessment.
    if fq is None or bq is None:
        defects.append("Card boundary not reliably detected (fill frame, reduce glare, strong contrast background)")
        confidence -= 0.14

        preapproval = "Review Required — card framing unclear (photo-based)"
        summary = "Pre-Assessment: Review Required"

        # Hidden value weighting
        preapproval = value_weighted_preapproval(preapproval, value_est)

        return JSONResponse(content={
            "pregrade": summary,
            "preapproval": preapproval,
            "series": series,
            "year": year,
            "name": name,
            "defects": defects,
            "warnings": warnings,
            "confidence": float(max(0.10, min(0.90, confidence))),
            "subgrades": subgrades,
            "version": APP_VERSION,
            "debug_openai": debug_openai
        })

    # If detected, proceed with warp-based checks
    front_w = warp_card(front_img, fq)
    back_w = warp_card(back_img, bq)

    front_g = cv2.cvtColor(front_w, cv2.COLOR_BGR2GRAY)
    back_g = cv2.cvtColor(back_w, cv2.COLOR_BGR2GRAY)

    edge_white_front = whitening_risk_edge(front_g)
    edge_white_back = whitening_risk_edge(back_g)

    if edge_white_front > 0.10:
        defects.append("Front: Edge/corner whitening risk detected")
        confidence -= 0.06
    if edge_white_back > 0.12:
        defects.append("Back: Edge/corner whitening risk detected")
        confidence -= 0.07

    surface_risk = None
    if angle_img is not None and aq is not None:
        angle_w = warp_card(angle_img, aq)
        angle_g = cv2.cvtColor(angle_w, cv2.COLOR_BGR2GRAY)
        surface_risk = surface_line_risk(angle_g)
        if surface_risk > 0.09:
            defects.append("Angled: Surface scratch / print-line risk detected")
            confidence -= 0.06
    elif angle_img is not None and aq is None:
        warnings.append("Angled: Could not detect card boundary (reduce glare / keep corners visible)")

    centering_note = "Centering: Review in-hand"
    if (edge_white_front < 0.06) and (edge_white_back < 0.07):
        centering_note = "Centering: Looks acceptable (photo-based)"

    subgrades.update({
        "photo_quality": "Warn" if warnings else "Pass",
        "card_detected": "Yes",
        "edge_whitening_front": round(edge_white_front, 3),
        "edge_whitening_back": round(edge_white_back, 3),
        "centering_note": centering_note
    })
    if surface_risk is not None:
        subgrades["surface_risk"] = round(float(surface_risk), 3)

    # Decision summary
    if len(defects) == 0:
        preapproval = "Pre-Approved — proceed to submission (final assessment in-hand)"
        summary = "Pre-Assessment: Clear"
        confidence += 0.08
    elif len(defects) <= 3:
        preapproval = "Pre-Approved — proceed (minor risks flagged)"
        summary = "Pre-Assessment: Minor Risks"
    else:
        preapproval = "Review Required — multiple risks flagged (photo-based)"
        summary = "Pre-Assessment: Review Required"
        confidence -= 0.05

    # Hidden value weighting (NOT exposed)
    preapproval = value_weighted_preapproval(preapproval, value_est)

    return JSONResponse(content={
        "pregrade": summary,
        "preapproval": preapproval,
        "series": series,
        "year": year,
        "name": name,
        "defects": defects,
        "warnings": warnings,
        "confidence": float(max(0.10, min(0.90, confidence))),
        "subgrades": subgrades,
        "version": APP_VERSION,
        "debug_openai": debug_openai
    })
