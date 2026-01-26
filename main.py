from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
import httpx
import base64
import os
import json
import re
import uuid

app = FastAPI(title="Collectors League Scan API")

APP_VERSION = os.getenv("CL_SCAN_VERSION", "2026-01-26d")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
POKEMONTCG_API_KEY = os.getenv("POKEMONTCG_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")


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
        "model": OPENAI_MODEL,
    }


def decode_image(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def b64_image_from_cv2(img_bgr, fmt=".jpg", quality=92):
    encode_params = []
    if fmt.lower() in [".jpg", ".jpeg"]:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(fmt, img_bgr, encode_params)
    if not ok:
        raise ValueError("Could not encode image for base64")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


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
        if area < img_area * 0.06:
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


def warp_card(img, quad, out_w=1200, out_h=1680):
    pts = order_points(quad)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


def whitening_risk_edge(gray_warped):
    h, w = gray_warped.shape[:2]
    band = 12
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


def clamp01(x):
    return float(max(0.0, min(1.0, x)))


def normalize_number(num):
    if not num:
        return None
    s = str(num).strip()
    s = s.replace(" ", "")
    s = s.replace("\\", "/")
    return s


def safe_str(x):
    if x is None:
        return None
    return str(x).strip() if str(x).strip() else None


def crop_regions_for_id(img_bgr):
    """
    Returns a list of (label, crop_bgr) to improve hit-rate:
    - full image
    - top name bar region
    - bottom number region
    """
    h, w = img_bgr.shape[:2]
    crops = []

    crops.append(("full", img_bgr))

    # top bar (name)
    y1 = int(h * 0.03)
    y2 = int(h * 0.22)
    x1 = int(w * 0.05)
    x2 = int(w * 0.95)
    top = img_bgr[y1:y2, x1:x2].copy()
    crops.append(("top", top))

    # bottom area (collector number)
    y1b = int(h * 0.78)
    y2b = int(h * 0.98)
    x1b = int(w * 0.03)
    x2b = int(w * 0.55)
    bot = img_bgr[y1b:y2b, x1b:x2b].copy()
    crops.append(("bottom", bot))

    return crops


async def openai_autoid_pokemon(img_bgr):
    """
    FIXED: Responses API needs image_url with a data: URL.
    Returns dict schema:
      {name, number, set_name, confidence, notes}
    or {error: True, ...}
    """
    if not OPENAI_API_KEY:
        return {"error": True, "reason": "missing_openai_key"}

    img_b64 = b64_image_from_cv2(img_bgr, fmt=".jpg", quality=92)
    data_url = f"data:image/jpeg;base64,{img_b64}"

    instructions = (
        "You are identifying a Pokémon trading card from a photo.\n"
        "Return ONLY valid JSON (no markdown).\n"
        "Do not guess. If you cannot clearly read something, return null.\n"
        "Schema:\n"
        "{"
        "\"name\": string|null,"
        "\"number\": string|null,"
        "\"set_name\": string|null,"
        "\"confidence\": number,"
        "\"notes\": string|null"
        "}\n"
        "Guidance:\n"
        "- name should be the printed card name (e.g., 'Charizard ex').\n"
        "- number should be the collector number like '006/165' (keep leading zeros if shown).\n"
        "- set_name should be the set as printed/known (e.g., 'Scarlet & Violet—151').\n"
        "- confidence is 0..1.\n"
    )

    payload = {
        "model": OPENAI_MODEL,
        "instructions": instructions,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Identify this Pokémon card. Output JSON only."},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        "response_format": {"type": "json_object"},
    }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
        if r.status_code < 200 or r.status_code >= 300:
            return {"error": True, "status": r.status_code, "body": r.text[:800]}

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
            return {"error": True, "reason": "no_output_text", "raw": str(data)[:800]}

        try:
            return json.loads(text)
        except Exception:
            return {"error": True, "reason": "json_parse_failed", "text": str(text)[:800]}


def score_id_result(res: Dict[str, Any]) -> float:
    """
    Prefer results that contain BOTH name and number, then set_name.
    Use confidence as base.
    """
    if not isinstance(res, dict) or res.get("error"):
        return -1.0
    conf = 0.0
    try:
        conf = float(res.get("confidence") or 0.0)
    except Exception:
        conf = 0.0

    bonus = 0.0
    if safe_str(res.get("name")):
        bonus += 0.20
    if safe_str(res.get("number")):
        bonus += 0.25
    if safe_str(res.get("set_name")):
        bonus += 0.10

    return conf + bonus


async def autoid_front(front_bgr, front_warped=None):
    """
    Run ID on:
    - warped (if available) full/top/bottom
    - raw full/top/bottom
    Return best.
    """
    candidates = []

    imgs = []
    if front_warped is not None:
        imgs.append(("warped", front_warped))
    imgs.append(("raw", front_bgr))

    for base_label, img in imgs:
        for crop_label, crop in crop_regions_for_id(img):
            res = await openai_autoid_pokemon(crop)
            if isinstance(res, dict) and not res.get("error"):
                res["_src"] = f"{base_label}:{crop_label}"
                candidates.append(res)

    if not candidates:
        return None

    candidates.sort(key=score_id_result, reverse=True)
    return candidates[0]


async def pokemontcg_lookup(name, number, set_name):
    if not name:
        return None

    q_parts = []
    safe_name_q = name.replace('"', '\\"')
    q_parts.append(f'name:"{safe_name_q}"')

    if number:
        safe_num = number.replace('"', '\\"')
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
            params = {"q": f'name:"{safe_name_q}"', "pageSize": 3}
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

        return {
            "name": c0.get("name"),
            "number": c0.get("number"),
            "set_name": set_obj.get("name"),
            "series": set_obj.get("series"),
            "releaseDate": release,
            "year": year,
        }


def quality_warnings(img_bgr, label) -> List[str]:
    """
    Much less strict. These are WARNINGS only (never block ID).
    """
    warns = []
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # resolution warning
    if min(h, w) < 900:
        warns.append(f"{label}: Low resolution (move closer / use better light)")

    # blur warning (lower threshold than before; less annoying)
    bs = blur_score(gray)
    if bs < 35:
        warns.append(f"{label}: May be out of focus (tap to focus, hold steady)")

    over = overexposed_ratio(gray)
    if over > 0.08:
        warns.append(f"{label}: Glare risk (tilt slightly, reduce reflections)")

    under = underexposed_ratio(gray)
    if under > 0.18:
        warns.append(f"{label}: Too dark (increase lighting)")

    return warns


@app.post("/api/identify")
async def identify(front: UploadFile = File(...)):
    fb = await front.read()
    if len(fb) < 1500:
        raise HTTPException(status_code=400, detail="Front image looks empty/corrupt.")

    front_img = decode_image(fb)
    if front_img is None:
        raise HTTPException(status_code=400, detail="Could not decode front image.")

    warnings = quality_warnings(front_img, "Front")

    fq = find_card_quad(front_img)
    front_warp = warp_card(front_img, fq, out_w=1200, out_h=1680) if fq is not None else None

    best = await autoid_front(front_img, front_warp)

    identify_token = str(uuid.uuid4())

    # Defaults
    out_name = "Unknown"
    out_series = "Unknown"
    out_year = "Unknown"
    out_number = ""
    out_conf = 0.10

    if isinstance(best, dict):
        card_name = safe_str(best.get("name"))
        card_number = normalize_number(best.get("number"))
        set_name = safe_str(best.get("set_name"))
        try:
            out_conf = clamp01(float(best.get("confidence") or 0.0))
        except Exception:
            out_conf = 0.10

        # If OpenAI gave anything usable, enrich via PokemonTCG
        tcg = await pokemontcg_lookup(card_name, card_number, set_name) if card_name else None

        if tcg:
            out_name = safe_str(tcg.get("name")) or (card_name or out_name)
            out_number = safe_str(tcg.get("number")) or (card_number or out_number)
            out_series = safe_str(tcg.get("series")) or safe_str(tcg.get("set_name")) or (set_name or out_series)
            out_year = safe_str(tcg.get("year")) or out_year
        else:
            if card_name:
                out_name = card_name
            if set_name:
                out_series = set_name
            if card_number:
                out_number = card_number

    # Add a helpful note if quad wasn’t found (but do NOT fail)
    if fq is None:
        warnings.append("Front: Could not detect full card boundary (OK for ID, but assessment may need better framing).")

    return JSONResponse(
        content={
            "identify_token": identify_token,
            "name": out_name,
            "series": out_series,
            "year": out_year,
            "number": out_number,
            "confidence": out_conf,
            "warnings": warnings,
            "version": APP_VERSION,
        }
    )


@app.post("/api/verify")
async def verify(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    angle: Optional[UploadFile] = File(None),
    identify_token: Optional[str] = None,
):
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

    # Always provide warnings (never hard-block)
    warnings += quality_warnings(front_img, "Front")
    warnings += quality_warnings(back_img, "Back")
    if angle_img is not None:
        warnings += quality_warnings(angle_img, "Angle")

    # Try ID again here as well (in case they skipped Identify)
    fq = find_card_quad(front_img)
    front_warp_id = warp_card(front_img, fq, out_w=1200, out_h=1680) if fq is not None else None
    best = await autoid_front(front_img, front_warp_id)

    series = "Unknown"
    year = "Unknown"
    name = "Unknown item/card"
    number = ""

    id_conf = None
    if isinstance(best, dict):
        card_name = safe_str(best.get("name"))
        card_number = normalize_number(best.get("number"))
        set_name = safe_str(best.get("set_name"))
        try:
            id_conf = float(best.get("confidence")) if best.get("confidence") is not None else None
        except Exception:
            id_conf = None

        tcg = await pokemontcg_lookup(card_name, card_number, set_name) if card_name else None
        if tcg:
            name = safe_str(tcg.get("name")) or name
            number = safe_str(tcg.get("number")) or (card_number or "")
            year = safe_str(tcg.get("year")) or year
            series = safe_str(tcg.get("series")) or safe_str(tcg.get("set_name")) or series
        else:
            if card_name:
                name = card_name
            if set_name:
                series = set_name
            if card_number:
                number = card_number

    if id_conf is not None and id_conf < 0.35:
        warnings.append("Auto-ID: Low confidence (try less glare / closer / sharper focus)")

    # ---------- Assessment (LESS STRICT) ----------
    bq = find_card_quad(back_img)
    aq = find_card_quad(angle_img) if angle_img is not None else None

    if fq is None or bq is None:
        defects.append("Assessment: Card boundary not detected (for assessment only). Ensure the full card is inside frame.")
        confidence -= 0.10

        return JSONResponse(
            content={
                "pregrade": "Pre-Assessment: Limited (Needs Better Framing)",
                "preapproval": "Pre-Approved — but assessment is limited due to framing (final grading in-hand)",
                "series": series,
                "year": year,
                "name": name,
                "number": number,
                "defects": defects,
                "warnings": warnings,
                "confidence": float(max(0.10, min(0.90, confidence))),
                "subgrades": {"assessment_quality": "Limited"},
                "version": APP_VERSION,
            }
        )

    # Warp for assessment
    front_w = warp_card(front_img, fq, out_w=900, out_h=1260)
    back_w = warp_card(back_img, bq, out_w=900, out_h=1260)

    front_g = cv2.cvtColor(front_w, cv2.COLOR_BGR2GRAY)
    back_g = cv2.cvtColor(back_w, cv2.COLOR_BGR2GRAY)

    edge_white_front = whitening_risk_edge(front_g)
    edge_white_back = whitening_risk_edge(back_g)

    if edge_white_front > 0.12:
        defects.append("Front: Edge/corner whitening risk detected")
        confidence -= 0.05
    if edge_white_back > 0.14:
        defects.append("Back: Edge/corner whitening risk detected")
        confidence -= 0.06

    surface_risk = None
    if angle_img is not None and aq is not None:
        angle_w = warp_card(angle_img, aq, out_w=900, out_h=1260)
        angle_g = cv2.cvtColor(angle_w, cv2.COLOR_BGR2GRAY)
        surface_risk = surface_line_risk(angle_g)
        if surface_risk > 0.11:
            defects.append("Angle: Surface scratch / print-line risk detected")
            confidence -= 0.05
    elif angle_img is not None and aq is None:
        warnings.append("Angle: Could not detect card boundary (OK — try again if you want surface screening).")

    centering_note = "Centering: Review in-hand"
    if (edge_white_front < 0.07) and (edge_white_back < 0.08):
        centering_note = "Centering: Looks acceptable (photo-based)"

    subgrades: Dict[str, Any] = {
        "edge_whitening_front": round(edge_white_front, 3),
        "edge_whitening_back": round(edge_white_back, 3),
        "centering_note": centering_note,
    }
    if surface_risk is not None:
        subgrades["surface_risk"] = round(float(surface_risk), 3)

    if len(defects) == 0:
        preapproval = "Pre-Approved — proceed to submission (final assessment in-hand)"
        summary = "Pre-Assessment: Clear"
        confidence += 0.08
    elif len(defects) <= 3:
        preapproval = "Pre-Approved — proceed (minor risks flagged)"
        summary = "Pre-Assessment: Minor Risks"
    else:
        preapproval = "Pre-Approved — manual review recommended (multiple risks flagged)"
        summary = "Pre-Assessment: Review Recommended"
        confidence -= 0.05

    return JSONResponse(
        content={
            "pregrade": summary,
            "preapproval": preapproval,
            "series": series,
            "year": year,
            "name": name,
            "number": number,
            "defects": defects,
            "warnings": warnings,
            "confidence": float(max(0.10, min(0.90, confidence))),
            "subgrades": subgrades,
            "version": APP_VERSION,
        }
    )
