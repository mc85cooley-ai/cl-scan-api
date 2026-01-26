from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
import httpx
import base64
import os
import uuid

app = FastAPI(title="Collectors League Scan API")

APP_VERSION = os.getenv("CL_SCAN_VERSION", "2026-01-26c")

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


# ---------------- Image helpers ----------------

def decode_image(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def b64_image_from_cv2(img_bgr, fmt=".jpg", quality=95):
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


def enhance_for_id(img_bgr):
    """
    Gentle enhancement for text readability (doesn't overcook).
    """
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 50, 50)
    g = cv2.equalizeHist(g)
    # unsharp
    blur = cv2.GaussianBlur(g, (0, 0), 1.2)
    sharp = cv2.addWeighted(g, 1.4, blur, -0.4, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


# ---------------- Card quad (best-effort only) ----------------

def find_card_quad(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 35, 115)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    h, w = img_bgr.shape[:2]
    img_area = h * w

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.04:  # loosened (was 0.08)
            continue

        peri = cv2.arcLength(c, True)
        # loosened approx (was 0.02)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)

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


# ---------------- Risk metrics ----------------

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


def photo_warnings(img_bgr, label: str) -> List[str]:
    warns = []
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # MUCH softer warnings (never block ID)
    if min(h, w) < 650:
        warns.append(f"{label}: Low resolution (move closer)")

    bs = blur_score(gray)
    if bs < 25:
        warns.append(f"{label}: Soft focus (tap to focus / more light)")
    if bs < 12:
        warns.append(f"{label}: Very blurry (may reduce ID accuracy)")

    over = overexposed_ratio(gray)
    if over > 0.10:
        warns.append(f"{label}: Glare/overexposure risk (tilt card / reduce reflections)")

    under = underexposed_ratio(gray)
    if under > 0.25:
        warns.append(f"{label}: Too dark (increase lighting)")

    return warns


# ---------------- OpenAI + TCG lookup ----------------

async def openai_autoid_pokemon(front_bgr):
    if not OPENAI_API_KEY:
        return {"error": True, "reason": "missing_openai_key"}

    img_b64 = b64_image_from_cv2(front_bgr, fmt=".jpg", quality=95)

    instructions = (
        "Identify this Pokémon trading card from the photo.\n"
        "Return ONLY valid JSON.\n"
        "If uncertain, use null.\n"
        "Schema:\n"
        "{"
        "\"name\": string|null,"
        "\"number\": string|null,"
        "\"set_name\": string|null,"
        "\"confidence\": number,"
        "\"notes\": string|null"
        "}\n"
        "Rules:\n"
        "- Do NOT guess.\n"
        "- If the card name is not clearly readable, set name=null.\n"
        "- If the set or number is not clearly readable, set them=null.\n"
    )

    payload = {
        "model": OPENAI_MODEL,
        "instructions": instructions,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Identify this Pokémon card (name, set, number)."},
                {"type": "input_image", "image_base64": img_b64}
            ]
        }],
        "response_format": {"type": "json_object"}
    }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

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


async def autoid_best(front_raw, front_warped=None, front_enh=None):
    """
    Try multiple candidates. Pick best by confidence.
    """
    best = None
    candidates = [front_warped, front_enh, front_raw]
    for img in candidates:
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
    return str(num).strip().replace(" ", "").replace("\\", "/")


async def pokemontcg_lookup(name, number, set_name):
    if not name:
        return None

    q_parts = []
    safe_name = name.replace('"', '\\"')
    q_parts.append(f'name:"{safe_name}"')

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
            # fallback on name only
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

        prices = None
        try:
            tcgplayer = (c0.get("tcgplayer") or {})
            tcg_prices = ((tcgplayer.get("prices") or {}) if isinstance(tcgplayer, dict) else {})
            prices = tcg_prices
        except Exception:
            prices = None

        return {
            "name": c0.get("name"),
            "number": c0.get("number"),
            "set_name": set_obj.get("name"),
            "series": set_obj.get("series"),
            "releaseDate": release,
            "year": year,
            "prices": prices,
        }


def compute_internal_value_usd(prices: Any) -> Optional[float]:
    if not prices or not isinstance(prices, dict):
        return None
    best = None
    for _, obj in prices.items():
        if not isinstance(obj, dict):
            continue
        market = obj.get("market")
        try:
            mv = float(market) if market is not None else None
        except Exception:
            mv = None
        if mv is None:
            continue
        if best is None or mv > best:
            best = mv
    return best


# ---------------- API: Identify ----------------

@app.post("/api/identify")
async def identify(front: UploadFile = File(...)):
    fb = await front.read()
    if len(fb) < 1500:
        raise HTTPException(status_code=400, detail="Front image looks empty/corrupt.")

    front_img = decode_image(fb)
    if front_img is None:
        raise HTTPException(status_code=400, detail="Could not decode Front image.")

    warnings = photo_warnings(front_img, "Front")

    fq = find_card_quad(front_img)
    front_warp = warp_card(front_img, fq, out_w=1200, out_h=1680) if fq is not None else None
    front_enh = enhance_for_id(front_warp if front_warp is not None else front_img)

    auto_best = await autoid_best(front_img, front_warp, front_enh)

    card_name = None
    card_number = None
    set_name = None
    id_conf = 0.0
    debug_openai = None

    if isinstance(auto_best, dict):
        card_name = auto_best.get("name")
        card_number = normalize_number(auto_best.get("number"))
        set_name = auto_best.get("set_name")
        try:
            id_conf = float(auto_best.get("confidence") or 0.0)
        except Exception:
            id_conf = 0.0
    else:
        tmp = await openai_autoid_pokemon(front_enh)
        if isinstance(tmp, dict) and tmp.get("error"):
            debug_openai = tmp
        warnings.append("Auto-ID: Could not identify (try closer, less glare, sharper text)")

    tcg = await pokemontcg_lookup(card_name, card_number, set_name) if card_name else None

    name_out = "Unknown"
    series_out = "Unknown"
    year_out = "Unknown"
    number_out = card_number or ""

    internal_value_usd = None
    internal_value_aud = None

    if tcg:
        name_out = tcg.get("name") or name_out
        series_out = tcg.get("series") or tcg.get("set_name") or series_out
        year_out = tcg.get("year") or year_out
        number_out = tcg.get("number") or number_out
        internal_value_usd = compute_internal_value_usd(tcg.get("prices"))
        if internal_value_usd is not None:
            internal_value_aud = round(internal_value_usd * 1.5, 2)
    else:
        if card_name:
            name_out = card_name
        if set_name:
            series_out = set_name

    identify_token = str(uuid.uuid4())

    return JSONResponse(content={
        "identify_token": identify_token,
        "name": name_out,
        "series": series_out,
        "year": year_out,
        "number": number_out,
        "confidence": float(max(0.0, min(1.0, id_conf))),
        "warnings": warnings,
        "internal_value_usd": internal_value_usd,
        "internal_value_aud": internal_value_aud,
        "version": APP_VERSION,
        "debug_openai": debug_openai
    })


# ---------------- API: Verify ----------------

@app.post("/api/verify")
async def verify(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    angle: Optional[UploadFile] = File(None),
    identify_token: Optional[str] = Form(None),
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

    # Always warnings (never block)
    warnings += photo_warnings(front_img, "Front")
    warnings += photo_warnings(back_img, "Back")
    if angle_img is not None:
        warnings += photo_warnings(angle_img, "Angled")
    else:
        warnings.append("Angled: Not provided (recommended)")

    # ID again for verify output (best effort)
    fq = find_card_quad(front_img)
    front_warp_id = warp_card(front_img, fq, out_w=1200, out_h=1680) if fq is not None else None
    front_enh = enhance_for_id(front_warp_id if front_warp_id is not None else front_img)

    auto_best = await autoid_best(front_img, front_warp_id, front_enh)

    series = "Unknown"
    year = "Unknown"
    name = "Unknown item/card"
    number = ""

    id_conf = None
    card_name = None
    card_number = None
    set_name = None
    debug_openai = None

    if isinstance(auto_best, dict):
        card_name = auto_best.get("name")
        card_number = normalize_number(auto_best.get("number"))
        set_name = auto_best.get("set_name")
        try:
            id_conf = float(auto_best.get("confidence")) if auto_best.get("confidence") is not None else None
        except Exception:
            id_conf = None
    else:
        tmp = await openai_autoid_pokemon(front_enh)
        if isinstance(tmp, dict) and tmp.get("error"):
            debug_openai = tmp
        warnings.append("Auto-ID: Could not confidently identify from photo")

    tcg = await pokemontcg_lookup(card_name, card_number, set_name) if card_name else None

    internal_value_usd = None
    internal_value_aud = None

    if tcg:
        name = tcg.get("name") or name
        year = tcg.get("year") or year
        series = tcg.get("series") or tcg.get("set_name") or series
        number = tcg.get("number") or (card_number or "")
        internal_value_usd = compute_internal_value_usd(tcg.get("prices"))
        if internal_value_usd is not None:
            internal_value_aud = round(internal_value_usd * 1.5, 2)
    else:
        if card_name:
            name = card_name
        if set_name:
            series = set_name
        number = card_number or ""

    confidence = 0.70

    # Assessment quad detection is now OPTIONAL
    bq = find_card_quad(back_img)
    aq = find_card_quad(angle_img) if angle_img is not None else None

    if fq is None:
        warnings.append("Front: Card corners not detected (OK if framed). Assessment may be limited.")
        confidence -= 0.05
    if bq is None:
        warnings.append("Back: Card corners not detected (OK if framed). Assessment may be limited.")
        confidence -= 0.06

    # If we CAN warp, do deeper checks; otherwise return a limited assessment (still usable)
    edge_white_front = None
    edge_white_back = None
    surface_risk = None

    if fq is not None:
        front_w = warp_card(front_img, fq, out_w=900, out_h=1260)
        front_g = cv2.cvtColor(front_w, cv2.COLOR_BGR2GRAY)
        edge_white_front = whitening_risk_edge(front_g)
        if edge_white_front > 0.10:
            defects.append("Front: Edge/corner whitening risk detected")
            confidence -= 0.06

    if bq is not None:
        back_w = warp_card(back_img, bq, out_w=900, out_h=1260)
        back_g = cv2.cvtColor(back_w, cv2.COLOR_BGR2GRAY)
        edge_white_back = whitening_risk_edge(back_g)
        if edge_white_back > 0.12:
            defects.append("Back: Edge/corner whitening risk detected")
            confidence -= 0.07

    if angle_img is not None and aq is not None:
        angle_w = warp_card(angle_img, aq, out_w=900, out_h=1260)
        angle_g = cv2.cvtColor(angle_w, cv2.COLOR_BGR2GRAY)
        surface_risk = surface_line_risk(angle_g)
        if surface_risk > 0.09:
            defects.append("Angled: Surface scratch / print-line risk detected")
            confidence -= 0.06

    centering_note = "Centering: Review in-hand"
    if edge_white_front is not None and edge_white_back is not None:
        if (edge_white_front < 0.06) and (edge_white_back < 0.07):
            centering_note = "Centering: Looks acceptable (photo-based)"

    subgrades: Dict[str, Any] = {
        "photo_quality": "Pass",
        "card_detected": "Partial" if (fq is None or bq is None) else "Yes",
        "centering_note": centering_note,
    }
    if edge_white_front is not None:
        subgrades["edge_whitening_front"] = round(edge_white_front, 3)
    if edge_white_back is not None:
        subgrades["edge_whitening_back"] = round(edge_white_back, 3)
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

    return JSONResponse(content={
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
        "internal_value_usd": internal_value_usd,
        "internal_value_aud": internal_value_aud,
        "debug_openai": debug_openai
    })
