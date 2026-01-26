from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
import httpx
import base64
import os

app = FastAPI(title="Collectors League Scan API")

# Bump this every time you redeploy, so you can confirm Render is running the right code.
APP_VERSION = os.getenv("CL_SCAN_VERSION", "2026-01-26a")

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
        "model": OPENAI_MODEL
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


def b64_image_from_cv2(img_bgr, fmt=".jpg", quality=90):
    encode_params = []
    if fmt.lower() in [".jpg", ".jpeg"]:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(fmt, img_bgr, encode_params)
    if not ok:
        raise ValueError("Could not encode image for base64")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


async def openai_autoid_pokemon(front_bgr):
    if not OPENAI_API_KEY:
        return None

    img_b64 = b64_image_from_cv2(front_bgr, fmt=".jpg", quality=90)

    instructions = (
        "You are identifying a Pokémon trading card from a photo. "
        "Return ONLY valid JSON. If unsure, use nulls. "
        "Fields: "
        "{"
        "\"name\": string|null, "
        "\"number\": string|null, "
        "\"set_name\": string|null, "
        "\"confidence\": number (0..1), "
        "\"notes\": string|null"
        "} "
        "Rules: "
        "- 'number' should look like '199/165' or '151' etc if visible. "
        "- 'set_name' should be the set on the card if visible; otherwise null. "
        "- Do not invent."
    )

    payload = {
        "model": OPENAI_MODEL,
        "instructions": instructions,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Identify this Pokémon card."},
                {"type": "input_image", "image_base64": img_b64}
            ]
        }],
        "response_format": {"type": "json_object"}
    }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
        if r.status_code < 200 or r.status_code >= 300:
            return None

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
            return None

        try:
            import json
            return json.loads(text)
        except Exception:
            return None


def normalize_number(num):
    if not num:
        return None
    s = str(num).strip().replace(" ", "").replace("\\", "/")
    return s


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

        return {
            "name": c0.get("name"),
            "number": c0.get("number"),
            "set_name": set_obj.get("name"),
            "series": set_obj.get("series"),
            "releaseDate": release,
            "year": year
        }


def assess_quality(img, label, defects, confidence_ref):
    confidence = confidence_ref
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if min(h, w) < 900:
        defects.append(f"{label}: Low resolution (move closer / higher quality)")
        confidence -= 0.12

    bs = blur_score(gray)
    if bs < 80:
        defects.append(f"{label}: Blurry / out of focus")
        confidence -= 0.14

    over = overexposed_ratio(gray)
    if over > 0.06:
        defects.append(f"{label}: Glare/overexposure risk (reduce reflections)")
        confidence -= 0.10

    under = underexposed_ratio(gray)
    if under > 0.15:
        defects.append(f"{label}: Too dark (increase lighting)")
        confidence -= 0.08

    return confidence


@app.post("/api/verify")
async def verify(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    angle: Optional[UploadFile] = File(None),  # ✅ optional
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
    confidence = 0.70

    # ---------- Always attempt Auto-ID (best-effort) ----------
    fq = find_card_quad(front_img)
    front_for_id = warp_card(front_img, fq) if fq is not None else front_img

    series = "Unknown"
    year = "Unknown"
    name = "Unknown item/card"

    auto = await openai_autoid_pokemon(front_for_id)
    card_name = None
    card_number = None
    set_name = None
    id_conf = None

    if isinstance(auto, dict):
        card_name = auto.get("name")
        card_number = normalize_number(auto.get("number"))
        set_name = auto.get("set_name")
        try:
            id_conf = float(auto.get("confidence")) if auto.get("confidence") is not None else None
        except Exception:
            id_conf = None

    tcg = await pokemontcg_lookup(card_name, card_number, set_name) if card_name else None

    if tcg:
        name = tcg.get("name") or name
        year = tcg.get("year") or year
        series = tcg.get("series") or tcg.get("set_name") or series
    else:
        if card_name:
            name = card_name
        if set_name:
            series = set_name

    if id_conf is not None and id_conf < 0.45:
        defects.append("Auto-ID: Low confidence (text/set not clear in photo)")
        confidence -= 0.05

    # ---------- Photo quality checks ----------
    confidence = assess_quality(front_img, "Front", defects, confidence)
    confidence = assess_quality(back_img, "Back", defects, confidence)
    if angle_img is not None:
        confidence = assess_quality(angle_img, "Angled", defects, confidence)
    else:
        defects.append("Angled: Not provided (recommended for surface/foil assessment)")

    critical = any(("Low resolution" in d) or ("Blurry" in d) for d in defects)
    if critical:
        return JSONResponse(content={
            "pregrade": "Pre-Assessment: Rescan Required",
            "preapproval": "Not pre-approved — photo quality insufficient for assessment",
            "series": series,
            "year": year,
            "name": name,
            "defects": defects,
            "confidence": float(max(0.10, min(0.90, confidence))),
            "subgrades": {"photo_quality": "Fail", "card_detected": "Unknown"},
            "version": APP_VERSION
        })

    # ---------- Card detection / warp (assessment) ----------
    bq = find_card_quad(back_img)
    aq = find_card_quad(angle_img) if angle_img is not None else None

    if fq is None or bq is None:
        defects.append("Card boundary not clearly detected (ensure all 4 corners visible, fill frame, reduce glare)")
        confidence -= 0.18
        return JSONResponse(content={
            "pregrade": "Pre-Assessment: Needs Rescan",
            "preapproval": "Not pre-approved — card framing insufficient for assessment",
            "series": series,
            "year": year,
            "name": name,
            "defects": defects,
            "confidence": float(max(0.10, min(0.90, confidence))),
            "subgrades": {"photo_quality": "Pass", "card_detected": "No"},
            "version": APP_VERSION
        })

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
        defects.append("Angled: Could not detect card boundary (keep corners visible / reduce glare)")

    centering_note = "Centering: Review in-hand"
    if (edge_white_front < 0.06) and (edge_white_back < 0.07):
        centering_note = "Centering: Looks acceptable (photo-based)"

    subgrades: Dict[str, Any] = {
        "photo_quality": "Pass",
        "card_detected": "Yes",
        "edge_whitening_front": round(edge_white_front, 3),
        "edge_whitening_back": round(edge_white_back, 3),
        "centering_note": centering_note
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
        preapproval = "Pre-Approved — manual review required (multiple risks flagged)"
        summary = "Pre-Assessment: Review Recommended"
        confidence -= 0.05

    return JSONResponse(content={
        "pregrade": summary,
        "preapproval": preapproval,
        "series": series,
        "year": year,
        "name": name,
        "defects": defects,
        "confidence": float(max(0.10, min(0.90, confidence))),
        "subgrades": subgrades,
        "version": APP_VERSION
    })
