# main.py - Simple Pokemon TCG API Only (No OCR, No OpenAI)
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
import httpx
import os
import json
import uuid

app = FastAPI(title="Collectors League Scan API")

APP_VERSION = os.getenv("CL_SCAN_VERSION", "2026-01-27-simple-tcg")

POKEMONTCG_API_KEY = os.getenv("POKEMONTCG_API_KEY", "").strip()


@app.get("/")
def root():
    return {"status": "ok", "service": "cl-scan-api", "version": APP_VERSION}


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "cl-scan-api",
        "version": APP_VERSION,
        "has_pokemontcg_key": bool(POKEMONTCG_API_KEY),
        "method": "Pokemon TCG API (manual entry)"
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


async def pokemontcg_search(name=None, number=None) -> Optional[dict]:
    """Search Pokemon TCG API"""
    if not name and not number:
        return None

    q_parts = []
    
    if name:
        clean_name = name.replace('"', '').strip()
        q_parts.append(f'name:"{clean_name}"')
    
    if number:
        safe_num = number.replace('"', '').strip()
        q_parts.append(f'number:"{safe_num}"')

    q = " ".join(q_parts)
    
    print(f"Pokemon TCG API query: {q}")

    headers = {}
    if POKEMONTCG_API_KEY:
        headers["X-Api-Key"] = POKEMONTCG_API_KEY

    url = "https://api.pokemontcg.io/v2/cards"
    params = {"q": q, "pageSize": 5, "orderBy": "-set.releaseDate"}

    try:
        async with httpx.AsyncClient(timeout=25) as client:
            r = await client.get(url, headers=headers, params=params)
            
            if r.status_code != 200:
                print(f"Pokemon TCG API error: {r.status_code}")
                return None

            data = r.json()
            cards = data.get("data", [])
            
            print(f"Pokemon TCG API returned {len(cards)} results")
            
            if not cards and name:
                # Try fuzzy search
                params = {"q": f'name:"{clean_name}"', "pageSize": 5}
                r2 = await client.get(url, headers=headers, params=params)
                if r2.status_code == 200:
                    cards = r2.json().get("data", [])
            
            if not cards:
                return None

            card = cards[0]
            set_obj = card.get("set", {}) or {}
            release = set_obj.get("releaseDate")
            year = release[:4] if isinstance(release, str) and len(release) >= 4 and release[:4].isdigit() else None

            return {
                "name": card.get("name"),
                "number": card.get("number"),
                "set_name": set_obj.get("name"),
                "series": set_obj.get("series"),
                "year": year
            }
    except Exception as e:
        print(f"Pokemon TCG API exception: {str(e)}")
        return None


def quality_warnings(img, label, warnings: List[str]):
    """NON-BLOCKING warnings only."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if min(h, w) < 900:
        warnings.append(f"{label}: Low resolution (move closer)")

    bs = blur_score(gray)
    if bs < 70:
        warnings.append(f"{label}: Slight blur risk (tap to focus)")

    over = overexposed_ratio(gray)
    if over > 0.06:
        warnings.append(f"{label}: Glare/overexposure risk")

    under = underexposed_ratio(gray)
    if under > 0.15:
        warnings.append(f"{label}: Too dark")


@app.post("/api/identify")
async def identify(front: UploadFile = File(...)):
    """
    Simple identification - just validates image quality
    User will manually enter card details in WordPress
    """
    fb = await front.read()
    if not fb or len(fb) < 1500:
        raise HTTPException(status_code=400, detail="Front image looks empty/corrupt.")

    front_img = decode_image(fb)
    if front_img is None:
        raise HTTPException(status_code=400, detail="Could not decode front image.")

    warnings: List[str] = []
    quality_warnings(front_img, "Front", warnings)

    # For now, return placeholder data
    # User can manually enter card details in WordPress
    return JSONResponse(content={
        "identify_token": str(uuid.uuid4()),
        "name": "Manual Entry Required",
        "series": "Please enter card details manually",
        "year": "Unknown",
        "number": "",
        "confidence": 0.0,
        "warnings": warnings,
        "version": APP_VERSION,
        "note": "Automatic identification requires manual card entry for now"
    })


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
    confidence = 0.72

    quality_warnings(front_img, "Front", warnings)
    quality_warnings(back_img, "Back", warnings)
    if angle_img is not None:
        quality_warnings(angle_img, "Angled", warnings)
    else:
        warnings.append("Angled: Not provided (recommended for surface/foil assessment)")

    name = "Manual Entry Required"
    series = "Unknown"
    year = "Unknown"

    fq = find_card_quad(front_img)
    bq = find_card_quad(back_img)
    aq = find_card_quad(angle_img) if angle_img is not None else None

    if fq is None or bq is None:
        defects.append("Card boundary detection weak (fill frame, reduce glare).")
        confidence -= 0.08
        return JSONResponse(content={
            "pregrade": "Pre-Assessment: Limited",
            "preapproval": "Pre-Approved — manual review recommended",
            "series": series,
            "year": year,
            "name": name,
            "defects": defects,
            "warnings": warnings,
            "confidence": float(max(0.10, min(0.90, confidence))),
            "subgrades": {"photo_quality": "Warn", "card_detected": "Partial"},
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
        confidence -= 0.05
    if edge_white_back > 0.12:
        defects.append("Back: Edge/corner whitening risk detected")
        confidence -= 0.06

    surface_risk = None
    if angle_img is not None and aq is not None:
        angle_w = warp_card(angle_img, aq)
        angle_g = cv2.cvtColor(angle_w, cv2.COLOR_BGR2GRAY)
        surface_risk = surface_line_risk(angle_g)
        if surface_risk > 0.09:
            defects.append("Angled: Surface scratch / print-line risk detected")
            confidence -= 0.05

    centering_note = "Centering: Review in-hand"
    if (edge_white_front < 0.06) and (edge_white_back < 0.07):
        centering_note = "Centering: Looks acceptable (photo-based)"

    subgrades: Dict[str, Any] = {
        "photo_quality": "Warn" if warnings else "Pass",
        "card_detected": "Yes",
        "edge_whitening_front": round(edge_white_front, 3),
        "edge_whitening_back": round(edge_white_back, 3),
        "centering_note": centering_note
    }
    if surface_risk is not None:
        subgrades["surface_risk"] = round(float(surface_risk), 3)

    if len(defects) == 0:
        preapproval = "Pre-Approved — proceed to submission"
        summary = "Pre-Assessment: Clear"
        confidence += 0.06
    elif len(defects) <= 3:
        preapproval = "Pre-Approved — proceed (minor risks flagged)"
        summary = "Pre-Assessment: Minor Risks"
    else:
        preapproval = "Pre-Approved — manual review required"
        summary = "Pre-Assessment: Review Recommended"
        confidence -= 0.03

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
        "version": APP_VERSION
    })
