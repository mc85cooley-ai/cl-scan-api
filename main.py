# main.py (FULL FILE) — FIXED for OpenAI Responses API
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
import httpx
import base64
import os
import json
import uuid

app = FastAPI(title="Collectors League Scan API")

APP_VERSION = os.getenv("CL_SCAN_VERSION", "2026-01-27-responses-api-fixed")

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


def normalize_number(num):
    if not num:
        return None
    s = str(num).strip().replace(" ", "").replace("\\", "/")
    return s


async def openai_autoid_pokemon(front_bgr) -> dict:
    """
    FIXED: Use correct Responses API format with proper image_url field
    Return dict in schema:
    {name, number, set_name, confidence, notes}
    Or {error: True, ...}
    """
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
        "- Prefer exact printed card name.\n"
        "- number should look like '006/165' if visible.\n"
        "- set_name should be the set title if visible (e.g. 'Scarlet & Violet—151').\n"
    )

    # FIXED: Use correct format for Responses API
    payload = {
        "model": OPENAI_MODEL,
        "instructions": instructions,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Identify this Pokémon card. Return JSON only."},
                {
                    "type": "input_image",
                    # FIXED: Use image_url with data URI, not image_base64
                    "image_url": f"data:image/jpeg;base64,{img_b64}"
                },
            ]
        }],
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json=payload
            )

        if r.status_code < 200 or r.status_code >= 300:
            error_body = r.text[:800] if r.text else "No response body"
            return {
                "error": True,
                "status": r.status_code,
                "body": error_body,
                "reason": f"openai_http_{r.status_code}"
            }

        data = r.json()
        
        # Try to extract output_text from various possible locations
        text = None
        
        # Method 1: Direct output_text field
        if "output_text" in data and isinstance(data["output_text"], str):
            text = data["output_text"]
        
        # Method 2: output array
        elif "output" in data and isinstance(data["output"], list):
            for item in data["output"]:
                if isinstance(item, dict):
                    # Check for message type
                    if item.get("type") == "message":
                        content = item.get("content", [])
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and "text" in c:
                                    text = c["text"]
                                    break
                    # Check for direct content
                    elif "content" in item:
                        content = item["content"]
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and "text" in c:
                                    text = c["text"]
                                    break
                        elif isinstance(content, str):
                            text = content
                if text:
                    break
        
        if not text:
            return {
                "error": True,
                "reason": "no_output_text",
                "raw": str(data)[:800]
            }

        # Parse JSON from text
        text = text.strip()
        
        try:
            result = json.loads(text)
            
            # Validate structure
            if not isinstance(result, dict):
                return {
                    "error": True,
                    "reason": "invalid_json_structure",
                    "text": text[:800]
                }
            
            # Ensure confidence is a number
            if "confidence" in result:
                try:
                    result["confidence"] = float(result["confidence"])
                except (ValueError, TypeError):
                    result["confidence"] = 0.5
            else:
                result["confidence"] = 0.5
            
            return result
            
        except json.JSONDecodeError:
            # Try to extract JSON from text (in case wrapped in markdown)
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    result = json.loads(text[start:end+1])
                    if "confidence" not in result:
                        result["confidence"] = 0.5
                    return result
                except:
                    pass
            
            return {
                "error": True,
                "reason": "json_parse_failed",
                "text": text[:800]
            }
    
    except httpx.TimeoutException:
        return {
            "error": True,
            "reason": "timeout",
            "message": "OpenAI API request timed out"
        }
    except httpx.RequestError as e:
        return {
            "error": True,
            "reason": "request_error",
            "message": str(e)
        }
    except Exception as e:
        return {
            "error": True,
            "reason": "exception",
            "message": str(e)
        }


async def autoid_best(front_raw, front_warped=None) -> Optional[dict]:
    """
    Try warped then raw; return best non-error result by confidence.
    """
    best = None
    for img in [front_warped, front_raw]:
        if img is None:
            continue
        res = await openai_autoid_pokemon(img)
        if isinstance(res, dict) and not res.get("error"):
            try:
                conf = float(res.get("confidence") or 0.0)
            except Exception:
                conf = 0.0
            if best is None:
                best = res
            else:
                try:
                    best_conf = float(best.get("confidence") or 0.0)
                except Exception:
                    best_conf = 0.0
                if conf > best_conf:
                    best = res
    return best


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

    try:
        async with httpx.AsyncClient(timeout=25) as client:
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
    except Exception:
        return None


def quality_warnings(img, label, warnings: List[str]):
    """
    NON-BLOCKING warnings only.
    """
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


def build_identity_from_auto(auto_best: Optional[dict], tcg: Optional[dict]) -> Dict[str, Any]:
    series = "Unknown"
    year = "Unknown"
    name = "Unknown card"
    number = ""

    id_conf = None
    if isinstance(auto_best, dict):
        try:
            id_conf = float(auto_best.get("confidence")) if auto_best.get("confidence") is not None else None
        except Exception:
            id_conf = None

    if tcg:
        name = tcg.get("name") or name
        year = tcg.get("year") or year
        series = tcg.get("series") or tcg.get("set_name") or series
        number = tcg.get("number") or number
    else:
        if isinstance(auto_best, dict):
            if auto_best.get("name"):
                name = str(auto_best.get("name"))
            if auto_best.get("set_name"):
                series = str(auto_best.get("set_name"))
            if auto_best.get("number"):
                number = str(auto_best.get("number"))

    return {
        "name": name,
        "series": series,
        "year": year,
        "number": number,
        "id_confidence": id_conf
    }


@app.post("/api/identify")
async def identify(front: UploadFile = File(...)):
    fb = await front.read()
    if not fb or len(fb) < 1500:
        raise HTTPException(status_code=400, detail="Front image looks empty/corrupt.")

    front_img = decode_image(fb)
    if front_img is None:
        raise HTTPException(status_code=400, detail="Could not decode front image.")

    warnings: List[str] = []
    quality_warnings(front_img, "Front", warnings)

    fq = find_card_quad(front_img)
    front_warp_id = warp_card(front_img, fq, out_w=1400, out_h=1960) if fq is not None else None

    auto_best = await autoid_best(front_img, front_warp_id)

    # Check for OpenAI errors
    if isinstance(auto_best, dict) and auto_best.get("error"):
        error_reason = auto_best.get("reason", "unknown")
        error_msg = auto_best.get("message", "") or auto_best.get("body", "")
        
        # Log for debugging
        print(f"OpenAI Error: {error_reason} - {error_msg}")
        
        # Add warning but continue (allow fallback to TCG lookup)
        warnings.append(f"Auto-ID error: {error_reason}")
        
        # If it's a critical error (missing key, auth failure), return error
        if error_reason in ["missing_openai_key", "openai_http_401", "openai_http_403"]:
            raise HTTPException(
                status_code=500, 
                detail=f"OpenAI API configuration error: {error_reason}. Check OPENAI_API_KEY environment variable."
            )
        
        # For other errors, continue with empty auto_best
        auto_best = None

    card_name = None
    card_number = None
    set_name = None
    if isinstance(auto_best, dict):
        card_name = auto_best.get("name")
        card_number = normalize_number(auto_best.get("number"))
        set_name = auto_best.get("set_name")

    tcg = await pokemontcg_lookup(card_name, card_number, set_name) if card_name else None
    ident = build_identity_from_auto(auto_best, tcg)

    # if we still can't identify, be explicit
    if ident["name"] == "Unknown card" and not tcg:
        warnings.append("Could not identify card. Ensure name/number are clearly visible with good lighting.")

    identify_token = str(uuid.uuid4())

    # Confidence: prefer OpenAI confidence if available; else use TCG lookup confidence
    conf = 0.0
    if ident.get("id_confidence") is not None:
        conf = float(max(0.0, min(1.0, ident["id_confidence"])))
    elif tcg:
        conf = 0.85

    return JSONResponse(content={
        "identify_token": identify_token,
        "name": ident["name"],
        "series": ident["series"],
        "year": ident["year"],
        "number": ident["number"],
        "confidence": conf,
        "warnings": warnings,
        "version": APP_VERSION,
        "debug_openai": (auto_best if isinstance(auto_best, dict) and auto_best.get("error") else None)
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

    # Non-blocking warnings
    quality_warnings(front_img, "Front", warnings)
    quality_warnings(back_img, "Back", warnings)
    if angle_img is not None:
        quality_warnings(angle_img, "Angled", warnings)
    else:
        warnings.append("Angled: Not provided (recommended for surface/foil assessment)")

    # Identity (best-effort, but do NOT hard-fail assessment)
    fq = find_card_quad(front_img)
    front_warp_id = warp_card(front_img, fq, out_w=1400, out_h=1960) if fq is not None else None
    auto_best = await autoid_best(front_img, front_warp_id)

    series = "Unknown"
    year = "Unknown"
    name = "Unknown card"

    card_name = None
    card_number = None
    set_name = None
    id_conf = None

    if isinstance(auto_best, dict) and not auto_best.get("error"):
        card_name = auto_best.get("name")
        card_number = normalize_number(auto_best.get("number"))
        set_name = auto_best.get("set_name")
        try:
            id_conf = float(auto_best.get("confidence")) if auto_best.get("confidence") is not None else None
        except Exception:
            id_conf = None

    tcg = await pokemontcg_lookup(card_name, card_number, set_name) if card_name else None
    ident = build_identity_from_auto(auto_best, tcg)

    name = ident["name"]
    series = ident["series"]
    year = ident["year"]

    if id_conf is not None and id_conf < 0.45:
        warnings.append("Auto-ID: Low confidence (try closer / less glare)")

    # Card detection / warp for assessment
    bq = find_card_quad(back_img)
    aq = find_card_quad(angle_img) if angle_img is not None else None

    if fq is None or bq is None:
        defects.append("Card boundary detection weak (fill frame, reduce glare). Assessment may be inaccurate.")
        confidence -= 0.08
        return JSONResponse(content={
            "pregrade": "Pre-Assessment: Limited (No Warp)",
            "preapproval": "Pre-Approved — manual review recommended (framing/edge data limited)",
            "series": series,
            "year": year,
            "name": name,
            "defects": defects,
            "warnings": warnings,
            "confidence": float(max(0.10, min(0.90, confidence))),
            "subgrades": {"photo_quality": "Warn", "card_detected": "Partial"},
            "version": APP_VERSION,
            "debug_openai": (auto_best if isinstance(auto_best, dict) and auto_best.get("error") else None)
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
    elif angle_img is not None and aq is None:
        warnings.append("Angled: Could not detect card boundary (keep corners visible / reduce glare)")

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
        preapproval = "Pre-Approved — proceed to submission (final assessment in-hand)"
        summary = "Pre-Assessment: Clear"
        confidence += 0.06
    elif len(defects) <= 3:
        preapproval = "Pre-Approved — proceed (minor risks flagged)"
        summary = "Pre-Assessment: Minor Risks"
    else:
        preapproval = "Pre-Approved — manual review required (multiple risks flagged)"
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
        "version": APP_VERSION,
        "debug_openai": (auto_best if isinstance(auto_best, dict) and auto_best.get("error") else None)
    })
