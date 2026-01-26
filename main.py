from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2

app = FastAPI(title="Collectors League Scan API")

@app.get("/health")
def health():
    return {"ok": True}

def decode_image(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def blur_score(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def overexposed_ratio(gray):
    return float((gray >= 245).mean())

def underexposed_ratio(gray):
    return float((gray <= 10).mean())

def find_card_quad(img_bgr):
    """Try to find a 4-point contour that looks like a card."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    h, w = img_bgr.shape[:2]
    img_area = h * w

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.08:  # too small
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            return pts
    return None

def order_points(pts):
    # Standard order: tl, tr, br, bl
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def warp_card(img, quad, out_w=900, out_h=1260):
    pts = order_points(quad)
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h))
    return warped

def whitening_risk_edge(gray_warped):
    """Very rough whitening/chipping risk along the outer edge band."""
    h, w = gray_warped.shape[:2]
    band = 10  # pixels
    top = gray_warped[0:band, :]
    bot = gray_warped[h-band:h, :]
    left = gray_warped[:, 0:band]
    right = gray_warped[:, w-band:w]
    ring = np.concatenate([top.flatten(), bot.flatten(), left.flatten(), right.flatten()])
    # ratio of near-white pixels
    return float((ring >= 235).mean())

def surface_line_risk(gray_angle_warped):
    """High-frequency line energy proxy for scratch/print-line risk."""
    blur = cv2.GaussianBlur(gray_angle_warped, (0,0), 2.0)
    high = cv2.absdiff(gray_angle_warped, blur)
    denom = max(1.0, float(gray_angle_warped.mean()))
    return float(high.mean() / denom)

@app.post("/api/verify")
async def verify(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    angle: UploadFile = File(...),
):
    fb = await front.read()
    bb = await back.read()
    ab = await angle.read()

    if min(len(fb), len(bb), len(ab)) < 1500:
        raise HTTPException(status_code=400, detail="One or more images look empty/corrupt.")

    front_img = decode_image(fb)
    back_img  = decode_image(bb)
    angle_img = decode_image(ab)

    if front_img is None or back_img is None or angle_img is None:
        raise HTTPException(status_code=400, detail="Could not decode one or more images.")

    defects = []
    notes = []
    confidence = 0.60

    # ---------- Photo quality checks ----------
    def assess_quality(img, label):
        nonlocal confidence
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if min(h, w) < 900:
            defects.append(f"{label}: Low resolution (move closer / higher quality)")
            confidence -= 0.10

        bs = blur_score(gray)
        if bs < 80:
            defects.append(f"{label}: Blurry / out of focus")
            confidence -= 0.12

        over = overexposed_ratio(gray)
        if over > 0.06:
            defects.append(f"{label}: Glare/overexposure risk (reduce reflections)")
            confidence -= 0.08

        under = underexposed_ratio(gray)
        if under > 0.15:
            defects.append(f"{label}: Too dark (increase lighting)")
            confidence -= 0.06

        return gray

    assess_quality(front_img, "Front")
    assess_quality(back_img,  "Back")
    assess_quality(angle_img, "Angled")

    # If quality is very poor, stop early with rescan required
    critical = any(("Low resolution" in d) or ("Blurry" in d) for d in defects)
    if critical:
        return JSONResponse(content={
            "pregrade": "Pre-Assessment: Rescan Required",
            "preapproval": "Not pre-approved — photo quality insufficient",
            "series": "Unknown",
            "year": "Unknown",
            "name": "Unknown item/card",
            "defects": defects,
            "confidence": max(0.10, min(0.90, confidence)),
            "subgrades": {}
        })

    # ---------- Card detection / warp ----------
    fq = find_card_quad(front_img)
    bq = find_card_quad(back_img)
    aq = find_card_quad(angle_img)

    if fq is None or bq is None or aq is None:
        defects.append("Card boundary not clearly detected (ensure all 4 corners visible, fill frame, reduce glare)")
        confidence -= 0.15

        return JSONResponse(content={
            "pregrade": "Pre-Assessment: Needs Rescan",
            "preapproval": "Not pre-approved — card framing insufficient",
            "series": "Unknown",
            "year": "Unknown",
            "name": "Unknown item/card",
            "defects": defects,
            "confidence": max(0.10, min(0.90, confidence)),
            "subgrades": {}
        })

    front_w = warp_card(front_img, fq)
    back_w  = warp_card(back_img, bq)
    angle_w = warp_card(angle_img, aq)

    front_g = cv2.cvtCol_
