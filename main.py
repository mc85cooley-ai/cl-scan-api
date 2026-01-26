from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Collectors League Scan API")

@app.get("/")
def root():
    return {"status": "ok", "service": "cl-scan-api"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/verify")
async def verify(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    angle: UploadFile = File(...),
    x_clis_key: str | None = Header(default=None),
):
    # OPTIONAL: if you want to enforce an API key later:
    # expected = os.getenv("CLIS_API_KEY")
    # if expected and x_clis_key != expected:
    #     raise HTTPException(status_code=401, detail="Invalid API key")

    # Read bytes (so you know upload is working)
    front_bytes = await front.read()
    back_bytes  = await back.read()
    angle_bytes = await angle.read()

    # Basic sanity check (prevents empty uploads)
    if len(front_bytes) < 1000 or len(back_bytes) < 1000 or len(angle_bytes) < 1000:
        raise HTTPException(status_code=400, detail="One or more images look empty/corrupt")

    # TODO: Replace with real analysis
    result = {
        "pregrade": "Estimated Grade: 9.0 (Mint)",
        "preapproval": "Pre-Approved — pending in-hand review",
        "series": "Unknown (Auto-ID next)",
        "year": "Unknown",
        "name": "Unknown item/card"
    }

    return JSONResponse(content=result)
