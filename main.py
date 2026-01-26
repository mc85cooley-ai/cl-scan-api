from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI(title="Collectors League Scan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "CL Scan API online"}

@app.post("/scan/start")
async def start_scan(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    angle: UploadFile = File(...)
):
    # Mock analysis process
    time.sleep(1)

    return {
        "card_name": "Charizard ex",
        "set": "Scarlet & Violet 151",
        "year": 2023,
        "centering": "Good",
        "surface": "Minor flaws detected",
        "edges": "Clean",
        "pre_approval": "Likely Approved"
    }
