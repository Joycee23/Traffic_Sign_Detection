"""
FastAPI server with Swagger UI for Traffic Sign Detection API
YOLOv8 Detection + CNN Classification
FULL & FIXED VERSION
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import uvicorn
import yaml
import cv2
import numpy as np
from pathlib import Path
import sys
import logging
import time

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

# =====================================================
# FIX PYTHON PATH
# =====================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# =====================================================
# IMPORT MODELS
# =====================================================
from src.yolo_detector import YOLODetector
from src.cnn_classifier import CNNClassifier

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrafficSignAPI")

# =====================================================
# LOAD CONFIG
# =====================================================
with open("config/server_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# =====================================================
# FASTAPI INIT
# =====================================================
app = FastAPI(
    title="Traffic Sign Detection API",
    description="YOLOv8 + CNN Traffic Sign Detection",
    version="1.0.0"
)

# =====================================================
# CORS
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["cors"]["allow_origins"],
    allow_credentials=True,
    allow_methods=config["cors"]["allow_methods"],
    allow_headers=config["cors"]["allow_headers"],
)

# =====================================================
# STATIC + TEMPLATE (OPTIONAL)
# =====================================================
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# =====================================================
# LOAD MODELS
# =====================================================
try:
    logger.info("Loading YOLO model...")
    yolo_detector = YOLODetector(
        model_path=config["models"]["yolo"]["path"]
    )
    logger.info("YOLO loaded successfully")

    logger.info("Loading CNN model...")
    with open("data/processed/data.yaml", "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    num_classes = len(data_cfg["names"])
    cnn_classifier = CNNClassifier(num_classes=num_classes)
    cnn_classifier.load(config["models"]["cnn"]["path"])
    logger.info("CNN loaded successfully")

except Exception as e:
    logger.error(f"MODEL LOAD FAILED: {e}")
    yolo_detector = None
    cnn_classifier = None

# =====================================================
# SCHEMAS
# =====================================================
class DetectionResult(BaseModel):
    bbox: List[int]                 # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    cnn_prediction: Optional[Dict[str, Any]] = None


class DetectionResponse(BaseModel):
    success: bool
    num_detections: int
    detections: List[DetectionResult]
    image_size: Dict[str, int]
    processing_time: float
    timestamp: str

# =====================================================
# ROUTES
# =====================================================
@app.get("/")
async def home():
    return {"message": "Traffic Sign Detection API is running 🚦"}

# -----------------------------------------------------
@app.post("/api/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    start_time = time.time()

    # ----- CHECK FILE -----
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if yolo_detector is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")

    # ----- YOLO DETECT -----
    results = yolo_detector.detect(image)
    detections = []

    h, w = image.shape[:2]

    for det in results["detections"]:
        x1, y1, x2, y2 = det["bbox"]

        # CLIP BBOX (FIX BUG)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = image[y1:y2, x1:x2]

        # CNN CLASSIFICATION
        if crop.size > 0 and cnn_classifier is not None:
            cnn_result = cnn_classifier.predict(crop)
            det["cnn_prediction"] = cnn_result
        else:
            det["cnn_prediction"] = None

        detections.append(det)

    return DetectionResponse(
        success=True,
        num_detections=len(detections),
        detections=detections,
        image_size={"h": h, "w": w},
        processing_time=round(time.time() - start_time, 3),
        timestamp=datetime.now().isoformat()
    )

# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=config["server"]["host"],
        port=config["server"]["port"],
        reload=config["server"]["reload"]
    )
