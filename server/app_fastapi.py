"""
FastAPI server with Swagger UI for Traffic Sign Detection API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import yaml
import cv2
import numpy as np
from pathlib import Path
import sys
import logging
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import time

# ================= PATH =================
sys.path.append('.')

from src.yolo_detector import YOLODetector
from src.cnn_classifier import CNNClassifier

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= LOAD CONFIG =================
with open("config/server_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Safe max image size (fallback 5MB)
MAX_IMAGE_SIZE = config.get("processing", {}).get(
    "max_image_size", 5 * 1024 * 1024
)

# ================= FASTAPI APP =================
app = FastAPI(
    title="Traffic Sign Detection API",
    description="YOLOv8 + CNN traffic sign detection system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["cors"]["allow_origins"],
    allow_credentials=True,
    allow_methods=config["cors"]["allow_methods"],
    allow_headers=config["cors"]["allow_headers"],
)

# ================= STATIC & TEMPLATES =================
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ================= LOAD MODELS =================
try:
    logger.info("Loading YOLO model...")
    yolo_detector = YOLODetector(
        model_path=config["models"]["yolo"]["path"],
        config_path="config/yolo_config.yaml"
    )
    logger.info("YOLO model loaded!")

    logger.info("Loading CNN classifier...")
    with open("data/processed/data.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)

    num_classes = len(data_cfg["names"])
    cnn_classifier = CNNClassifier(num_classes)
    cnn_classifier.load(config["models"]["cnn"]["path"])

    logger.info(f"CNN loaded with {num_classes} classes")

except Exception as e:
    logger.exception("Failed to load models")
    yolo_detector = None
    cnn_classifier = None

# ================= SCHEMAS =================
class DetectionResult(BaseModel):
    bbox: List[int]
    confidence: float
    class_id: int
    class_name: str
    cnn_prediction: Optional[dict] = None

class DetectionResponse(BaseModel):
    success: bool
    num_detections: int
    detections: List[DetectionResult]
    image_size: dict
    processing_time: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    yolo_model_loaded: bool
    cnn_model_loaded: bool
    timestamp: str
    version: str = "1.0.0"

class ClassResponse(BaseModel):
    success: bool = True
    num_classes: int
    classes: dict
    timestamp: str

# ================= ROUTES =================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        yolo_model_loaded=yolo_detector is not None,
        cnn_model_loaded=cnn_classifier is not None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/api/classes", response_model=ClassResponse)
async def get_classes():
    try:
        with open("config/yolo_config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        return ClassResponse(
            num_classes=len(cfg["classes"]["names"]),
            classes=cfg["classes"]["names"],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.exception("Failed to load classes")
        raise HTTPException(status_code=500, detail="Cannot load class list")

@app.post("/api/detect", response_model=DetectionResponse)
async def detect_traffic_signs(file: UploadFile = File(...)):
    start_time = time.time()

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()

    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Image size too large")

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if yolo_detector is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")

    try:
        results = yolo_detector.detect(
            image,
            conf_threshold=config["models"]["yolo"]["confidence"],
            iou_threshold=config["models"]["yolo"]["iou_threshold"]
        )

        detections = []

        for det in results["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            crop = image[y1:y2, x1:x2]

            if crop.size > 0 and cnn_classifier:
                cnn_res = cnn_classifier.predict(crop)
                det["cnn_prediction"] = cnn_res

            detections.append(det)

        return DetectionResponse(
            success=True,
            num_detections=len(detections),
            detections=detections,
            image_size={
                "height": image.shape[0],
                "width": image.shape[1]
            },
            processing_time=round(time.time() - start_time, 3),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.exception("Detection failed")
        raise HTTPException(status_code=500, detail="Detection error")

@app.post("/api/classify")
async def classify_sign(file: UploadFile = File(...)):
    if cnn_classifier is None:
        raise HTTPException(status_code=500, detail="CNN model not loaded")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        result = cnn_classifier.predict(image)
        return JSONResponse({
            "success": True,
            "prediction": result,
            "timestamp": datetime.now().isoformat()
        })
    except Exception:
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail="Classification error")

# ================= RUN =================
if __name__ == "__main__":
    uvicorn.run(
        "app_fastapi:app",
        host=config["server"]["host"],
        port=config["server"]["port"],
        reload=config["server"]["reload"]
    )
