from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
import yaml
import cv2
import numpy as np
from pathlib import Path
import sys
import logging
from typing import List, Optional
from pydantic import BaseModel
import json
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from src.yolo_detector import YOLODetector
from src.cnn_classifier import CNNClassifier

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open("config/server_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="Traffic Sign Detection API",
    description="API for detecting and classifying traffic signs using YOLOv8 and CNN",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['cors']['allow_origins'],
    allow_credentials=True,
    allow_methods=config['cors']['allow_methods'],
    allow_headers=config['cors']['allow_headers'],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize models
try:
    logger.info("Loading YOLO model...")
    yolo_detector = YOLODetector(
        model_path=config['models']['yolo']['path'],
        config_path="config/yolo_config.yaml"
    )
    logger.info("YOLO model loaded successfully!")
    
    logger.info("Loading CNN classifier...")
    # Load number of classes from data.yaml
    with open("data/processed/data.yaml", 'r') as f:
        data_config = yaml.safe_load(f)
    num_classes = len(data_config['names'])
    
    cnn_classifier = CNNClassifier(num_classes=num_classes)
    cnn_classifier.load(config['models']['cnn']['path'])
    logger.info(f"CNN classifier loaded successfully with {num_classes} classes!")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    yolo_detector = None
    cnn_classifier = None

# Pydantic models for API responses
class DetectionResult(BaseModel):
    bbox: List[int]
    confidence: float
    class_id: int
    class_name: str
    cnn_prediction: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    yolo_model_loaded: bool
    cnn_model_loaded: bool
    timestamp: str

class DetectionResponse(BaseModel):
    success: bool
    num_detections: int
    detections: List[DetectionResult]
    image_size: dict
    processing_time: Optional[float] = None
    timestamp: str

class ErrorResponse(BaseModel):
    success: bool
    error: str
    timestamp: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Traffic Sign Detection API - Swagger UI"
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_endpoint():
    return get_openapi(
        title="Traffic Sign Detection API",
        version="1.0.0",
        routes=app.routes
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        yolo_model_loaded=yolo_detector is not None,
        cnn_model_loaded=cnn_classifier is not None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/api/classes")
async def get_classes():
    """Get list of supported traffic sign classes"""
    try:
        with open("config/yolo_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        return JSONResponse(content={
            "success": True,
            "num_classes": len(config['classes']['names']),
            "classes": config['classes']['names'],
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect")
async def detect_traffic_signs(file: UploadFile = File(...)):
    """
    Detect traffic signs in an uploaded image
    
    Args:
        file: Image file (jpg, jpeg, png)
    
    Returns:
        JSON with detection results
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Check file size
        if len(contents) > config['processing']['max_image_size']:
            raise HTTPException(status_code=400, detail="Image size too large")
        
        # Detect with YOLO
        if yolo_detector is None:
            raise HTTPException(status_code=500, detail="YOLO model not loaded")
        
        conf_threshold = config['models']['yolo']['confidence']
        iou_threshold = config['models']['yolo']['iou_threshold']
        
        results = yolo_detector.detect(
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # Classify each detection with CNN (optional refinement)
        enhanced_detections = []
        for det in results['detections']:
            x1, y1, x2, y2 = det['bbox']
            
            # Crop detected region
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size > 0 and cnn_classifier is not None:
                try:
                    # Classify with CNN
                    cnn_result = cnn_classifier.predict(cropped)
                    
                    det['cnn_prediction'] = {
                        'class_id': cnn_result['class_id'],
                        'confidence': cnn_result['confidence']
                    }
                except Exception as e:
                    logger.warning(f"CNN classification failed for detection: {e}")
            
            enhanced_detections.append(det)
        
        return JSONResponse(content={
            "success": True,
            "num_detections": len(enhanced_detections),
            "detections": enhanced_detections,
            "image_size": {
                "height": image.shape[0],
                "width": image.shape[1]
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect_batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    """
    Detect traffic signs in multiple images
    
    Args:
        files: List of image files
    
    Returns:
        JSON with detection results for each image
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for file in files:
        try:
            # Read image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Invalid image file"
                })
                continue
            
            # Detect
            detections = yolo_detector.detect(image)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "detections": detections
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "success": True,
        "total_images": len(files),
        "results": results
    })

@app.get("/api/classes")
async def get_classes():
    """Get list of supported traffic sign classes"""
    try:
        with open("config/yolo_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        return JSONResponse(content={
            "success": True,
            "num_classes": len(config['classes']['names']),
            "classes": config['classes']['names']
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify")
async def classify_sign(file: UploadFile = File(...)):
    """
    Classify a cropped traffic sign image using CNN
    
    Args:
        file: Cropped traffic sign image
    
    Returns:
        Classification result
    """
    try:
        if cnn_classifier is None:
            raise HTTPException(status_code=500, detail="CNN model not loaded")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Classify
        result = cnn_classifier.predict(image)
        
        return JSONResponse(content={
            "success": True,
            "prediction": result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    server_config = config['server']
    uvicorn.run(
        "app:app",
        host=server_config['host'],
        port=server_config['port'],
        reload=server_config['reload']
    )