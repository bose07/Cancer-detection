from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import pathlib
import uvicorn
from transformers import pipeline  # type: ignore
from huggingface_hub import hf_hub_download  # Import for downloading the model
from keras.models import load_model  # Import for loading the Keras model
import os  # Ensure os is imported for environment variable access
import logging  # Import for logging errors
from dotenv import load_dotenv  # Import for loading environment variables
from functools import lru_cache
import traceback
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load environment variables from .env file
load_dotenv()
import numpy as np
from datetime import datetime
from reportlab.pdfgen import canvas
import io
from reportlab.lib.utils import ImageReader  # Import ImageReader for handling images
from reportlab.lib.pagesizes import letter  # Import letter for PDF page size

# Create lifespan context
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up...")
    try:
        # Create directories
        static_dir = pathlib.Path("static")
        static_dir.mkdir(exist_ok=True)
        temp_dir = static_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Make temp_dir available to routes
        app.state.temp_dir = temp_dir
        
        # Pre-load model on startup
        if Config.IS_PRODUCTION:
            logger.info("Production environment detected, pre-loading model...")
            try:
                model = get_model()
                logger.info("Model pre-loaded successfully")
            except Exception as e:
                logger.error(f"Failed to pre-load model: {str(e)}")
                # Don't raise the error - let the app start anyway
                
        yield
    finally:
        # Shutdown
        logger.info("Shutting down...")
        # Cleanup temp files
        for temp_file in temp_dir.glob("*.jpg"):
            temp_file.unlink(missing_ok=True)

app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration
class Config:
    MODEL_REPO = os.getenv("MODEL_REPO", "schandel08/cancer_prediction_spc_v0")
    MODEL_FILENAME = os.getenv("MODEL_FILENAME", "DSnet_cancer_prediction.keras")
    HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
    API_URL = os.getenv("API_URL", "http://localhost:8000")
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    IS_PRODUCTION = os.getenv("RENDER", "false").lower() == "true"
    MODELS_CACHE_DIR = "/opt/render/project/models" if IS_PRODUCTION else "./models"

# Use configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "https://your-frontend-domain.com",  # Add your frontend domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 
from keras.preprocessing.image import load_img, img_to_array

security = HTTPBearer()

@lru_cache()
def get_model():
    try:
        logger.info("Loading model from Hugging Face Hub...")
        
        # Get HF token from environment
        hf_token = Config.HF_ACCESS_TOKEN
        if not hf_token:
            raise ValueError("HF_ACCESS_TOKEN is not set in environment variables")
            
        # Use persistent storage path on Render
        os.makedirs(Config.MODELS_CACHE_DIR, exist_ok=True)
        
        logger.info("Attempting to download model...")
        model_path = hf_hub_download(
            repo_id=Config.MODEL_REPO,
            filename=Config.MODEL_FILENAME,
            cache_dir=Config.MODELS_CACHE_DIR,
            token=hf_token,
            force_download=not Config.IS_PRODUCTION  # Only force download in development
        )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        logger.info(f"Model path: {model_path}")
        
        # Load the model with error handling
        try:
            model = load_model(model_path)
            if not model:
                raise ValueError("Model loaded as None")
        except Exception as model_error:
            logger.error(f"Error loading model: {str(model_error)}")
            raise RuntimeError(f"Failed to load model from path: {str(model_error)}")
            
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Model loading failed: {str(e)}")

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        logger.info("Starting prediction request...")
        
        # Verify token
        token = credentials.credentials
        logger.info("Checking authentication token...")
        if token != Config.HF_ACCESS_TOKEN:
            logger.error("Invalid authentication token")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Authentication Error",
                    "message": "Invalid token",
                    "status": "failed"
                }
            )

        # Log request details
        logger.info(f"File received: {file.filename}, Content-Type: {file.content_type}")
        
        # Get cached model with detailed logging
        try:
            logger.info("Loading model...")
            model = get_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Model Loading Error",
                    "message": str(e),
                    "status": "failed"
                }
            )

        # Process image
        try:
            content = await file.read()
            logger.info(f"Read file content, size: {len(content)} bytes")
            
            temp_path = app.state.temp_dir / "temp_image.jpg"
            with open(temp_path, "wb") as temp_file:
                temp_file.write(content)
            logger.info(f"Saved temporary file to: {temp_path}")

            image = load_img(temp_path, target_size=(380, 380))
            image_array = img_to_array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            logger.info(f"Preprocessed image shape: {image_array.shape}")

        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Image Processing Error",
                    "message": str(e),
                    "status": "failed"
                }
            )

        # Make prediction
        try:
            logger.info("Making prediction...")
            predictions = model.predict(image_array)
            logger.info(f"Raw predictions: {predictions}")
            
            class_label = "Cancer" if predictions[0][0] < 0.5 else "Normal"
            confidence = float(abs(predictions[0][0] - 0.5) * 2 * 100)
            
            logger.info(f"Prediction result: {class_label} with confidence: {confidence}%")
            
            return JSONResponse(
                content={
                    "prediction": class_label,
                    "confidence": f"{confidence:.2f}",
                    "status": "success"
                }
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Prediction Error",
                    "message": str(e),
                    "status": "failed"
                }
            )
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": str(e),
                "status": "failed"
            }
        )

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", "r") as f:
        html_content = f.read()
        # Inject the token into the HTML
        html_content = html_content.replace(
            '</head>',
            f'<script>const HF_ACCESS_TOKEN = "{Config.HF_ACCESS_TOKEN}";</script></head>'
        )
        return html_content

def generate_pdf_report(image_path, prediction, confidence, patient_name, age, gender, phone_number):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add hospital header - centered
    c.setFont("Helvetica-Bold", 20)
    title = "Gastrointestinal Cancer Detection"
    title_width = c.stringWidth(title, "Helvetica-Bold", 20)
    c.drawString((width - title_width) / 2, height - 40, title)
    
    c.setFont("Helvetica", 10)
    subtitle = "Advanced AI-Powered Diagnostic System"
    subtitle_width = c.stringWidth(subtitle, "Helvetica", 10)
    c.drawString((width - subtitle_width) / 2, height - 55, subtitle)
    
    # Add horizontal line
    c.line(50, height - 65, width - 50, height - 65)

    # Add report title - centered
    c.setFont("Helvetica-Bold", 14)
    report_title = "Detection Report"
    report_width = c.stringWidth(report_title, "Helvetica-Bold", 14)
    c.drawString((width - report_width) / 2, height - 85, report_title)
    
    # Add report date and time only
    c.setFont("Helvetica", 9)
    c.drawString(width - 180, height - 85, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    c.drawString(width - 180, height - 100, f"Time: {datetime.now().strftime('%H:%M:%S')}")

    # Add patient information section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 130, "Patient Information")
    
    # Add patient details in a box
    c.rect(50, height - 190, width - 100, 40)
    c.setFont("Helvetica", 12)
    c.drawString(60, height - 160, f"Name: {patient_name}")
    c.drawString(60, height - 175, f"Age: {age} years")
    c.drawString(300, height - 160, f"Gender: {gender}")
    c.drawString(300, height - 175, f"Contact: {phone_number}")
    
    # Add examination details
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 220, "Examination Details")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 240, "Type: Gastrointestinal Cancer Screening")
    c.drawString(50, height - 255, "Method: AI-Assisted Image Analysis")

    # Add the analysis image with border
    c.rect(50, height - 475, 230, 200)
    img = ImageReader(image_path)
    c.drawImage(img, 55, height - 470, width=220, height=190)
    c.setFont("Helvetica", 9)
    c.drawString(50, height - 485, "Diagnostic Image")

    # Add analysis results in a highlighted box
    c.setFillColorRGB(0.95, 0.95, 0.95)
    c.rect(300, height - 475, width - 350, 200, fill=True)
    c.setFillColorRGB(0, 0, 0)
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(310, height - 295, "Analysis Results")
    c.setFont("Helvetica", 10)
    
    result_color = (1, 0, 0) if prediction.lower() == 'cancer' else (0, 0.5, 0)
    c.setFillColorRGB(*result_color)
    c.drawString(310, height - 315, f"Diagnosis: {prediction}")
    c.setFillColorRGB(0, 0, 0)
    c.drawString(310, height - 335, f"Confidence Level: {confidence}%")

    # Add recommendations section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 505, "Medical Recommendations")
    c.setFont("Helvetica", 9)  # Regular font for normal text
    y = height - 525

    if prediction.lower() == 'cancer':
        recommendations = [
            ("1. Immediate Consultation Required:", True),  # (text, is_bold)
            ("   • Schedule an appointment with a gastroenterologist within 48 hours", False),
            ("   • Bring this report and any previous medical records", False),
            ("2. Additional Testing Recommended:", True),
            ("   • Endoscopic examination", False),
            ("   • Tissue biopsy", False),
            ("   • Complete blood count (CBC)", False),
            ("3. Precautionary Measures:", True),
            ("   • Maintain detailed symptom diary", False),
            ("   • Follow prescribed dietary restrictions", False),
            ("   • Avoid irritants and specific foods as advised", False)
        ]
    else:
        recommendations = [
            ("1. Preventive Care:", True),
            ("   • Schedule regular annual screenings", False),
            ("   • Maintain healthy dietary habits", False),
            ("2. Lifestyle Recommendations:", True),
            ("   • Regular exercise (30 minutes daily)", False),
            ("   • Balanced diet rich in fiber", False),
            ("   • Adequate hydration (8 glasses of water daily)", False),
            ("3. Follow-up:", True),
            ("   • Regular check-ups as per schedule", False),
            ("   • Report any new symptoms promptly", False),
            ("   • Maintain healthy lifestyle habits", False)
        ]

    for text, is_bold in recommendations:
        if is_bold:
            c.setFont("Helvetica-Bold", 9)  # Bold font for headers
        else:
            c.setFont("Helvetica", 9)  # Regular font for bullet points
        c.drawString(50, y, text)
        y -= 15  # Reduced spacing between lines

    # Add footer with disclaimer and page number
    c.line(50, 80, width - 50, 80)
    c.setFont("Helvetica", 7)  # Smaller font for footer
    c.drawString(50, 65, "CONFIDENTIAL MEDICAL REPORT")
    c.drawString(50, 50, "This report is generated using advanced AI technology and should be reviewed by a qualified healthcare professional.")
    c.drawString(width - 100, 50, "Page 1 of 1")

    c.save()
    buffer.seek(0)
    return buffer 

@app.post("/generate-report/")
async def generate_report(
    file: UploadFile = File(...),
    patientName: str = Form(default=""),
    age: str = Form(default=""),
    gender: str = Form(default=""),
    phoneNumber: str = Form(default=""),
    prediction: str = Form(default=""),
    confidence: str = Form(default=""),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        # Verify token
        token = credentials.credentials
        if token != Config.HF_ACCESS_TOKEN:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Authentication Error",
                    "message": "Invalid token",
                    "status": "failed"
                }
            )

        # Call the generate_pdf_report function
        image_path = app.state.temp_dir / "temp_image.jpg"
        pdf_buffer = generate_pdf_report(image_path, prediction, confidence, patientName, age, gender, phoneNumber)
        
        return StreamingResponse(
            pdf_buffer, 
            media_type='application/pdf',
            headers={"Content-Disposition": "attachment; filename=report.pdf"}
        )
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "error": "Report Generation Error",
                "message": str(e),
                "status": "failed"
            }
        )

@app.get("/health")
async def health_check():
    try:
        # Test model loading
        model = get_model()
        logger.info("Health check: Model loaded successfully")
        
        # Test environment configuration
        config_status = {
            "hf_token": bool(Config.HF_ACCESS_TOKEN),
            "model_repo": bool(Config.MODEL_REPO),
            "model_filename": bool(Config.MODEL_FILENAME),
        }
        logger.info(f"Health check: Configuration status - {config_status}")
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "api_version": "1.0",
            "config_status": config_status,
            "server_time": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "config_status": {
                    "hf_token": bool(Config.HF_ACCESS_TOKEN),
                    "model_repo": bool(Config.MODEL_REPO),
                    "model_filename": bool(Config.MODEL_FILENAME),
                }
            }
        )

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Unhandled error: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "status": "failed"
                }
            )

# Add the middleware
app.add_middleware(ErrorHandlerMiddleware)
