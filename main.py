from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import pathlib
import uvicorn
from keras.models import load_model  # type: ignore
from keras.preprocessing.image import load_img, img_to_array  # type: ignore
import numpy as np
import os
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image
import io
from reportlab.lib.utils import ImageReader

# Create lifespan context
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")
    # Create directories
    static_dir = pathlib.Path("static")
    static_dir.mkdir(exist_ok=True)
    temp_dir = static_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Make temp_dir available to routes
    app.state.temp_dir = temp_dir
    
    yield
    
    # Shutdown
    print("Shutting down...")
    # Cleanup temp files
    for temp_file in temp_dir.glob("*.jpg"):
        temp_file.unlink(missing_ok=True)

app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", "r") as f:
        return f.read()

# Define a prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load the model
    model = load_model("model\DSnet_cancer_prediction.keras")

    # Read the uploaded image
    content = await file.read()
    temp_path = app.state.temp_dir / "temp_image.jpg"
    with open(temp_path, "wb") as temp_file:
        temp_file.write(content)

    # Preprocess the image - Changed target size to 380x380
    image = load_img(str(temp_path), target_size=(380, 380))
    image_array = img_to_array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)

    # Make a prediction
    prediction = model.predict(image_array)
    confidence = float(abs(prediction[0][0] - 0.5) * 2 * 100)  # Convert to percentage
    class_label = "Cancer" if prediction[0][0] < 0.5 else "Normal"
    confidence_rounded = round(confidence, 2)

    # Clean up temporary file
    if temp_path.exists():
        temp_path.unlink()

    return {
        "prediction": class_label,
        "confidence": confidence_rounded
    }

@app.post("/generate-report/")
async def generate_report(
    file: UploadFile = File(...),
    patientName: str = Form(default=""),
    age: str = Form(default=""),
    gender: str = Form(default=""),
    phoneNumber: str = Form(default=""),
    prediction: str = Form(default=""),
    confidence: str = Form(default="")
):
    try:
        # Print received data for debugging
        print(f"Received data: name={patientName}, age={age}, gender={gender}, phone={phoneNumber}")
        print(f"Prediction={prediction}, confidence={confidence}")

        # Validate form data
        if not all([patientName, age, gender, phoneNumber, prediction, confidence]):
            missing_fields = []
            if not patientName: missing_fields.append("Patient Name")
            if not age: missing_fields.append("Age")
            if not gender: missing_fields.append("Gender")
            if not phoneNumber: missing_fields.append("Phone Number")
            if not prediction: missing_fields.append("Prediction")
            if not confidence: missing_fields.append("Confidence")
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        # Save the image temporarily
        content = await file.read()
        temp_path = app.state.temp_dir / "temp_report_image.jpg"
        with open(temp_path, "wb") as temp_file:
            temp_file.write(content)

        # Validate image
        if not os.path.exists(temp_path):
            raise ValueError("Failed to save uploaded image")

        try:
            # Generate PDF
            pdf_buffer = generate_pdf_report(
                str(temp_path),
                prediction,
                confidence,
                patientName,
                age,
                gender,
                phoneNumber
            )
        except Exception as pdf_error:
            raise ValueError(f"Failed to generate PDF: {str(pdf_error)}")

        # Clean up
        if temp_path.exists():
            temp_path.unlink()

        # Return PDF as StreamingResponse
        filename = f"gi_cancer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        headers = {
            'Content-Disposition': f'attachment; filename="{filename}"'
        }
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers=headers
        )

    except Exception as e:
        error_msg = str(e)
        print(f"Error generating report: {error_msg}")
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        return JSONResponse(
            status_code=422,
            content={"detail": error_msg}
        )

if __name__ == "__main__":
    config = uvicorn.Config(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        log_level="info",
        loop="asyncio"
    )
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        print("Received shutdown signal, cleaning up...")
        # Additional cleanup if needed
    finally:
        print("Server shutdown complete")
