from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
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
import contextlib

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
model = None

@app.on_event("startup")
async def load_ml_model():
    global model
    try:
        model = load_model("model\DSnet_cancer_prediction.keras")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

@contextlib.contextmanager
def temporary_file(suffix=None):
    temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix or ''}"
    try:
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def generate_pdf_report(image_path, prediction, confidence):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add centered title
    title = "Gastrointestinal Cancer Detection using AI"
    title_font_size = 28
    c.setFont("Helvetica-Bold", title_font_size)
    title_width = c.stringWidth(title, "Helvetica-Bold", title_font_size)
    c.drawString((width - title_width) / 2, height - 50, title)
    
    # Add horizontal line under title
    c.setStrokeColor(colors.blue)
    c.line(50, height - 60, width - 50, height - 60)

    # Add centered header
    header = "Detection Report"
    header_font_size = 20
    c.setFont("Helvetica-Bold", header_font_size)
    header_width = c.stringWidth(header, "Helvetica-Bold", header_font_size)
    c.drawString((width - header_width) / 2, height - 100, header)
    
    # Add centered timestamp
    timestamp = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    c.setFont("Helvetica", 12)
    timestamp_width = c.stringWidth(timestamp, "Helvetica", 12)
    c.drawString((width - timestamp_width) / 2, height - 120, timestamp)

    # Add image with smaller dimensions and moved down
    img = Image.open(image_path)
    img_width, img_height = img.size
    aspect = img_height / float(img_width)
    
    # Reduced maximum dimensions for the image
    max_width = width/2  # Half of page width
    max_height = height/3  # One-third of page height
    
    img_width = min(max_width, img_width)
    img_height = img_width * aspect
    if img_height > max_height:
        img_height = max_height
        img_width = img_height / aspect
    
    # Center the image horizontally and move down by adjusting vertical position
    x_position = (width - img_width) / 2
    y_position = height - 400  # Moved down from 350 to 400
    c.drawImage(image_path, x_position, y_position, width=img_width, height=img_height)

    # Add "Analyzed Image" caption
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.grey)
    caption = "Analyzed Image"
    caption_width = c.stringWidth(caption, "Helvetica", 10)
    c.drawString((width - caption_width) / 2, y_position - 10, caption)

    # Adjust all subsequent y-positions accordingly
    # Add results with improved styling
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y_position - 50, "Analysis Results")
    
    # Add box around results
    c.setStrokeColor(colors.grey)
    c.rect(45, y_position - 120, width - 90, 90)
    
    c.setFont("Helvetica", 14)
    result_color = colors.red if prediction == "Cancer" else colors.green
    c.setFillColor(result_color)
    c.drawString(60, y_position - 80, f"Prediction: {prediction}")
    c.drawString(60, y_position - 100, f"Confidence: {confidence}%")
    
    # Add precautions with improved styling
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y_position - 150, "Precautions and Next Steps")
    
    # Add box around precautions
    c.setStrokeColor(colors.grey)
    c.rect(45, y_position - 270, width - 90, 100)

    c.setFont("Helvetica", 12)
    precautions = []
    if prediction == "Cancer":
        precautions = [
            "• Seek immediate medical attention for further evaluation",
            "• Schedule an appointment with a gastroenterologist",
            "• Avoid delays in medical consultation",
            "• Prepare a list of symptoms and medical history"
        ]
    else:
        precautions = [
            "• Maintain regular medical check-ups",
            "• Follow a healthy diet and lifestyle",
            "• Report any new symptoms to your healthcare provider",
            "• Schedule routine screening as recommended by your doctor"
        ]

    y_position_prec = y_position - 180
    for precaution in precautions:
        c.drawString(60, y_position_prec, precaution)
        y_position_prec -= 20

    # Add disclaimer with box at the bottom
    c.setStrokeColor(colors.grey)
    c.rect(45, 20, width - 90, 50)
    
    c.setFont("Helvetica", 10)
    disclaimer_text = [
        "Disclaimer: This is an AI-assisted analysis and should not be used as a substitute",
        "for professional medical advice, diagnosis, or treatment. Please consult with a qualified",
        "healthcare provider."
    ]
    y_position_disc = 55
    for line in disclaimer_text:
        c.drawString(50, y_position_disc, line)
        y_position_disc -= 15

    c.save()
    buffer.seek(0)
    return buffer

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("templates/index.html", "r") as f:
            return f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading template: {str(e)}")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    with temporary_file(".jpg") as temp_path:
        try:
            # Read the uploaded image
            content = await file.read()
            with open(temp_path, "wb") as temp_file:
                temp_file.write(content)

            # Preprocess the image
            image = load_img(temp_path, target_size=(380, 380))
            image_array = img_to_array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Make a prediction
            prediction = model.predict(image_array)
            confidence = float(abs(prediction[0][0] - 0.5) * 2 * 100)
            class_label = "Cancer" if prediction[0][0] < 0.5 else "Normal"
            confidence_rounded = round(confidence, 2)

            return {
                "prediction": class_label,
                "confidence": confidence_rounded
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/generate-report/")
async def generate_report(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    with temporary_file(".jpg") as temp_path:
        try:
            # Save the image temporarily
            content = await file.read()
            with open(temp_path, "wb") as temp_file:
                temp_file.write(content)

            # Get prediction
            image = load_img(temp_path, target_size=(380, 380))
            image_array = img_to_array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            prediction = model.predict(image_array)
            confidence = float(abs(prediction[0][0] - 0.5) * 2 * 100)
            class_label = "Cancer" if prediction[0][0] < 0.5 else "Normal"
            
            # Generate PDF
            pdf_buffer = generate_pdf_report(temp_path, class_label, round(confidence, 2))
            
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
            raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
