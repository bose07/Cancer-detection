from fastapi import FastAPI, UploadFile, File
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

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_pdf_report(image_path, prediction, confidence):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add header
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "GI Cancer Detection Report")
    
    # Add timestamp
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Add image
    img = Image.open(image_path)
    img_width, img_height = img.size
    aspect = img_height / float(img_width)
    max_width = width - 100
    max_height = height / 2
    img_width = min(max_width, img_width)
    img_height = img_width * aspect
    if img_height > max_height:
        img_height = max_height
        img_width = img_height / aspect
    
    c.drawImage(image_path, 50, height - 350, width=img_width, height=img_height)

    # Add results
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 400, "Analysis Results")
    
    c.setFont("Helvetica", 14)
    result_color = colors.red if prediction == "Cancer" else colors.green
    c.setFillColor(result_color)
    c.drawString(50, height - 430, f"Prediction: {prediction}")
    c.drawString(50, height - 450, f"Confidence: {confidence}%")
    
    # Add precautions
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 500, "Precautions and Next Steps")
    
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

    y_position = height - 530
    for precaution in precautions:
        c.drawString(50, y_position, precaution)
        y_position -= 20

    # Add disclaimer
    c.setFont("Helvetica", 10)
    disclaimer_text = [
        "Disclaimer: This is an AI-assisted analysis and should not be used as a substitute",
        "for professional medical advice, diagnosis, or treatment. Please consult with a qualified",
        "healthcare provider."
    ]
    y_position = 50
    for line in disclaimer_text:
        c.drawString(50, y_position, line)
        y_position -= 15

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
    with open("temp_image.jpg", "wb") as temp_file:
        temp_file.write(content)

    # Preprocess the image - Changed target size to 380x380
    image = load_img("temp_image.jpg", target_size=(380, 380))
    image_array = img_to_array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)

    # Make a prediction
    prediction = model.predict(image_array)
    confidence = float(abs(prediction[0][0] - 0.5) * 2 * 100)  # Convert to percentage
    class_label = "Cancer" if prediction[0][0] < 0.5 else "Normal"
    confidence_rounded = round(confidence, 2)

    # Clean up temporary file
    if os.path.exists("temp_image.jpg"):
        os.remove("temp_image.jpg")

    return {
        "prediction": class_label,
        "confidence": confidence_rounded
    }

@app.post("/generate-report/")
async def generate_report(file: UploadFile = File(...)):
    try:
        # Save the image temporarily
        content = await file.read()
        with open("temp_report_image.jpg", "wb") as temp_file:
            temp_file.write(content)

        # Get prediction
        model = load_model("model\DSnet_cancer_prediction.keras")
        image = load_img("temp_report_image.jpg", target_size=(380, 380))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        confidence = float(abs(prediction[0][0] - 0.5) * 2 * 100)
        class_label = "Cancer" if prediction[0][0] < 0.5 else "Normal"
        
        # Generate PDF
        pdf_buffer = generate_pdf_report("temp_report_image.jpg", class_label, round(confidence, 2))
        
        # Clean up
        if os.path.exists("temp_report_image.jpg"):
            os.remove("temp_report_image.jpg")

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
        print(f"Error generating report: {str(e)}")
        if os.path.exists("temp_report_image.jpg"):
            os.remove("temp_report_image.jpg")
        raise
