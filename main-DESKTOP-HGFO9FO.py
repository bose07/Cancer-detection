from fastapi import FastAPI, UploadFile, File, HTTPException
from keras.models import load_model  # type: ignore
from keras.preprocessing.image import load_img, img_to_array  # type: ignore
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the model outside the endpoint to avoid reloading on every request
model_path = "model/DSnet_cancer_prediction.keras"
if not os.path.exists(model_path):
    logger.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = load_model(model_path)

# Define a prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        content = await file.read()
        with open("temp_image.jpg", "wb") as temp_file:
            temp_file.write(content)

        # Preprocess the image
        image = load_img("temp_image.jpg", target_size=(224, 224))
        image_array = img_to_array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)

        # Make a prediction
        prediction = model.predict(image_array)
        class_label = "Cancer" if prediction[0][0] < 0.5 else "Normal"

        return {"prediction": class_label}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
