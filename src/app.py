from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import io
from PIL import Image
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = tf.keras.models.load_model("models/latest/model.keras")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    result = "Dog" if prediction[0][0] > 0.5 else "Cat"
    
    return {"prediction": result, "confidence": float(abs(prediction[0][0]))}
