
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.pardir, os.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from fastapi import FastAPI, UploadFile, File
import io
from PIL import Image
import numpy as np

from model_app import model_pipeline

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    print(file)
    image = Image.open(file.file)
    # content = image.file.read()
    # image = Image.open(io.BytesIO(content))
    print(image)
    result = model_pipeline(image)
    print(result)
    return {"prediction": result}
