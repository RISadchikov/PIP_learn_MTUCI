from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from io import BytesIO

app = FastAPI()
model = joblib.load('laptop_price_model.pkl')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(BytesIO(content))
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}
