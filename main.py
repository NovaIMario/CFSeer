from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

with open("model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    vectorizer = data["vectorizer"]
    mlb = data["mlb"]

class PredictRequest(BaseModel):
    text: str
    top_k: int = 5
    threshold: float = 0.1

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/predict")
def predict(req: PredictRequest):
    X = vectorizer.transform([req.text])
    probs = model.predict_proba(X)[0]
    top_indices = np.argsort(probs)[::-1][:req.top_k]
    predictions = [{"tag": mlb.classes_[i], "confidence": round(float(probs[i]), 3)} for i in top_indices if probs[i] >= req.threshold]
    return JSONResponse({"predictions": predictions})

@app.get("/tags")
def get_tags():
    return JSONResponse({"tags": list(mlb.classes_)})