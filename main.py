from contextlib import asynccontextmanager
import os
import pickle
from fastapi.staticfiles import StaticFiles
import numpy as np
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

# --- DB connection (for statement fallback) ---
engine = create_engine(os.getenv("DATABASE_URL"), connect_args={"sslmode": "require"})

# --- Load tag predictor model ---
with open("model.pkl", "rb") as f:
    data = pickle.load(f)
    tag_model = data["model"]
    vectorizer = data["vectorizer"]
    mlb = data["mlb"]

# --- Load rating predictor model ---
with open("rating_model.pkl", "rb") as f:
    rating_data = pickle.load(f)
    rating_model = rating_data["model"]
    rating_mlb = rating_data["mlb"]

solved_lookup = {}
tags_lookup = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global solved_lookup, tags_lookup
    response = requests.get("https://codeforces.com/api/problemset.problems?lang=en")
    data = response.json()

    stats = data["result"]["problemStatistics"]
    solved_lookup = {
        (s["contestId"], s["index"]): s["solvedCount"]
        for s in stats
    }

    problems = data["result"]["problems"]
    tags_lookup = {
        (p["contestId"], p["index"]): p.get("tags", [])
        for p in problems
    }
    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


class PredictRequest(BaseModel):
    text: str
    top_k: int = 5
    threshold: float = 0.1


class RatingPredictRequest(BaseModel):
    contest_id: int
    problem_index: str


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict")
def predict(req: PredictRequest):
    X = vectorizer.transform([req.text])
    probs = tag_model.predict_proba(X)[0]
    top_indices = np.argsort(probs)[::-1][:req.top_k]
    predictions = [
        {"tag": mlb.classes_[i], "confidence": round(float(probs[i]), 3)}
        for i in top_indices if probs[i] >= req.threshold
    ]
    return JSONResponse({"predictions": predictions})


@app.get("/tags")
def get_tags():
    return JSONResponse({"tags": list(mlb.classes_)})


def predict_tags_from_statement(statement: str, top_k: int = 5, threshold: float = 0.1):
    """Run the tag model over a problem statement, return a plain list of tag strings."""
    X = vectorizer.transform([statement])
    probs = tag_model.predict_proba(X)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [mlb.classes_[i] for i in top_indices if probs[i] >= threshold]


def fetch_statement(contest_id: int, problem_index: str):
    """Look up the statement column for a problem from the DB. Returns None if not found."""
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT statement FROM problems
                WHERE contest_id = :cid AND problem_index = :idx
            """),
            {"cid": contest_id, "idx": problem_index}
        ).fetchone()
    if row is None or not row.statement:
        return None
    return row.statement


@app.post("/predict-rating")
def predict_rating(req: RatingPredictRequest):
    key = (req.contest_id, req.problem_index)
    solved_count = solved_lookup.get(key)

    if solved_count is None:
        return JSONResponse(
            {"error": "No solve-count data found for this problem."},
            status_code=404
        )

    tags = tags_lookup.get(key, [])
    tag_source = "codeforces"

    # Fallback: no tags from CF API -> try predicting tags from the DB statement
    if not tags:
        statement = fetch_statement(req.contest_id, req.problem_index)
        if statement:
            tags = predict_tags_from_statement(statement)
            tag_source = "predicted_from_statement"
        else:
            tag_source = "none_available"

    # Keep only tags the rating model was trained on, to avoid unseen-label errors
    known_tags = [t for t in tags if t in rating_mlb.classes_]

    log_solved_count = np.log1p([solved_count]).reshape(-1, 1)
    tag_features = rating_mlb.transform([known_tags])
    X = np.hstack([log_solved_count, tag_features])

    predicted_rating = rating_model.predict(X)[0]
    rounded_rating = int(round(predicted_rating / 100) * 100)

    return JSONResponse({
        "contest_id": req.contest_id,
        "problem_index": req.problem_index,
        "solved_count": solved_count,
        "tags_used": known_tags,
        "tag_source": tag_source,
        "predicted_rating_raw": round(float(predicted_rating), 1),
        "predicted_rating": rounded_rating
    })