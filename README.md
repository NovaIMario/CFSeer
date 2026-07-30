# CFSeer

A companion service that scrapes Codeforces problem data, predicts problem tags, and estimates difficulty ratings for problems Codeforces hasn't officially rated yet.

**Live demo:** https://cfseer.onrender.com/
**Repo:** https://github.com/NovaIMario/CFSeer

---

## Features

- **Resumable scraper** for 10,000+ Codeforces problems (11050/11140 scraped), with crash recovery — progress is persisted to PostgreSQL so a restart doesn't mean re-scraping from scratch, and intermittent Cloudflare blocks don't lose progress.
- **Tag prediction**: a TF-IDF + logistic regression classifier predicts problem tags across 38 categories, used to auto-fill tags for problems missing them. When a problem has no officially assigned tags, CFSeer falls back to predicting tags directly from the problem statement text.
- **Rating prediction**: estimates a numeric difficulty rating for problems Codeforces hasn't rated yet, using solve count and (multi-hot encoded) tags as features.
  - Baseline: solve-count-only linear regression, MAE 508.9
  - Improved: log-transforming solve counts + adding tag features, MAE 212.8 — a 58% reduction
- **Consumed by [AlgoReady](https://github.com/NovaIMario/AlgoReady)** via a JSON API, surfacing provisional ratings directly in the problem browser.

## Tech Stack

- **Backend:** Python, FastAPI
- **ML:** scikit-learn (TF-IDF vectorizer, logistic regression, linear regression)
- **Database:** PostgreSQL (hosted on Supabase)
- **Containerization:** Docker
- **Deployment:** Render (with model persistence via pickle)

## API

### `POST /predict`
Predict tags from raw problem statement text.
```json
{ "text": "...", "top_k": 5, "threshold": 0.1 }
```

### `GET /tags`
List all known tag categories.

### `POST /predict-rating`
Predict a difficulty rating for a given problem.
```json
{ "contest_id": 1234, "problem_index": "A" }
```
Looks up the problem's solve count and tags (from the Codeforces API); if no tags are available, falls back to predicting tags from the stored problem statement before generating a rating estimate. Response includes the raw and rounded predicted rating, plus which tags were used and where they came from (`codeforces` vs `predicted_from_statement`).

## Local Setup

```bash
git clone https://github.com/NovaIMario/CFSeer.git
cd CFSeer
pip install -r requirements.txt
```

Create a `.env` file with:

```
DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<db>
```

Run locally:

```bash
uvicorn main:app --reload
```

Or via Docker:

```bash
docker build -t cfseer .
docker run -p 8000:8000 --env-file .env cfseer
```
