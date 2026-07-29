import os
import pickle
import numpy as np
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"), connect_args={"sslmode": "require"})

# --- Step 1: Load rated problems from DB ---
print("Loading rated problems from database...")
with engine.connect() as conn:
    rows = conn.execute(
        text("""
            SELECT contest_id, problem_index, rating
            FROM problems
            WHERE rating IS NOT NULL
        """)
    ).fetchall()

print(f"Loaded {len(rows)} rated problems")

# --- Step 2: Pull solved_count (accepted submissions) from CF API ---
print("Fetching problem statistics from Codeforces API...")
response = requests.get("https://codeforces.com/api/problemset.problems?lang=en")
data = response.json()
stats = data["result"]["problemStatistics"]

# Build lookup: (contestId, index) -> solvedCount
solved_lookup = {
    (s["contestId"], s["index"]): s["solvedCount"]
    for s in stats
}

# --- Step 3: Join DB rows with solved_count ---
X = []
y = []
skipped = 0

for r in rows:
    key = (r.contest_id, r.problem_index)
    solved_count = solved_lookup.get(key)
    if solved_count is None:
        skipped += 1
        continue
    X.append([solved_count])
    y.append(r.rating)

print(f"Matched {len(X)} problems with solve counts (skipped {skipped} with no match)")

X = np.array(X)
y = np.array(y)

# --- Step 4: Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

# --- Step 5: Train linear model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Step 6: Evaluate ---
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

# "Correct" = predicted rating (rounded to nearest 100) matches actual exactly
y_pred_rounded = np.round(y_pred / 100) * 100
exact_matches = np.sum(y_pred_rounded == y_test)
exact_accuracy = exact_matches / len(y_test)

# Looser tolerance: within 100 points either way (adjacent tier)
within_100 = np.sum(np.abs(y_pred - y_test) <= 100)
within_100_accuracy = within_100 / len(y_test)

print(f"\n--- Results ---")
print(f"MAE: {mae:.1f} rating points")
print(f"Exact match (rounded to nearest 100): {exact_matches}/{len(y_test)} ({exact_accuracy:.1%})")
print(f"Within ±100 points: {within_100}/{len(y_test)} ({within_100_accuracy:.1%})")
print(f"Coefficient (solved_count): {model.coef_[0]:.6f}")
print(f"Intercept: {model.intercept_:.1f}")

# --- Step 7: Save model ---
print("\nSaving model...")
with open("rating_model.pkl", "wb") as f:
    pickle.dump({"model": model}, f)
print("Done!")