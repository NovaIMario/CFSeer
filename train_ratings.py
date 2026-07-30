import os
import pickle
import numpy as np
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"), connect_args={"sslmode": "require"})

# --- Step 1: Load rated problems + tags from DB ---
print("Loading rated problems from database...")
with engine.connect() as conn:
    rows = conn.execute(
        text("""
            SELECT contest_id, problem_index, rating, tags
            FROM problems
            WHERE rating IS NOT NULL AND tags IS NOT NULL
        """)
    ).fetchall()

print(f"Loaded {len(rows)} rated problems")

# --- Step 2: Pull solved_count (accepted submissions) from CF API ---
print("Fetching problem statistics from Codeforces API...")
response = requests.get("https://codeforces.com/api/problemset.problems?lang=en")
data = response.json()
stats = data["result"]["problemStatistics"]

solved_lookup = {
    (s["contestId"], s["index"]): s["solvedCount"]
    for s in stats
}

# --- Step 3: Join DB rows with solved_count, keep tags alongside ---
solved_counts = []
tag_lists = []
y = []
skipped = 0

for r in rows:
    key = (r.contest_id, r.problem_index)
    solved_count = solved_lookup.get(key)
    if solved_count is None:
        skipped += 1
        continue
    solved_counts.append(solved_count)
    tag_lists.append(r.tags)
    y.append(r.rating)

print(f"Matched {len(solved_counts)} problems with solve counts (skipped {skipped} with no match)")

y = np.array(y)

# --- Step 4: Log-transform solved_count ---
# Raw solved_count vs rating is not linear - solve counts span orders of
# magnitude (dozens to 100k+) while ratings span linearly (~800-3500).
# log1p(x) = log(1 + x), handles solved_count == 0 safely.
solved_counts = np.array(solved_counts)
log_solved_counts = np.log1p(solved_counts).reshape(-1, 1)

# --- Step 5: Multi-hot encode tags (same pattern as the tag predictor) ---
mlb = MultiLabelBinarizer()
tag_features = mlb.fit_transform(tag_lists)
print(f"Encoded {len(mlb.classes_)} distinct tags")

# --- Step 6: Combine features ---
# log_solved_counts is (n, 1), tag_features is (n, num_tags) -> hstack into (n, 1+num_tags)
X = np.hstack([log_solved_counts, tag_features])

# --- Step 7: Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

# --- Step 8: Train linear model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Step 9: Evaluate ---
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

y_pred_rounded = np.round(y_pred / 100) * 100
exact_matches = np.sum(y_pred_rounded == y_test)
exact_accuracy = exact_matches / len(y_test)

within_100 = np.sum(np.abs(y_pred - y_test) <= 100)
within_100_accuracy = within_100 / len(y_test)

print(f"\n--- Results ---")
print(f"MAE: {mae:.1f} rating points")
print(f"Exact match (rounded to nearest 100): {exact_matches}/{len(y_test)} ({exact_accuracy:.1%})")
print(f"Within ±100 points: {within_100}/{len(y_test)} ({within_100_accuracy:.1%})")

# --- Step 10: Feature importance (which tags push rating up/down) ---
feature_names = ["log_solved_count"] + list(mlb.classes_)
coefs = model.coef_
sorted_idx = np.argsort(coefs)[::-1]

print("\nTop tags associated with HIGHER rating:")
for i in sorted_idx[:5]:
    print(f"  {feature_names[i]}: {coefs[i]:+.1f}")

print("\nTop tags associated with LOWER rating:")
for i in sorted_idx[-5:]:
    print(f"  {feature_names[i]}: {coefs[i]:+.1f}")

# --- Step 11: Save model ---
print("\nSaving model...")
with open("rating_model.pkl", "wb") as f:
    pickle.dump({"model": model, "mlb": mlb}, f)
print("Done!")