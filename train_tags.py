import os
import pickle
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"), connect_args={"sslmode": "require"})

print("Loading problems from database...")

with engine.connect() as conn:
    rows = conn.execute(
        text("""
            SELECT statement, tags
            FROM problems
            WHERE statement IS NOT NULL AND tags IS NOT NULL
        """)
    ).fetchall()

print(f"Loaded {len(rows)} problems")

texts = [r.statement for r in rows]
labels = [r.tags for r in rows]  # tags is a Postgres array -> comes back as a Python list

# --- Rest is unchanged ---
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels)

vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english", min_df=5)
X = vectorizer.fit_transform(texts)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=77)

model = OneVsRestClassifier(LogisticRegression(max_iter=1000), n_jobs=-1)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

f1_micro = f1_score(Y_test, Y_pred, average="micro", zero_division=0)
f1_macro = f1_score(Y_test, Y_pred, average="macro", zero_division=0)
print(f"F1 Score (micro): {f1_micro:.3f}")
print(f"F1 Score (macro): {f1_macro:.3f}")
print(len(mlb.classes_))

print("Saving model...")
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "vectorizer": vectorizer, "mlb": mlb}, f)
print("Done!")