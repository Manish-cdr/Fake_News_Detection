import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import preprocess

# Load + label
fake = pd.read_csv("Fake.csv"); fake["label"] = 0
true = pd.read_csv("True.csv"); true["label"] = 1
df = pd.concat([fake, true], ignore_index=True)
df["title"] = df["title"].fillna(""); df["text"] = df["text"].fillna("")

# Clean: drop exact duplicates (they leak across train/test) and empty bodies
df = df.drop_duplicates(["title", "text"])
df = df[df["text"].str.strip() != ""]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df["content"] = df["title"] + " " + df["text"]
X, y = df["content"], df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Tuned hyperparameters (chosen on a validation split of the training data)
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=preprocess,
        sublinear_tf=True,   # dampen very frequent terms
        min_df=2,            # ignore terms seen in <2 docs
        max_df=0.95,         # ignore terms in >95% of docs
        ngram_range=(1, 1),
        # no stop-word removal: function-word usage is a useful style signal
    )),
    ("clf", LogisticRegression(C=100, solver="liblinear", max_iter=1000)),
])

model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test), target_names=["Fake", "Real"], digits=4))

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
print("✅ Model saved to model/model.pkl")
