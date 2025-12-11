from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import clean_resume


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "resume_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "resume_classifier.joblib"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    if "Resume" not in df.columns or "Category" not in df.columns:
        raise ValueError("CSV must contain 'Resume' and 'Category' columns")
    df["cleaned_resume"] = df["Resume"].apply(clean_resume)
    return df


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    sublinear_tf=True,
                    stop_words="english",
                    max_features=5000,
                ),
            ),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced")),
        ]
    )


def train() -> None:
    df = load_data()
    X = df["cleaned_resume"].values
    y = df["Category"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def predict_resume(text: str):
    model = joblib.load(MODEL_PATH)
    cleaned = clean_resume(text)
    prediction = model.predict([cleaned])
    probabilities = model.predict_proba([cleaned])
    return prediction[0], probabilities[0]


if __name__ == "__main__":
    train()
    # Example prediction
    sample_text = "3+ years as Frontend Developer, React, JavaScript, HTML, CSS, building responsive web apps..."
    predicted_category, top_class_probabilities = predict_resume(sample_text)
    print("Prediction:", predicted_category)
    print("Top class probabilities:", top_class_probabilities)
