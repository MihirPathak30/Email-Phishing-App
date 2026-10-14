from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_phishing_email_model.joblib"
VECTORIZER_PATH = OUTPUT_DIR / "tfidf_vectorizer.joblib"
METADATA_PATH = OUTPUT_DIR / "training_metadata.json"
DATASET_PATH = DATASET_DIR / "phishing_email_dataset.csv"

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "for", "to", "of", "at", "by", "with", "is", "are",
    "was", "were", "be", "been", "am", "it", "this", "that", "these", "those", "from", "as", "your", "our", "we", "you",
    "he", "she", "they", "them", "their", "his", "her", "its", "have", "has", "had", "will", "would", "should", "can",
    "could", "may", "might", "do", "does", "did", "not", "so", "than", "then", "there", "here", "about", "into", "up",
    "down", "out", "over", "under", "again", "further", "once", "please", "dear", "hello", "hi"
}


PHISHING_TEMPLATES = [
    "Subject: Urgent account verification required. Dear user, your {account_type} account will be suspended today unless you verify your {credential} at {url}. Reply with your {credential} and otp immediately.",
    "Security alert: suspicious login detected from {ip}. Confirm your password and billing details at {url} to avoid permanent closure.",
    "Final warning from billing department. Your invoice remains unpaid. Open attachment invoice_{number}.pdf and submit payment now via {url}.",
    "Congratulations, you have won a gift reward. Click {url} and enter bank account details to claim the amount before deadline.",
    "Your mailbox is over quota. Login at {url} using your email {email} and password to restore access today.",
    "Important payroll update. Finance needs your account number, pin, and verification code right away. Use {url} to confirm identity.",
    "Package delivery failed. Verify address and card payment at {url}. Contact +1-202-{number}-8844 to avoid return of parcel.",
    "Crypto wallet alert: unusual withdrawal detected. Secure your wallet through {url} and reset passcode now.",
    "Employee benefit form pending. Download attached document and sign in at {url} to keep your salary release active.",
    "Tax refund available. Submit national id, bank details, and password through secure form {url} before expiry."
]

LEGITIMATE_TEMPLATES = [
    "Subject: Weekly team sync reminder. Hello team, our meeting is scheduled for {date}. Please join the conference room at {time}.",
    "Dear colleague, attached is the monthly project update. Review the document and share comments by {date}. Regards, PMO office.",
    "Subject: Training session confirmation. Your seat is confirmed for the workshop on {date}. Contact hr@company.com if you need support.",
    "Hello customer, your recent order has shipped successfully. Track the package from your account dashboard whenever convenient.",
    "Reminder: submit your leave application in the HR portal before {date}. This is a routine administrative notice.",
    "Finance memo: payroll will be processed on {date}. No further action is required from your side.",
    "Subject: Campus library notice. The library will remain open until {time} during exam week. Thank you.",
    "Dear student, your assignment feedback has been published in the learning portal. Please review it before next class.",
    "This is a confirmation for your service appointment on {date}. If you need to reschedule, call our office during business hours.",
    "Hello everyone, the department lunch is planned for {date}. Please confirm attendance for catering arrangements."
]

URL_POOL = [
    "https://bit.ly/secure-login-reset",
    "https://tinyurl.com/account-update-center",
    "http://account-verify.co/secure",
    "https://secure-update.net/login",
    "http://wallet-alert.io/restore",
    "https://company.com/portal",
]
ACCOUNT_TYPES = ["bank", "payroll", "mailbox", "student portal", "cloud storage", "payment"]
CREDENTIALS = ["password", "passcode", "otp", "security code", "credentials"]
EMAILS = ["support@account-verify.co", "security@wallet-alert.io", "helpdesk@secure-update.net"]
DATES = ["Monday", "Tuesday", "Friday", "2026-08-01", "next week", "tomorrow afternoon"]
TIMES = ["10:00 AM", "1:30 PM", "3:00 PM", "09:15 AM"]


def preprocess_text(text: str) -> str:
    text = text or ""
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"[^a-z0-9\s@._:-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [token for token in text.split() if token not in STOP_WORDS and len(token) > 2]
    return " ".join(tokens)


def generate_synthetic_dataset(phish_n: int = 220, legit_n: int = 220) -> pd.DataFrame:
    random.seed(42)
    rows: List[Dict[str, object]] = []
    for i in range(phish_n):
        template = random.choice(PHISHING_TEMPLATES)
        text = template.format(
            account_type=random.choice(ACCOUNT_TYPES),
            credential=random.choice(CREDENTIALS),
            url=random.choice(URL_POOL[:5]),
            ip=f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            number=random.randint(100, 999),
            email=random.choice(EMAILS),
        )
        if i % 4 == 0:
            text += f" Click now and confirm login from device {random.randint(1000, 9999)}."
        rows.append({"text": text, "label": 1})
    for i in range(legit_n):
        template = random.choice(LEGITIMATE_TEMPLATES)
        text = template.format(date=random.choice(DATES), time=random.choice(TIMES))
        if i % 5 == 0:
            text += " Please contact admin@company.com for assistance."
        rows.append({"text": text, "label": 0})
    random.shuffle(rows)
    return pd.DataFrame(rows)


def load_dataset() -> pd.DataFrame:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
        columns = {col.lower(): col for col in df.columns}
        text_col = columns.get("text") or columns.get("email") or columns.get("message")
        label_col = columns.get("label") or columns.get("target") or columns.get("class")
        if text_col and label_col:
            cleaned = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).dropna()
            cleaned["label"] = cleaned["label"].apply(normalize_label)
            cleaned = cleaned.dropna()
            if not cleaned.empty:
                return cleaned
    df = generate_synthetic_dataset()
    df.to_csv(DATASET_PATH, index=False)
    return df


def normalize_label(value):
    if isinstance(value, str):
        value_l = value.strip().lower()
        if value_l in {"phishing", "spam", "fraud", "malicious", "1", "true", "yes"}:
            return 1
        if value_l in {"legitimate", "ham", "safe", "normal", "0", "false", "no"}:
            return 0
        return np.nan
    if value in {1, True}:
        return 1
    if value in {0, False}:
        return 0
    return np.nan


def extract_feature_importance(model, feature_names: List[str], top_n: int = 20) -> List[Dict[str, object]]:
    if hasattr(model, "coef_"):
        scores = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        scores = np.abs(model.feature_importances_)
    else:
        return []
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [{"feature": feature_names[i], "importance": float(scores[i])} for i in top_indices]


def train_models(force: bool = False) -> Dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists() and METADATA_PATH.exists() and not force:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    df = load_dataset()
    df = df.dropna(subset=["text", "label"]).copy()
    df["processed_text"] = df["text"].astype(str).apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=220, random_state=42, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=16, random_state=42, class_weight="balanced"),
        "Support Vector Machine": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42),
    }

    metrics = {}
    best_name = None
    best_score = -1.0
    best_model = None
    best_predictions = None

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        metric = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
        }
        metrics[name] = metric
        if metric["f1"] > best_score:
            best_score = metric["f1"]
            best_name = name
            best_model = model
            best_predictions = preds

    feature_names = vectorizer.get_feature_names_out().tolist()
    importance = extract_feature_importance(best_model, feature_names, top_n=20)
    conf = confusion_matrix(y_test, best_predictions).tolist()

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    metadata = {
        "best_model_name": best_name,
        "dataset_path": str(DATASET_PATH.name),
        "dataset_size": int(len(df)),
        "class_distribution": {"legitimate": int((df['label'] == 0).sum()), "phishing": int((df['label'] == 1).sum())},
        "models": metrics,
        "feature_importance": importance,
        "confusion_matrix": conf,
        "labels": ["Legitimate", "Phishing"],
        "vectorizer_path": str(VECTORIZER_PATH.name),
        "model_path": str(MODEL_PATH.name),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    meta = train_models(force=True)
    print(f"Training complete. Best model: {meta['best_model_name']}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved vectorizer: {VECTORIZER_PATH}")
