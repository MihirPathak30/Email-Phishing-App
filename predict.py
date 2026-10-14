from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib

from model_training import MODEL_PATH, VECTORIZER_PATH, METADATA_PATH, preprocess_text, train_models
from utils import (
    analyze_email_indicators,
    build_reason_and_summary,
    compute_email_stats,
    risk_level_from_score,
    timestamp_now,
)

BASE_DIR = Path(__file__).resolve().parent


def ensure_artifacts() -> Dict[str, object]:
    if not (MODEL_PATH.exists() and VECTORIZER_PATH.exists() and METADATA_PATH.exists()):
        return train_models(force=True)
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def load_model_bundle():
    metadata = ensure_artifacts()
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer, metadata


def analyze_email(text: str) -> Dict[str, object]:
    model, vectorizer, metadata = load_model_bundle()
    processed = preprocess_text(text)
    transformed = vectorizer.transform([processed])
    prediction_num = int(model.predict(transformed)[0])
    if hasattr(model, "predict_proba"):
        phishing_probability = float(model.predict_proba(transformed)[0][1])
    else:
        decision = float(model.decision_function(transformed)[0])
        phishing_probability = 1 / (1 + (2.71828 ** (-decision)))
    prediction = "Phishing" if prediction_num == 1 else "Legitimate"
    confidence = phishing_probability * 100 if prediction == "Phishing" else (1 - phishing_probability) * 100
    indicators = analyze_email_indicators(text)
    stats = compute_email_stats(text, indicators)
    risk_level, risk_color, risk_score = risk_level_from_score(phishing_probability, indicators)
    reason, summary = build_reason_and_summary(prediction, confidence, indicators, stats)

    result = {
        "timestamp": timestamp_now(),
        "prediction": prediction,
        "prediction_numeric": prediction_num,
        "confidence": round(confidence, 2),
        "phishing_probability": round(phishing_probability, 4),
        "legitimate_probability": round(1 - phishing_probability, 4),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_score": risk_score,
        "reason": reason,
        "summary": summary,
        "processed_text": processed,
        "indicators": indicators,
        "stats": stats,
        "metadata": metadata,
        "preview": text[:120].replace("\n", " "),
    }
    return result
