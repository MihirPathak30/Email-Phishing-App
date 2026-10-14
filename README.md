# AI-Driven Phishing Email Detection

A production-style Streamlit cybersecurity dashboard for detecting phishing emails using NLP and machine learning.

## Features

- Paste complete email content for instant analysis
- Upload `.txt` and `.eml` email files
- Train and compare:
  - Logistic Regression
  - Naive Bayes
  - Random Forest
  - Decision Tree
  - Support Vector Machine
- Automatically select and save the best model
- Detect and display:
  - URLs and shortened URLs
  - IP addresses
  - Email addresses
  - Phone numbers
  - HTML tags
  - Attachment mentions
  - Urgency, financial, password, and suspicious-domain keywords
- Professional UI with glassmorphism cards, dark/light support, and dashboard-style layout
- Downloadable PDF and TXT reports
- Saved analysis history with recent predictions table
- Visualizations:
  - Confusion matrix
  - Feature importance
  - Prediction probability chart
  - Keyword frequency chart
  - Word cloud
  - Dataset distribution chart

## Project Structure

```text
phishing_email_app/
├── app.py
├── model_training.py
├── predict.py
├── utils.py
├── requirements.txt
├── README.md
├── run.bat
├── run.sh
├── assets/
│   └── shield_logo.svg
├── dataset/
│   └── phishing_email_dataset.csv   # generated automatically if missing
└── outputs/
    ├── best_phishing_email_model.joblib
    ├── tfidf_vectorizer.joblib
    ├── training_metadata.json
    └── analysis_history.json
```

## Quick Start

### 1) Create a virtual environment (recommended)

#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the training pipeline (optional)

The app can auto-train on first launch if model files are missing.

```bash
python model_training.py
```

### 4) Launch the app

```bash
streamlit run app.py
```

## Notes

- If `dataset/phishing_email_dataset.csv` does not exist, the training script creates a bundled starter dataset automatically.
- Replace the starter dataset with a larger real-world dataset for stronger final-year project evaluation.
- The app saves these files automatically after training:
  - `outputs/best_phishing_email_model.joblib`
  - `outputs/tfidf_vectorizer.joblib`
  - `outputs/training_metadata.json`
- Analysis history is saved in `outputs/analysis_history.json`.

## Recommended Real Dataset Format

Use a CSV file with at least these columns:

```csv
text,label
"Your account will be suspended...",1
"Meeting reminder for tomorrow...",0
```

Accepted label examples:
- `1`, `phishing`, `spam`, `fraud`
- `0`, `legitimate`, `ham`, `safe`

## Demo Flow

1. Launch the app.
2. Click **Sample Phishing** or **One-click Demo**.
3. Review the prediction, confidence, risk gauge, and indicators.
4. Open **Visuals** for model charts and word cloud.
5. Download PDF or TXT report.

## Production Improvement Ideas

- Replace synthetic starter data with a curated Kaggle or enterprise email dataset
- Add SHAP/LIME explanations for local interpretability
- Integrate sender reputation APIs and domain WHOIS checks
- Add authentication and persistent database storage
- Deploy on Streamlit Community Cloud, Render, or an internal server

## Python Version

Tested for **Python 3.11+**.
