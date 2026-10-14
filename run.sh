#!/usr/bin/env bash
set -e

PYTHON_BIN=${PYTHON_BIN:-python3}

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python model_training.py || true
streamlit run app.py
