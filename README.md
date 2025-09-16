# Injury Risk Detection Dashboard

An end-to-end Streamlit app that scores player injury risk using your CSV data.

## Features
- Supervised mode (RandomForest) if you have a binary injury label.
- Unsupervised mode (IsolationForest) if you do not.
- Interactive table and top-30 risk chart.
- One-click CSV export with risk scores.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
