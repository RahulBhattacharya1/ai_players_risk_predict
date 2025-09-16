import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from typing import List, Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest

st.set_page_config(page_title="Injury Risk Detection", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def guess_id_columns(cols: List[str]) -> List[str]:
    patterns = [
        r"player[_\s]?id",
        r"player",
        r"athlete",
        r"name$",
        r"full[_\s]?name",
    ]
    out = []
    for c in cols:
        lc = c.lower()
        if any(re.search(p, lc) for p in patterns):
            out.append(c)
    return out

def guess_label_column(cols: List[str]) -> Optional[str]:
    patterns = [
        r"injur",         # injury, injured, injuries
        r"out[_\s]?status",
        r"availability[_\s]?risk",
        r"risk[_\s]?label",
        r"missed[_\s]?games[_\s]?flag",
    ]
    for c in cols:
        lc = c.lower()
        if any(re.search(p, lc) for p in patterns):
            return c
    return None

def numeric_feature_candidates(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in nums if c not in exclude]

def safe_downcast(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes(include=["float64"]).columns:
        out[c] = pd.to_numeric(out[c], downcast="float")
    for c in out.select_dtypes(include=["int64"]).columns:
        out[c] = pd.to_numeric(out[c], downcast="integer")
    return out

def build_supervised_pipeline(num_cols: List[str]) -> Pipeline:
    num_proc = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pre = ColumnTransformer(
        transformers=[("num", num_proc, num_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe

def build_unsupervised_model() -> IsolationForest:
    return IsolationForest(
        contamination="auto",
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )

def compute_unsupervised_risk_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    # IsolationForest returns higher scores for "normal" via score_samples.
    # Convert to risk where higher = more risky:
    raw = model.score_samples(X)
    risk = (raw.min() - raw)  # invert
    # Normalize 0..1
    if np.nanmax(risk) > np.nanmin(risk):
        risk = (risk - np.nanmin(risk)) / (np.nanmax(risk) - np.nanmin(risk))
    else:
        risk = np.zeros_like(risk)
    return risk

def make_feature_preview_chart(df: pd.DataFrame, id_col: Optional[str], risk_col: str):
    tmp = df.copy()
    if id_col is None:
        tmp["Entity"] = np.arange(len(tmp))
        id_col = "Entity"
    fig = px.bar(
        tmp.sort_values(risk_col, ascending=False).head(30),
        x=id_col,
        y=risk_col,
        title="Top 30 Highest Risk (sorted)",
    )
    st.plotly_chart(fig, use_container_width=True)

def to_downloadable_csv(df: pd.DataFrame, filename: str = "injury_risk_scored.csv") -> Tuple[bytes, str]:
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    return buff.getvalue().encode("utf-8"), filename

# -----------------------------
# UI
# -----------------------------
st.title("Injury Risk Detection Dashboard")

st.markdown(
    """
This app supports **two modes**:
- **Supervised**: If your data has an injury/label column (e.g., `injury`, `injured_flag`), it trains a classifier and outputs risk probabilities.
- **Unsupervised**: If there is no injury label, it uses Isolation Forest to flag abnormal workload patterns as higher risk.
"""
)

uploaded = st.file_uploader("Upload season or player dataset (CSV)", type=["csv"])
if not uploaded:
    st.info("Upload a CSV to proceed.")
    st.stop()

df_raw = pd.read_csv(uploaded)
df_raw = safe_downcast(df_raw)
st.success(f"Loaded {df_raw.shape[0]} rows × {df_raw.shape[1]} columns.")
with st.expander("Preview data", expanded=False):
    st.dataframe(df_raw.head(20), use_container_width=True)

cols = df_raw.columns.tolist()
id_guesses = guess_id_columns(cols)
label_guess = guess_label_column(cols)

left, right = st.columns([1, 1])
with left:
    mode = st.radio("Mode", ["Auto (detect label)", "Supervised", "Unsupervised"], index=0)
with right:
    if mode == "Auto (detect label)":
        st.write(f"Detected label column: **{label_guess or 'None'}**")
    pass

if mode == "Auto (detect label)":
    if label_guess is not None:
        active_mode = "Supervised"
    else:
        active_mode = "Unsupervised"
else:
    active_mode = mode

# Choose ID column (optional)
id_col = st.selectbox(
    "Player/Entity identifier column (optional, used for display)",
    options=["<none>"] + id_guesses + [c for c in cols if c not in id_guesses],
    index=0 if not id_guesses else 1,
)
id_col = None if id_col == "<none>" else id_col

# Choose label (only for supervised)
label_col = None
if active_mode == "Supervised":
    defaults = [label_guess] if label_guess and label_guess in cols else []
    label_col = st.selectbox(
        "Injury label column (binary: 1=injured/at-risk, 0=healthy)",
        options=cols,
        index=cols.index(defaults[0]) if defaults else 0,
    )

# Feature selection
exclude = [label_col] if label_col else []
num_candidates = numeric_feature_candidates(df_raw, exclude=exclude)
with st.expander("Feature selection", expanded=True):
    st.caption("Pick numeric workload and profile features (minutes, matches, distance, sprints, age, rest days, etc.).")
    features = st.multiselect(
        "Numeric features",
        options=num_candidates,
        default=[c for c in num_candidates if len(features) < 0] if False else num_candidates[: min(12, len(num_candidates))],
        help="You can change this anytime. Choose meaningful workload indicators.",
    )
    if not features:
        st.warning("Select at least one numeric feature.")
        st.stop()

# Additional controls
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    if active_mode == "Unsupervised":
        contamination = st.slider("Assumed at-risk share (unsupervised)", 0.01, 0.20, 0.06, 0.01)
    else:
        contamination = None
with c2:
    test_size = st.slider("Test size (supervised)", 0.1, 0.4, 0.2, 0.05)
with c3:
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)

st.markdown("---")

# -----------------------------
# Run
# -----------------------------
run = st.button("Train / Score")
if not run:
    st.stop()

work_df = df_raw.copy()

if active_mode == "Supervised":
    y = work_df[label_col].astype(float)
    X = work_df[features]

    # Guardrail: ensure label is binary
    unique_labels = sorted(pd.Series(y).dropna().unique().tolist())
    if not set(unique_labels).issubset({0, 1}):
        st.error(f"Label {label_col} must be binary 0/1. Found values: {unique_labels[:10]}")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = build_supervised_pipeline(features)
    pipe.fit(X_train, y_train)

    # Metrics
    y_prob = pipe.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")

    try:
        ap = average_precision_score(y_test, y_prob)
    except ValueError:
        ap = float("nan")

    st.subheader("Validation Metrics")
    m1, m2 = st.columns([1, 1])
    m1.metric("ROC AUC", f"{auc:.3f}" if np.isfinite(auc) else "NA")
    m2.metric("Average Precision", f"{ap:.3f}" if np.isfinite(ap) else "NA")

    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix (threshold=0.5)")
    st.dataframe(pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))

    # Score all rows
    all_prob = pipe.predict_proba(work_df[features])[:, 1]
    scored = work_df.copy()
    scored["risk_score"] = all_prob
    if id_col is not None and id_col in scored.columns:
        display_cols = [id_col, "risk_score"] + [c for c in features if c != id_col]
    else:
        display_cols = ["risk_score"] + features

    st.subheader("Risk Scores (Supervised)")
    st.dataframe(
        scored.sort_values("risk_score", ascending=False)[display_cols].head(200),
        use_container_width=True,
        height=480,
    )
    make_feature_preview_chart(scored, id_col=id_col, risk_col="risk_score")

    # Download
    csv_bytes, fname = to_downloadable_csv(scored)
    st.download_button("Download Scored CSV", data=csv_bytes, file_name=fname, mime="text/csv")

else:
    # Unsupervised
    X = work_df[features].copy()
    pre = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_pre = pre.fit_transform(X)

    iso = IsolationForest(
        contamination=contamination if contamination is not None else "auto",
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_pre)

    risk_scores = compute_unsupervised_risk_scores(iso, X_pre)
    scored = work_df.copy()
    scored["risk_score"] = risk_scores
    if id_col is not None and id_col in scored.columns:
        display_cols = [id_col, "risk_score"] + [c for c in features if c != id_col]
    else:
        display_cols = ["risk_score"] + features

    st.subheader("Risk Scores (Unsupervised)")
    st.dataframe(
        scored.sort_values("risk_score", ascending=False)[display_cols].head(200),
        use_container_width=True,
        height=480,
    )
    make_feature_preview_chart(scored, id_col=id_col, risk_col="risk_score")

    csv_bytes, fname = to_downloadable_csv(scored)
    st.download_button("Download Scored CSV", data=csv_bytes, file_name=fname, mime="text/csv")

st.markdown("---")
st.caption(
    "Tips: Include workload features like minutes played, matches in last 7/14 days, age, "
    "rest days, sprint counts, high-speed distance, past injuries, and travel. "
    "If you don’t have labels, start with Unsupervised mode to triage risk."
)
