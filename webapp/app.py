from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from backend.train import run_training
from backend.evaluate import run_evaluation

st.set_page_config(page_title="OpenLHC-Anomaly Toolkit", layout="wide")

st.title("OpenLHC-Anomaly Toolkit (OLAT)")

with st.sidebar:
    st.header("Configuration")
    data_config_path = st.text_input("Data config path", value="configs/cms_config.yaml")
    model_config_path = st.text_input("Model config path", value="configs/model_config.yaml")
    run_outdir = st.text_input("Run output dir", value="results/runs/olat_demo_run")
    plots_outdir = st.text_input("Plots output dir", value="results/plots/olat_demo_run")
    leaderboard_path = st.text_input("Leaderboard JSON", value="results/leaderboard.json")
    st.markdown("Use synthetic data automatically if no Parquet/ROOT found.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Train")
    if st.button("Run Training", type="primary"):
        with st.spinner("Training..."):
            result = run_training(Path(data_config_path), Path(model_config_path), Path(run_outdir))
        st.success("Training complete")
        st.json(result)

with col2:
    st.subheader("2) Evaluate")
    if st.button("Run Evaluation"):
        with st.spinner("Evaluating..."):
            result = run_evaluation(Path(run_outdir), Path(plots_outdir), Path(leaderboard_path))
        st.success("Evaluation complete")
        st.json(result)
        scores_csv = Path(plots_outdir) / "scores.csv"
        if scores_csv.exists():
            df = pd.read_csv(scores_csv)
            st.dataframe(df.head(50))
            st.download_button("Download scores.csv", data=scores_csv.read_bytes(), file_name="scores.csv")

st.divider()

st.caption("Tip: Edit configs to switch models (pca/iforest/ae/vae). Synthetic data is used if real data is missing.")
