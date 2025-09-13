You are an expert in full-stack Python, data science, and HEP tooling.
Build an open-source project called OpenLHC-Anomaly that lets researchers and enthusiasts run anomaly detection on CERN Open Data with a simple web interface.

Requirements:
1) Backend & Data Pipeline
   - Python 3.10+, libraries: uproot, awkward, numpy, pandas, scikit-learn, matplotlib, plotly, streamlit, and torch OR tensorflow.
   - Add data loader to fetch CMS Open Data (ROOT) or local files; optional ROOT→Parquet export.
   - Preprocess to physics-friendly features (pT, η, φ, m_inv, jet counts, MET).
   - Models: PCA, IsolationForest, Autoencoder, Variational Autoencoder.
   - Metrics: ROC/AUC, Average Precision, discovery significance (Z_bi).
   - Configs (YAML) for dataset + model params; fixed seeds and logging.

2) Web Interface (Streamlit)
   - Dataset selector, model selector, hyperparameter sliders.
   - “Run Analysis” triggers train/eval; show metrics, plots (histograms, score distributions, 2D embeddings).
   - Table of top-N anomalous events; CSV export for scores.
   - Multipage UI (optional): Dataset, Models, Results.

3) Project Structure
   - Mirror the repo tree in this README (backend/, webapp/, configs/, notebooks/, results/…).
   - Provide scripts: train.py, evaluate.py, export_parquet.py, prepare_data.py.
   - Provide notebooks/01_quickstart_cms.ipynb (Colab-ready).
   - Include requirements.txt, README.md sections, and CITATION.cff stub.

4) Deliverables & Quality
   - Clean, modular code (PEP8), type hints where helpful, docstrings.
   - Sensible defaults for non-technical users; tooltips in UI.
   - Save artifacts under results/; maintain leaderboard.json.
   - Clear exceptions and friendly error messages.

5) Deployment
   - Add notes to deploy on Streamlit Cloud and Hugging Face Spaces.

Output:
- All files in logical order, with code blocks ready to paste into the repo.
- Minimal end-to-end working example for CMS sample (toy-sized) with AE baseline and functioning Streamlit app.
