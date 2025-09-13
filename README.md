# OpenLHC-Anomaly Toolkit (OLAT)

**Open-source toolkit + web UI for running anomaly detection on CERN Open Data.**
Built for researchers, educators, and science enthusiasts. No heavy setup required—run locally, on Colab, or deploy the web app for others.

---

## ✨ Why this exists

* **Researchers** need comparable baselines & easy reproducibility across public LHC datasets.
* **Educators** want plug-and-play labs grounded in real data, not toy examples.
* **Enthusiasts** should be able to explore collisions with a **web interface**—no ROOT expertise needed.

**OLAT** ships:

* A **Streamlit web app** (point-and-click): choose a dataset, pick a model (PCA/IF/AE/VAE), tune sliders, run, and download results.
* Reproducible **notebooks & scripts** for deeper work (Colab-ready).
* A modular backend using widely adopted HEP/DS stacks (**uproot**, **awkward**, **numpy/pandas**, **scikit-learn**, **PyTorch/TensorFlow**).

---

## 🗺️ Feature Map

* **Datasets**: loaders for CMS Open Data samples (extendable to LHC Olympics / Dark Machines).
* **I/O**: ROOT → columnar arrays; optional Parquet export for general DS tools.
* **Models**: PCA, Isolation Forest, Autoencoder (AE), Variational Autoencoder (VAE).
* **Metrics**: ROC/AUC, Average Precision (AP), discovery significance (**Z\_bi**).
* **Web UI**: dataset & model selection, hyperparameter sliders, run button, plots, CSV export.
* **Reproducibility**: configs (YAML), fixed seeds, environment specs.
* **Deploy**: local, Streamlit Cloud, or Hugging Face Spaces.

---

## 🏗️ Architecture

```
openlhc-anomaly/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ requirements.txt           # or environment.yml
├─ configs/
│  ├─ cms_config.yaml         # dataset paths & selections
│  └─ model_config.yaml       # model hyperparameters
├─ backend/
│  ├─ data_loader.py          # ROOT/Parquet readers, schema checks
│  ├─ preprocess.py           # feature engineering (pT, η, φ, m_inv, sums)
│  ├─ models/
│  │  ├─ baselines.py         # PCA, IsolationForest, KDE (optional)
│  │  ├─ autoencoder.py       # AE (torch or tf)
│  │  └─ vae.py               # Variational AE
│  ├─ train.py                # CLI: train + save artifacts
│  ├─ evaluate.py             # CLI: metrics, plots, leaderboard entries
│  └─ utils.py                # seeds, logging, paths, plotting helpers
├─ webapp/
│  ├─ app.py                  # Streamlit UI
│  ├─ pages/                  # optional multipage (Datasets, Models, Results)
│  └─ static/                 # logos, CSS
├─ notebooks/
│  └─ 01_quickstart_cms.ipynb # end-to-end demo
├─ data/                      # (created by you) raw/processed caches
├─ results/
│  ├─ runs/                   # model checkpoints, logs
│  ├─ plots/                  # figures saved from UI/CLI
│  └─ leaderboard.json        # standardized results
└─ scripts/
   ├─ prepare_data.py         # download/convert helpers
   └─ export_parquet.py       # optional ROOT→Parquet conversion
```

---

## 🖥️ Quick Start (10 minutes)

### 1) Install (local)

```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Configure dataset paths

Create/edit `configs/cms_config.yaml`:

```yaml
dataset_name: "cms_open_data_minimal"
root_files:
  - "data/cms/DoubleMuon.root"   # put your local paths or URLs
  # - "https://opendata.cern.ch/.../DoubleMuon.root"
features:
  - "pho1_pt"
  - "pho2_pt"
  - "diphoton_mass"
  - "jet_n"
  - "met"
split:
  train: 0.7
  val: 0.15
  test: 0.15
cache_parquet: "data/processed/cms_minimal.parquet"
```

### 3) Launch the web app

```bash
streamlit run webapp/app.py
```

Open the browser link shown (usually [http://localhost:8501](http://localhost:8501)).
Pick a dataset → choose a model (e.g., **Autoencoder**) → tune sliders → **Run Analysis** → view plots → **Download CSV** of anomaly scores.

### 4) Optional: one-click on Colab

* Open `notebooks/01_quickstart_cms.ipynb` in Google Colab.
* It installs dependencies and runs a small AE baseline end-to-end.

---

## 📦 Requirements

**Data/HEP I/O:** `uproot`, `awkward`
**DS/ML:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `plotly`, `torch` *or* `tensorflow` (pick one)
**Web UI:** `streamlit`
**Utilities:** `pyyaml`, `tqdm`, `joblib`

`requirements.txt` example:

```txt
uproot>=5.3
awkward>=2.6
numpy>=1.26
pandas>=2.2
scikit-learn>=1.5
matplotlib>=3.8
plotly>=5.20
streamlit>=1.36
pyyaml>=6.0
tqdm>=4.66
joblib>=1.4
torch>=2.3    # or tensorflow>=2.16 (choose one framework)
```

> If you prefer **conda**, provide an `environment.yml` with the same packages.

---

## 🔢 Data: where & how

1. **Obtain CMS Open Data** (public): download small ROOT samples suitable for education/benchmarks.
2. Place files under `data/cms/` or set direct URLs in `configs/cms_config.yaml`.
3. (Optional) Convert to Parquet for speed/interoperability:

```bash
python scripts/export_parquet.py \
  --input data/cms/DoubleMuon.root \
  --tree Events \
  --output data/processed/cms_minimal.parquet
```

**Minimal ROOT read snippet (what `data_loader.py` does):**

```python
import uproot, awkward as ak, numpy as np
with uproot.open("data/cms/DoubleMuon.root") as f:
    events = f["Events"].arrays(["pho1_pt","pho2_pt","diphoton_mass","jet_n","met"])
    # events is an Awkward Array; convert to NumPy/Pandas as needed
    diphoton = ak.to_numpy(events["diphoton_mass"])
    print("Mass shape:", diphoton.shape)
```

> Always review dataset DOIs, licenses, and usage notes. Cite appropriately.

---

## ⚙️ Configuration

### `configs/model_config.yaml` example

```yaml
model: "autoencoder"   # options: pca, iforest, ae, vae
seed: 42

pca:
  n_components: 4

iforest:
  n_estimators: 200
  max_samples: "auto"
  contamination: 0.01

ae:
  framework: "torch"
  hidden_dims: [64, 32, 16, 32, 64]
  activation: "relu"
  dropout: 0.0
  lr: 1.0e-3
  batch_size: 256
  epochs: 20

vae:
  framework: "torch"
  hidden_dims: [64, 32]
  latent_dim: 8
  beta: 1.0
  lr: 1.0e-3
  batch_size: 256
  epochs: 30
```

---

## 🧪 Train & Evaluate via CLI

Train:

```bash
python backend/train.py \
  --data-config configs/cms_config.yaml \
  --model-config configs/model_config.yaml \
  --outdir results/runs/ae_cms_v01
```

Evaluate:

```bash
python backend/evaluate.py \
  --run-dir results/runs/ae_cms_v01 \
  --outdir results/plots/ae_cms_v01 \
  --append-leaderboard results/leaderboard.json
```

This produces:

* **Plots** (ROC, precision-recall, score histograms)
* **CSV** of anomaly scores
* **leaderboard.json** entries like:

```json
[
  {
    "timestamp": "2025-09-13T15:30:00Z",
    "dataset": "cms_open_data_minimal",
    "model": "autoencoder",
    "metrics": { "auc": 0.87, "ap": 0.42, "z_bi": 3.1 },
    "config_hash": "ae_torch_h64-32-16_lr1e-3_ep20",
    "notes": "baseline run"
  }
]
```

---

## 📊 Metrics (incl. **Z\_bi**)

* **ROC/AUC**: standard trade-off between TPR/FPR.
* **Average Precision (AP)**: threshold-free summary of PR curve.
* **Discovery significance (Z\_bi)**: commonly used in HEP for counting experiments; implemented with background uncertainty option.
  In OLAT, `metrics.py` provides:

  ```python
  z = discovery_significance(signal, background, sigma_b=None)
  ```

---

## 🌐 Web UI Guide (Streamlit)

1. **Start**: `streamlit run webapp/app.py`
2. **Select dataset**: either a local Parquet cache or direct ROOT via loader.
3. **Choose model**: PCA, IF, AE, VAE.
4. **Tune sliders**: epochs, learning rate, latent dim, contamination, etc.
5. **Run Analysis**: training + evaluation; progress shown in sidebar.
6. **Inspect**:

   * Metrics panel (AUC/AP/Z\_bi)
   * Score histograms, scatter/UMAP views
   * **Top-N anomalous events** table
7. **Export**: “Download CSV” → anomaly scores with event indices.

> Designed for **non-technical** users: defaults are sensible, with tooltips and inline notes.

---

## 🚀 Deployments

### A) Local (recommended for first use)

```
streamlit run webapp/app.py
```

### B) Streamlit Cloud

* Connect your GitHub repo.
* Add required secrets if you fetch remote datasets.
* Set main script: `webapp/app.py`.

### C) Hugging Face Spaces

* Space type: **Streamlit**.
* `requirements.txt` included; push repo.
* Cache small sample files or instruct users to upload.

---

## 📚 Reproducibility & Publishing

* Fix seeds (`utils.py`) and store configs alongside results.
* Include **CITATION.cff** with dataset DOIs and software versions.
* For **educational papers/blogs**, export plots from `results/plots/`.
* Publish snapshots to **Zenodo** (gets a DOI) and link in README.

---

## 🔌 Extending OLAT

### Add a new dataset

1. Implement a loader in `backend/data_loader.py` (or `datasets/your_dataset.py`).
2. Provide a data card: schema, features, selection cuts, license/DOI.
3. Add an example config in `configs/`.

### Add a new model

1. Create `backend/models/your_model.py` with `fit(X_train)`, `score_samples(X)` or `predict_proba`.
2. Register in `train.py` / `evaluate.py`.
3. Add UI controls in `webapp/app.py` (sliders, selects).

### Add a metric

1. Implement in `backend/metrics.py`.
2. Call from `evaluate.py` and surface in the UI.

---

## 🧑‍🏫 Education Mode (recommended)

* Use the included **Quickstart** notebook in class or workshops.
* Encourage students to change one thing (e.g., AE depth) and compare AUC/Z\_bi.
* Export their runs to the **leaderboard** for a collaborative experiment.

---

## 🔐 Legal, Ethics, & Safety

* Use public datasets under their licenses; **always cite DOIs**.
* Don’t claim new physics based on educational subsets alone—contact relevant collaborations for proper review if you find something intriguing.
* Avoid uploading large ROOT files to public repos; provide download scripts instead.

---

## 🧰 Troubleshooting

* **Streamlit can’t find data**: verify `configs/cms_config.yaml` paths/URLs; try converting to Parquet.
* **CUDA errors**: switch AE/VAE to CPU (`device="cpu"`) or install correct CUDA wheel.
* **Memory issues**: reduce batch size; use Parquet; down-select features.
* **ROOT read issues**: check `tree` name, branch names; update `uproot`.

---

## 🗺️ Roadmap

* v0.1: CMS loader, PCA/IF/AE/VAE, Streamlit UI, metrics, leaderboard.
* v0.2: Add LHC Olympics R\&D dataset + ready baselines.
* v0.3: Optional `root→parquet` CLI and dataset data cards.
* v0.4: Advanced models (e.g., VAE-flow, GNN prototype), richer event visuals.

---

## 🤝 Contributing

PRs welcome! Please:

* Add tests/tiny fixtures for loaders and models.
* Document new features (README + example config).
* Keep the UI friendly (tooltips, sensible defaults).

---

## 📜 License

Apache-2.0. See `LICENSE`.

---

## 👏 Acknowledgements

Thanks to the open-data and Scikit-HEP communities for the tools that make public HEP analysis possible.

---
