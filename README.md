# 📊 AI-Powered Data Analysis Web App
[![CI](https://github.com/sanchitghule17/Ai_Powered_data_analysis_web_app/actions/workflows/ci.yml/badge.svg)](https://github.com/sanchitghule17/Ai_Powered_data_analysis_web_app/actions/workflows/ci.yml)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_cloud.svg)](https://aipowereddataanalysiswebapp-dnu7hztzf4rc4ztdk4ntvh.streamlit.app/)

Live demo → **https://aipowereddataanalysiswebapp-ublapprbdmaarkbqtls9dm7.streamlit.app/**  
GitHub repo → **https://github.com/sanchitghule17/Ai_Powered_data_analysis_web_app**


A one-click Streamlit dashboard that transforms any CSV / Excel file into an **EDA → AutoML → Hyper-parameter tuning → SHAP explainability** workflow.  
Designed as a portfolio-grade project for data-science students and a template for production analytics tools.

---

## 🚀 Quick start (local)


Open `http://localhost:8501`, upload the sample Titanic CSV and explore.

---

## ✨ Feature overview

| Category              | Details                                                                   |
| --------------------- | ------------------------------------------------------------------------- |
| **Automatic EDA**     | Column summary, data types, pair plots, missing-value heatmap             |
| **Pre-processing**    | Numeric/ordinal encoding, imputation, scaling                             |
| **Baseline models**   | Linear Regression, Random Forest (Classifier/Regressor)                   |
| **Hyper-parameter tuning** | 30-trial Optuna TPE optimisation (user-tunable)                   |
| **Explainability**    | SHAP beeswarm plots via a checkbox                                        |
| **CI / Tests**        | Pytest suite + GitHub Actions (green badge above)                         |
| **Container ready**   | Dockerfile for reproducible builds (`docker build -t ai-eda-app .`)       |

---

## 🛠️ Tech stack

| Layer      | Library / Tool |
| ---------- | -------------- |
| Web UI     | Streamlit 1.32 |
| ML core    | scikit-learn 1.4 |
| Tuning     | Optuna 3.6 |
| Explainability | SHAP 0.45 + matplotlib 3.9 |
| Packaging  | Docker |
| CI         | GitHub Actions |

---

## 🐳 Docker usage



---

## ☁️ Deploy on Streamlit Cloud (free)

1. Push the repo to GitHub.  
2. Log in to [streamlit.io](https://streamlit.io) → **New app**.  
3. Select `<USER>/<REPO>` and path `ai_eda_app/app.py`.  
4. Click **Deploy** – the platform builds from `requirements.txt` and serves the app.

---

## 📂 Project structure


---

## 🧪 Running tests


| Test file            | What it verifies                                   |
| -------------------- | --------------------------------------------------- |
| `test_modeling.py`   | `train_and_evaluate` returns correct shapes & best-model name |
| `test_tuning.py`     | `optimise_rf` completes a 5-trial study and yields params + float score |

---

## ➕ Contributing

1. Fork, clone, create a feature branch.  
2. Run `black ai_eda_app` and `ruff .` before committing.  
3. Open a pull request – CI must stay green.  
Ideas welcome: CatBoost support, SHAP force plots, FastAPI prediction endpoint.

---

## 📑 License

Distributed under the MIT License – do anything you like, just give credit and don’t blame me if it breaks.

