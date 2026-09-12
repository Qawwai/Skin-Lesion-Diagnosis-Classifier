# Skin Lesion Classifier

A Streamlit app that classifies six skin lesion categories from patient details and symptoms, then shows the model's class probabilities and SHAP feature contributions.

Built as a final project for **Fundamentals of Machine Learning (SWE3050)** using tabular metadata from PAD-UFES-20. The app takes structured inputs; it does not analyze photographs.

## How it works

1. Enter patient details, lesion measurements, symptoms, and history in the sidebar.
2. The app encodes the inputs and derives features such as lesion area, aspect ratio, and symptom count.
3. Select **Predict diagnosis** to see the predicted category, probabilities across all six classes, and a local SHAP explanation.

The six classes are ACK (actinic keratosis), BCC (basal cell carcinoma), MEL (melanoma), NEV (nevus), SCC (squamous cell carcinoma), and SEK (seborrheic keratosis).

## Run locally

Clone the repository and run these commands from its root directory:

```bash
git clone https://github.com/Qawwai/Skin-Lesion-Diagnosis-Classifier.git
cd Skin-Lesion-Diagnosis-Classifier
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

The app loads the included model files from the current directory. Dependencies are currently unpinned, so compatibility with every package version has not been established.

## Repository contents

| File | Purpose |
| --- | --- |
| `streamlit_app.py` | Input form, preprocessing, prediction, and SHAP plots |
| `feature_spec.json` | Feature order, column groups, and class labels |
| `lesion_model.pkl` | Exported prediction model |
| `label_encoder.pkl` | Exported label encoder |
| `requirements.txt` | Python dependencies |

## Model and results

The original project summary reports **0.79 accuracy** and **0.71 macro-F1**, and describes a LightGBM, CatBoost, and XGBoost ensemble. The app's module documentation identifies the exported model as LightGBM; the app itself loads a single model artifact.

The training notebook, evaluation split, and experiment results are not included in this repository. The reported scores therefore cannot be reproduced from these files, and it is not yet documented whether they describe the exported model or the wider ensemble experiment.

## Limitations

This is an educational project, not a clinical diagnostic tool. Its outputs should not be used for medical decisions. SHAP plots describe the model's feature contributions, not medical causes.

## Author

[Theodore Furui Widyatmoko](https://github.com/Qawwai)
