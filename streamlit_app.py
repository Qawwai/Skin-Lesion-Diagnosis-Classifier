"""
Streamlit prediction app for the PAD-UFES-20 multi-class skin-lesion classifier.

Run with:
    pip install streamlit lightgbm shap pandas scikit-learn matplotlib
    streamlit run streamlit_app.py

It loads the LightGBM model exported from the project notebook
(lesion_model.pkl, label_encoder.pkl, feature_spec.json), takes a lesion's
metadata + symptom checklist through the sidebar, applies the SAME preprocessing
used in training, and returns the predicted diagnosis with a probability
breakdown and a per-prediction SHAP explanation.

Author: Theodore Furui Widyatmoko (2025313608)
"""
import json, pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🔬", layout="wide")

# ---- Professional neutral theme (no blue-only) ----
ACCENT = "#3F7C85"
PAL = ['#5B6770', '#3F7C85', '#7A8450', '#C28A3C', '#B25D4C', '#876C8E']

CLASS_FULL = {
    'BCC': 'Basal Cell Carcinoma',
    'MEL': 'Melanoma',
    'SCC': 'Squamous Cell Carcinoma',
    'ACK': 'Actinic Keratosis',
    'NEV': 'Nevus (mole)',
    'SEK': 'Seborrheic Keratosis',
}
MALIGNANT = {'BCC', 'MEL', 'SCC'}


@st.cache_resource
def load_artifacts():
    with open('lesion_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('feature_spec.json') as f:
        spec = json.load(f)
    return model, le, spec


def preprocess(df, spec):
    """Identical transform to the training notebook — no train/serve skew."""
    bool_cols, sym_cols = spec['bool_cols'], spec['sym_cols']
    cat_cols, num_cols = spec['cat_cols'], spec['num_cols']
    df = df.copy()
    df['clinical_only'] = df['smoke'].isna().astype(int)
    for col in bool_cols:
        df[col] = df[col].map({True: 1, False: 0, 'True': 1, 'False': 0})
    for col in sym_cols:
        df[col] = df[col].map({'True': 1, 'False': 0, 'UNK': -1}).fillna(-1)
    df['area'] = df['diameter_1'] * df['diameter_2']
    df['symptom_count'] = df[sym_cols].apply(lambda r: (r == 1).sum(), axis=1)
    df['aspect'] = df['diameter_1'] / (df['diameter_2'] + 1e-6)
    for col in cat_cols:
        df[col] = df[col].fillna('MISSING').astype(str)
    for col in cat_cols:
        df[col] = df[col].astype('category')
    return df[spec['features']]


# ============ HEADER ============
st.title("🔬 Skin Lesion Diagnosis Classifier")
st.caption(
    "PAD-UFES-20 · LightGBM + CatBoost + XGBoost ensemble · "
    "Fundamentals of Machine Learning (SWE3050) final project — Theodore Furui Widyatmoko"
)
st.markdown(
    "Enter a lesion's clinical details in the sidebar and the model predicts its "
    "diagnostic category among six classes, with a SHAP explanation of *why*."
)

model, le, spec = load_artifacts()
CLASSES = spec['classes']

# ============ SIDEBAR INPUTS ============
st.sidebar.header("Lesion & patient details")

st.sidebar.subheader("Patient")
age = st.sidebar.slider("Age", 0, 100, 60)
gender = st.sidebar.selectbox("Gender", ["MALE", "FEMALE", "MISSING"])
region = st.sidebar.selectbox(
    "Body region",
    ["FACE", "FOREARM", "ARM", "CHEST", "BACK", "NECK", "NOSE", "HAND",
     "SCALP", "EAR", "THIGH", "ABDOMEN", "FOOT", "LIP", "MISSING"],
)
fitspatrick = st.sidebar.slider("Fitzpatrick skin type", 1, 6, 3)

st.sidebar.subheader("Lesion geometry")
diameter_1 = st.sidebar.number_input("Diameter 1 (mm)", 0.0, 100.0, 8.0, step=0.5)
diameter_2 = st.sidebar.number_input("Diameter 2 (mm)", 0.0, 100.0, 6.0, step=0.5)

st.sidebar.subheader("Symptoms")
def sym_input(label):
    v = st.sidebar.selectbox(label, ["False", "True", "UNK"], key=label)
    return v
itch = sym_input("Itches")
grew = sym_input("Recently grew")
hurt = sym_input("Hurts")
changed = sym_input("Changed")
bleed = sym_input("Bleeds")
elevation = sym_input("Raised / elevated")

st.sidebar.subheader("History & lifestyle")
biopsed = st.sidebar.checkbox("Was biopsied", value=True)
skin_cancer_history = st.sidebar.checkbox("Personal skin-cancer history")
cancer_history = st.sidebar.checkbox("Family cancer history")
smoke = st.sidebar.checkbox("Smoker")
drink = st.sidebar.checkbox("Drinks alcohol")
pesticide = st.sidebar.checkbox("Pesticide exposure")
has_piped_water = st.sidebar.checkbox("Has piped water", value=True)
has_sewage_system = st.sidebar.checkbox("Has sewage system", value=True)
bg_father = st.sidebar.text_input("Father's background", "POMERANIA")
bg_mother = st.sidebar.text_input("Mother's background", "POMERANIA")

# ============ BUILD A ROW ============
# If a lesion wasn't biopsied, the clinical block is missing in real data — emulate that
clinical_block_present = biopsed
row = {
    'smoke': smoke if clinical_block_present else np.nan,
    'drink': drink if clinical_block_present else np.nan,
    'pesticide': pesticide if clinical_block_present else np.nan,
    'gender': gender if clinical_block_present else np.nan,
    'fitspatrick': fitspatrick if clinical_block_present else np.nan,
    'diameter_1': diameter_1 if clinical_block_present else np.nan,
    'diameter_2': diameter_2 if clinical_block_present else np.nan,
    'background_father': bg_father,
    'background_mother': bg_mother,
    'age': age,
    'skin_cancer_history': skin_cancer_history,
    'cancer_history': cancer_history,
    'has_piped_water': has_piped_water,
    'has_sewage_system': has_sewage_system,
    'region': region,
    'itch': itch, 'grew': grew, 'hurt': hurt,
    'changed': changed, 'bleed': bleed, 'elevation': elevation,
    'biopsed': biopsed,
}
raw = pd.DataFrame([row])
X = preprocess(raw, spec)

# ============ PREDICT ============
if st.button("Predict diagnosis", type="primary"):
    proba = model.predict_proba(X)[0]
    pred_idx = int(proba.argmax())
    pred = CLASSES[pred_idx]
    conf = proba[pred_idx]

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.subheader("Prediction")
        full = CLASS_FULL.get(pred, pred)
        tag = "🔴 Malignant" if pred in MALIGNANT else "🟢 Benign / low-risk"
        st.metric(label=full + f"  ({pred})", value=f"{conf*100:.1f}%")
        st.markdown(f"**Category:** {tag}")
        if pred in MALIGNANT:
            st.warning(
                "Model leans toward a malignant category. This is a course "
                "project, not a medical device — always confirm with a clinician."
            )
        else:
            st.info(
                "Model leans toward a benign / low-risk category. Still a course "
                "project, not a diagnosis."
            )

    with col2:
        st.subheader("Probability across all six classes")
        order_idx = np.argsort(proba)[::-1]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        labels = [CLASSES[i] for i in order_idx]
        vals = [proba[i] for i in order_idx]
        colors = [PAL[1] if CLASSES[i] == pred else '#C7CDD1' for i in order_idx]
        ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        for i, v in enumerate(vals[::-1]):
            ax.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    # ============ SHAP LOCAL EXPLANATION ============
    st.subheader("Why? — SHAP explanation for this prediction")
    st.caption(
        "Each bar shows how a feature pushed the model toward (right) or away from "
        f"(left) the predicted class **{pred}**, starting from the model's baseline."
    )
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)            # (1, n_feat, n_class)
    base = explainer.expected_value
    expl = shap.Explanation(
        values=np.array(sv)[0, :, pred_idx],
        base_values=base[pred_idx],
        data=X.iloc[0].values,
        feature_names=spec['features'],
    )
    fig2 = plt.figure()
    shap.plots.waterfall(expl, max_display=12, show=False)
    plt.tight_layout()
    st.pyplot(fig2)
else:
    st.info("Set the lesion details in the sidebar, then press **Predict diagnosis**.")

st.divider()
st.caption(
    "⚠️ Educational project only — not a diagnostic tool. The model is trained on "
    "tabular metadata from PAD-UFES-20 and is not a substitute for professional "
    "dermatological assessment."
)
