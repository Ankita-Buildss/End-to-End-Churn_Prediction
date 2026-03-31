import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="⚡",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background-color: #0a0c10 !important;
    color: #e8eaf0 !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

.block-container { padding: 2.5rem 2rem 4rem; max-width: 820px; }

/* Section headers */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #00e5ff;
    padding-bottom: 8px;
    border-bottom: 1px solid #222733;
    margin-bottom: 4px;
}

/* Cards */
.card {
    background: #111318;
    border: 1px solid #222733;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,229,255,0.35), transparent);
}

/* Result boxes */
.result-safe {
    background: rgba(0,229,160,0.07);
    border: 1px solid rgba(0,229,160,0.3);
    border-radius: 16px;
    padding: 28px;
    margin-top: 20px;
    border-top: 3px solid #00e5a0;
}
.result-danger {
    background: rgba(255,77,109,0.07);
    border: 1px solid rgba(255,77,109,0.3);
    border-radius: 16px;
    padding: 28px;
    margin-top: 20px;
    border-top: 3px solid #ff4d6d;
}

.verdict-safe  { color: #00e5a0; font-size: 22px; font-weight: 800; }
.verdict-danger{ color: #ff4d6d; font-size: 22px; font-weight: 800; }
.verdict-sub   { font-family: 'DM Mono', monospace; font-size: 12px; color: #6b7280; margin-top: 4px; }
.prob-num-safe  { font-family: 'DM Mono', monospace; font-size: 42px; font-weight: 500; color: #00e5a0; }
.prob-num-danger{ font-family: 'DM Mono', monospace; font-size: 42px; font-weight: 500; color: #ff4d6d; }

/* Streamlit widget overrides */
div[data-testid="stSelectbox"] select,
div[data-testid="stNumberInput"] input {
    background-color: #181c24 !important;
    border: 1px solid #222733 !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Mono', monospace !important;
}
div[data-testid="stSlider"] > div > div > div {
    background-color: #00e5ff !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: #6b7280 !important;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00e5ff, #a78bfa) !important;
    color: #0a0c10 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 15px !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 0 !important;
    width: 100% !important;
    box-shadow: 0 4px 24px rgba(0,229,255,0.2) !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    box-shadow: 0 8px 32px rgba(0,229,255,0.35) !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model = tf.keras.models.load_model('model.h5')
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        ohe_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    return model, ohe_geo, scaler, le_gender

model, onehotencoder_geo, scaler, labelencoder_gender = load_artefacts()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:40px">
  <div style="width:52px;height:52px;background:linear-gradient(135deg,#00e5ff,#a78bfa);
              border-radius:14px;display:flex;align-items:center;justify-content:center;
              font-size:26px;box-shadow:0 0 24px rgba(0,229,255,0.25);flex-shrink:0">⚡</div>
  <div>
    <div style="font-size:26px;font-weight:800;letter-spacing:-0.5px">
      Customer <span style="color:#00e5ff">Churn</span> Prediction
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:11px;color:#6b7280;margin-top:4px;letter-spacing:0.5px">
      // NEURAL NETWORK CLASSIFIER · ANN MODEL
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Section 01: Customer Profile ───────────────────────────────────────────────
st.markdown('<div class="section-label">01 — Customer Profile</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    geography = st.selectbox("Geography", onehotencoder_geo.categories_[0])
with col2:
    gender = st.selectbox("Gender", labelencoder_gender.classes_)
with col3:
    age = st.slider("Age", 18, 100, 35)

col4, col5 = st.columns(2)
with col4:
    tenure = st.slider("Tenure (years)", 0, 10, 5)
with col5:
    num_of_products = st.slider("Number of Products", 1, 4, 1)

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 02: Financial Data ─────────────────────────────────────────────────
st.markdown('<div class="section-label">02 — Financial Data</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

col6, col7, col8 = st.columns(3)
with col6:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
with col7:
    balance = st.number_input("Balance ($)", min_value=0.0, value=0.0, step=100.0)
with col8:
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0)

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 03: Account Status ─────────────────────────────────────────────────
st.markdown('<div class="section-label">03 — Account Status</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

col9, col10 = st.columns(2)
with col9:
    has_credit_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x else "No")
with col10:
    is_active_member = st.selectbox("Active Member", [1, 0], format_func=lambda x: "Yes" if x else "No")

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ────────────────────────────────────────────────────────────────────
predict = st.button("RUN PREDICTION →")

if predict:
    # Build input dataframe
    input_data = {
        'CreditScore':      credit_score,
        'Gender':           labelencoder_gender.transform([gender])[0],
        'Age':              age,
        'Tenure':           tenure,
        'Balance':          balance,
        'NumOfProducts':    num_of_products,
        'HasCrCard':        has_credit_card,
        'IsActiveMember':   is_active_member,
        'EstimatedSalary':  estimated_salary,
    }

    geo_encoded    = onehotencoder_geo.transform([[geography]])
    geo_encoded_df = pd.DataFrame(
        geo_encoded.toarray(),
        columns=onehotencoder_geo.get_feature_names_out(['Geography'])
    )
    input_df    = pd.DataFrame([input_data])
    final_input = pd.concat(
        [input_df.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)],
        axis=1
    )
    scaled_input      = scaler.transform(final_input)
    prediction_proba  = float(model.predict(scaled_input)[0][0])
    churn_pct         = prediction_proba * 100

    # Risk factor tags
    risk_factors = []
    if age > 45:               risk_factors.append("AGE > 45")
    if not is_active_member:   risk_factors.append("INACTIVE")
    if num_of_products > 2:    risk_factors.append("OVER-BANKED")
    if credit_score < 500:     risk_factors.append("LOW CREDIT")
    if balance > 100_000:      risk_factors.append("HIGH BALANCE")
    if geography == "Germany": risk_factors.append("GEO: DE")
    if not has_credit_card:    risk_factors.append("NO CC")
    tag_html = " ".join(
        f'<span style="font-family:DM Mono,monospace;font-size:10px;padding:3px 10px;'
        f'border-radius:6px;border:1px solid #222733;color:#6b7280;background:#181c24">{t}</span>'
        for t in (risk_factors or ["NO MAJOR FLAGS"])
    )

    if prediction_proba > 0.5:
        st.markdown(f"""
        <div class="result-danger">
          <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px">
            <div style="width:52px;height:52px;background:rgba(255,77,109,0.12);border-radius:14px;
                        display:flex;align-items:center;justify-content:center;font-size:26px">⚠️</div>
            <div>
              <div class="verdict-danger">Likely to Churn</div>
              <div class="verdict-sub">// Intervention recommended · High risk segment</div>
            </div>
          </div>
          <div style="font-family:'DM Mono',monospace;font-size:11px;letter-spacing:1.5px;
                      text-transform:uppercase;color:#6b7280;margin-bottom:6px">Churn Probability</div>
          <div class="prob-num-danger">{churn_pct:.0f}%</div>
          <div style="height:8px;background:#181c24;border-radius:8px;border:1px solid #222733;
                      margin:10px 0 16px;overflow:hidden">
            <div style="width:{churn_pct:.0f}%;height:100%;border-radius:8px;
                        background:linear-gradient(90deg,#ff4d6d,rgba(255,77,109,0.5))"></div>
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap">{tag_html}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
          <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px">
            <div style="width:52px;height:52px;background:rgba(0,229,160,0.1);border-radius:14px;
                        display:flex;align-items:center;justify-content:center;font-size:26px">✅</div>
            <div>
              <div class="verdict-safe">Retention Likely</div>
              <div class="verdict-sub">// Customer stable · Low churn signal</div>
            </div>
          </div>
          <div style="font-family:'DM Mono',monospace;font-size:11px;letter-spacing:1.5px;
                      text-transform:uppercase;color:#6b7280;margin-bottom:6px">Churn Probability</div>
          <div class="prob-num-safe">{churn_pct:.0f}%</div>
          <div style="height:8px;background:#181c24;border-radius:8px;border:1px solid #222733;
                      margin:10px 0 16px;overflow:hidden">
            <div style="width:{churn_pct:.0f}%;height:100%;border-radius:8px;
                        background:linear-gradient(90deg,#00e5a0,rgba(0,229,160,0.4))"></div>
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap">{tag_html}</div>
        </div>
        """, unsafe_allow_html=True)
