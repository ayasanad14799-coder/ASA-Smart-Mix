import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. إعدادات الصفحة
st.set_page_config(page_title="Eco-Efficient Concrete AI | LCA & Economics", layout="wide", page_icon="🏗️")

# --- تنسيق الواجهة (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #004a99; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .header-box { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 2px solid #004a99; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .result-section { background-color: #eef2f6; padding: 20px; border-radius: 10px; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. الهوية الأكاديمية (الرأس) ---
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image("https://upload.wikimedia.org/wikipedia/ar/thumb/0/01/Mansoura_University_logo.png/200px-Mansoura_University_logo.png", width=130)

with col_title:
    st.markdown(f"""
    <div class="header-box">
        <h2 style="color: #004a99; margin-bottom:10px; line-height: 1.2;">Multi-criteria analysis of eco-efficient concrete from Technical, Environmental and Economic aspects</h2>
        <hr style="border: 0.5px solid #004a99; width: 60%; margin: auto;">
        <p style="margin-top:15px; font-size: 1.2em;"><b>By: Aya Mohammed Sanad Aboud</b></p>
        <p style="font-size: 1em; color: #444;">Under Supervision of: <br> <b>Prof. Ahmed Tahwia</b> & <b>Assoc. Prof. Asser El-Sheikh</b></p>
    </div>
    """, unsafe_allow_html=True)

# --- بوابة الدخول ---
if "auth" not in st.session_state: st.session_state.auth = False
if not st.session_state.auth:
    col_l, col_mid, col_r = st.columns([1, 2, 1])
    with col_mid:
        st.subheader("🔒 Secure Access Portal")
        with st.form("login"):
            pwd = st.text_input("Enter Access Code", type="password")
            if st.form_submit_button("Access Engine"):
                if pwd == "ASA2026": 
                    st.session_state.auth = True
                    st.rerun()
                else: st.error("❌ Invalid Code")
    st.stop()

# --- 3. تحميل الموديلات ---
@st.cache_resource
def load_assets():
    model = joblib.load('concrete_model .pkl')
    scaler = joblib.load('scaler_weights .pkl')
    return model, scaler

try:
    model, scaler = load_assets()

    # --- التبويبات ---
    tab_engine, tab_durability, tab_lca, tab_validation = st.tabs(["🚀 Technical Prediction", "🛡️ Durability", "🌍 LCA & Economics", "📊 Validation"])

    with st.sidebar:
        st.header("⚙️ Mix Design Inputs")
        cement = st.number_input("Cement", 100, 600, 350)
        water = st.number_input("Water", 100, 250, 175)
        nca = st.number_input("NCA", 500, 1500, 1050)
        nfa = st.number_input("NFA", 300, 1000, 750)
        rca_p = st.slider("RCA Replacement (%)", 0, 100, 0)
        rfa_p = st.slider("RFA Replacement (%)", 0, 100, 0)
        sf = st.number_input("Silica Fume", 0, 100, 0)
        fa = st.number_input("Fly Ash", 0, 200, 0)
        rha = st.number_input("RHA (%)", 0, 20, 0)
        fiber = st.number_input("Nylon Fiber", 0.0, 5.0, 0.0)
        sp = st.number_input("Superplasticizer", 0.0, 15.0, 2.5)
        wc = water/cement if cement > 0 else 0
        density = st.number_input("Target Density", 2000, 2600, 2400)
        
        st.divider()
        st.markdown("### 💰 Economic Adjustment")
        inflation = st.slider("Price Index Multiplier", 0.5, 2.0, 1.0, help="Adjusts cost based on market inflation")

    # إعداد المصفوفة (15 مدخلاً)
    input_arr = np.array([[cement, water, nca, nfa, rca_p, rfa_p, sf, fa, rha, fiber, sp, wc, 20, 100, density]])
    scaled = scaler.transform(input_arr)
    preds = model.predict(scaled)[0]

    with tab_engine:
        st.subheader("🎯 Mechanical Strength")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CS 7d (MPa)", f"{preds[0]:.2f}")
        c2.metric("CS 28d (MPa)", f"{preds[1]:.2f}", delta="± 2.34")
        c3.metric("CS 90d (MPa)", f"{preds[2]:.2f}")
        c4.metric("STS (MPa)", f"{preds[3]:.2f}")
        
        st.divider()
        st.subheader("📈 Strength Growth Curve")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(['7d', '28d', '90d'], [preds[0], preds[1], preds[2]], marker='o', color='#004a99')
        st.pyplot(fig)

    with tab_durability:
        st.subheader("🛡️ Durability & Physical Properties")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("FS (MPa)", f"{preds[4]:.2f}")
        d2.metric("EM (GPa)", f"{preds[5]:.2f}")
        d3.metric("Water Abs. (%)", f"{preds[6]:.2f}")
        d4.metric("UPV (km/s)", f"{preds[7]:.2f}")
        
        st.divider()
        st.markdown("#### Additional Durability Indices")
        k1, k2, k3 = st.columns(3)
        k1.metric("Shrinkage (µε)", f"{preds[8]:.0f}")
        k2.metric("Carb. Depth (mm)", f"{preds[9]:.2f}")
        k3.metric("Cl. Perm (Coulombs)", f"{preds[10]:.0f}")

    with tab_lca:
        st.subheader("🌍 Environmental Impact (LCA)")
        e1, e2, e3 = st.columns(3)
        e1.metric("CO2 Footprint (kg/m³)", f"{preds[11]:.2f}")
        e2.metric("Energy Consumption", f"{preds[12]:.2f}")
        e3.metric("Sustainability Index", f"{preds[16]:.3f}")
        
        st.divider()
        st.subheader("💰 Economic Analysis")
        base_cost = preds[13]
        final_cost = base_cost * inflation
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Base Cost (Predicted)", f"${base_cost:.2f}")
        m2.metric("Adjusted Cost", f"${final_cost:.2f}", delta=f"{((inflation-1)*100):.0f}% Inflation")
        m3.metric("ACV Value", f"{preds[14]:.2f}")

    with tab_validation:
        st.header("📊 Model Integrity")
        st.write("Validation metrics based on 400 experimental samples.")
        st.metric("R-Squared (R²)", "0.941")
        st.image("https://via.placeholder.com/600x300?text=Scatter+Plot+Placeholder")

except Exception as e:
    st.error(f"Error: {e}")

st.markdown("---")
st.caption("© 2026 | Multi-criteria analysis of eco-efficient concrete | Mansoura University")
