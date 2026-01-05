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
    .run-btn { text-align: center; margin: 20px 0; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. الهوية الأكاديمية ---
st.markdown(f"""
    <div class="header-box">
        <h2 style="color: #004a99; margin-bottom:10px; line-height: 1.2;">Multi-criteria analysis of eco-efficient concrete from Technical, Environmental and Economic aspects</h2>
        <p style="margin-top:15px; font-size: 1.2em;"><b>Prepared by: Aya Mohammed Sanad Aboud</b></p>
        <p style="font-size: 1em; color: #444;">Supervision: <b>Prof. Ahmed Tahwia</b> & <b>Assoc. Prof. Asser El-Sheikh</b></p>
    </div>
    """, unsafe_allow_html=True)

# --- بوابة الدخول ---
if "auth" not in st.session_state: st.session_state.auth = False
if not st.session_state.auth:
    pwd = st.text_input("Enter Access Code", type="password")
    if st.button("Access Engine"):
        if pwd == "ASA2026": st.session_state.auth = True; st.rerun()
    st.stop()

# --- 3. تحميل الموديلات ---
@st.cache_resource
def load_assets():
    model = joblib.load('concrete_model .pkl')
    scaler = joblib.load('scaler_weights .pkl')
    return model, scaler

model, scaler = load_assets()

# --- القائمة الجانبية للمدخلات ---
with st.sidebar:
    st.header("⚙️ Mix Composition (kg/m³)")
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
    density = st.number_input("Density", 2000, 2600, 2400)
    
    st.divider()
    st.markdown("### 💰 Cost Adjuster")
    inflation = st.slider("Price Index Multiplier", 0.5, 2.5, 1.0)

# --- زر التشغيل الرئيسي ---
st.markdown('<div class="run-btn">', unsafe_allow_html=True)
run_analysis = st.button("🚀 Run Multi-Criteria Analysis", use_container_width=True, type="primary")
st.markdown('</div>', unsafe_allow_html=True)

if run_analysis:
    # حساب W/C والتنبؤ
    wc = water/cement if cement > 0 else 0
    input_arr = np.array([[cement, water, nca, nfa, rca_p, rfa_p, sf, fa, rha, fiber, sp, wc, 20, 100, density]])
    scaled = scaler.transform(input_arr)
    preds = model.predict(scaled)[0]

    # --- عرض النتائج المجمعة ---
    t1, t2, t3, t4 = st.tabs(["🏗️ Technical", "🛡️ Durability", "🌍 LCA & Economics", "📊 Database"])

    with t1:
        st.subheader("Mechanical Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CS 7d", f"{preds[0]:.2f} MPa")
        c2.metric("CS 28d", f"{preds[1]:.2f} MPa", delta="± 2.34")
        c3.metric("CS 90d", f"{preds[2]:.2f} MPa")
        c4.metric("STS", f"{preds[3]:.2f} MPa")
        
        # رسم بياني لتطور المقاومة
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(['7d', '28d', '90d'], [preds[0], preds[1], preds[2]], marker='s', color='#1e3a8a', linewidth=2)
        ax.set_title("Strength Growth Profile")
        ax.set_ylabel("MPa")
        st.pyplot(fig)

    with t2:
        st.subheader("Durability & Service Life Indicators")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("FS (MPa)", f"{preds[4]:.2f}")
        d2.metric("EM (GPa)", f"{preds[5]:.2f}")
        d3.metric("Water Abs.", f"{preds[6]:.2f} %")
        d4.metric("UPV", f"{preds[7]:.2f} km/s")
        
        st.divider()
        k1, k2, k3 = st.columns(3)
        k1.metric("Shrinkage", f"{preds[8]:.0f} µε")
        k2.metric("Carb. Depth", f"{preds[9]:.2f} mm")
        k3.metric("Cl. Permeability", f"{preds[10]:.0f} Coulombs")

    with t3:
        st.subheader("Sustainability & Economic Multi-Criteria")
        e1, e2, e3 = st.columns(3)
        e1.metric("CO2 Footprint", f"{preds[11]:.2f} kg/m³")
        e2.metric("Energy Consumption", f"{preds[12]:.2f} MJ")
        e3.metric("Sustainability Index", f"{preds[16]:.3f}")
        
        st.divider()
        base_cost = preds[13]
        final_cost = base_cost * inflation
        m1, m2, m3 = st.columns(3)
        m1.metric("Base Cost", f"${base_cost:.2f}")
        m2.metric("Adjusted Cost", f"${final_cost:.2f}")
        m3.metric("ACV", f"{preds[14]:.2f}")

    with t4:
        st.subheader("Reference Data Sample")
        df_sample = pd.read_csv('Database_Inputs jimini2.csv', sep=';')
        st.dataframe(df_sample.head(20))
else:
    st.info("👈 Please adjust your mix proportions in the sidebar and click 'Run Multi-Criteria Analysis' to see results.")
