import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json

# 1. إعدادات الصفحة والهوية الأكاديمية
st.set_page_config(page_title="Eco-Concrete AI Optimizer", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #004a99; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .header-container { display: flex; align-items: center; justify-content: center; background-color: #f8f9fa; padding: 20px; border-radius: 15px; border: 2px solid #004a99; margin-bottom: 25px; }
    .logo-img { width: 100px; margin-right: 25px; }
    .footer-text { text-align: center; color: #666; font-size: 0.85em; margin-top: 50px; padding: 20px; border-top: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

# عرض الشعار وعنوان البحث
# الرابط المباشر لشعارك من GitHub
logo_url = "https://raw.githubusercontent.com/ayasanad14799-coder/ASA-Smart-Mix/1a458aafdcfcc51f7f6f3cb65a9437581dbb8f7f/download.jfif"

# عرض الشعار وعنوان البحث
st.markdown(f"""
    <div class="header-container">
        <img src="{logo_url}" style="width: 120px; margin-right: 25px; border-radius: 10px;">
        <div style="text-align: center;">
            <h2 style="color: #004a99; margin-bottom:5px;">Multi-criteria analysis of eco-efficient concrete from Technical, Environmental and Economic aspects</h2>
            <p style="font-size: 1.1em; margin-bottom:5px;"><b>Prepared by: Aya Mohammed Sanad Aboud</b></p>
            <p style="color: #666; margin-bottom:5px;">Supervision: <b>Prof. Ahmed Tahwia</b> & <b>Assoc. Prof. Asser El-Sheikh</b></p>
            <p style="color: #004a99;"><b>Mansoura University | Faculty of Engineering</b></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 2. نظام الدخول الآمن
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

# 3. تحميل الموديلات
@st.cache_resource
def load_assets():
    model = joblib.load('concrete_model.pkl')
    scaler = joblib.load('scaler_weights.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("Error: Model files not found. Ensure 'concrete_model.pkl' and 'scaler_weights.pkl' are on GitHub.")
    st.stop()

metrics_real = {"R2": 0.9557, "RMSE": 2.91, "COV": "6.16%"}

def send_to_sheets(data):
    url = "https://script.google.com/macros/s/AKfycby2DeRUQE87VDanU2wIS43tzbOCyGKLGLT-AU3yc4TtPBYQft-TZKvupbi3Aad03MK8/exec"
    try: requests.post(url, json=data, timeout=5)
    except: pass

# 4. القائمة الجانبية
with st.sidebar:
    st.header("⚙️ Mix Ingredients (kg/m³)")
    c = st.number_input("Cement", 100, 600, 350)
    w = st.number_input("Water", 100, 250, 175)
    nca = st.number_input("NCA", 500, 1500, 1050)
    nfa = st.number_input("NFA", 300, 1000, 750)
    rca = st.slider("RCA Replacement %", 0, 100, 0)
    rfa = st.slider("RFA Replacement %", 0, 100, 0)
    sf = st.number_input("Silica Fume", 0, 100, 0)
    fa = st.number_input("Fly Ash", 0, 200, 0)
    rha = st.number_input("Rice Husk Ash %", 0, 20, 0)
    fib = st.number_input("Nylon Fiber", 0.0, 5.0, 0.0)
    sp = st.number_input("Superplasticizer", 0.0, 15.0, 2.5)
    sz = st.selectbox("Max Agg Size (mm)", [10, 20, 40], index=1)
    sl = st.number_input("Slump (mm)", 0, 250, 100)
    den = st.number_input("Density", 2000, 2600, 2400)
    wc = w/c if c > 0 else 0
    inf = st.slider("Price Inflation", 0.5, 2.5, 1.0)
    run_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

# 5. التبويبات
t1, t2, t3, t4, t5 = st.tabs(["🏗️ Strength", "🛡️ Durability", "🌍 LCA & Econ", "💡 Optimizer", "📖 Technical Docs"])

if run_btn:
    # الهندسة والمنطق (Validation)
    if wc < 0.25 or wc > 0.65:
        st.sidebar.warning(f"⚠️ W/C Ratio ({wc:.2f}) is outside standard limits.")

    inp = np.array([[c, w, nca, nfa, rca, rfa, sf, fa, rha, fib, sp, wc, sz, sl, den]])
    p = model.predict(scaler.transform(inp))[0]
    
    # إرسال البيانات
    send_to_sheets({"c":c,"w":w,"nca":nca,"nfa":nfa,"rca":rca,"rfa":rfa,"sf":sf,"fa":fa,"rha":rha,"fib":fib,"sp":sp,"sz":sz,"sl":sl,"den":den,"wc":wc,
                    "p0":p[0],"p1":p[1],"p2":p[2],"p3":p[3],"p4":p[4],"p5":p[5],"p6":p[6],"p7":p[7],
                    "p8":p[8],"p9":p[9],"p10":p[10],"p11":p[11],"p12":p[12],"p13":p[13]*inf,"p14":p[14],"p16":p[16]})

    with t1:
        st.subheader("🎯 Predictive Performance")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy (R²)", f"{metrics_real['R2']*100:.2f}%")
        m2.metric("Mean Error", f"{metrics_real['RMSE']} MPa")
        m3.metric("COV", metrics_real['COV'])
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("CS 28d", f"{p[1]:.2f} MPa", f"±{metrics_real['RMSE']}")
        c2.metric("CS 7d", f"{p[0]:.2f} MPa")
        c3.metric("CS 90d", f"{p[2]:.2f} MPa")
        fig, ax = plt.subplots(figsize=(10, 3)); ax.plot(['7d', '28d', '90d'], [p[0], p[1], p[2]], marker='o'); st.pyplot(fig)

    with t2:
        st.subheader("🛡️ Durability Profile")
        st.write(f"**Elastic Modulus:** {p[5]:.2f} GPa | **Water Absorption:** {p[6]:.2f} %")
        st.write(f"**UPV Speed:** {p[7]:.2f} km/s | **Carbonation:** {p[9]:.2f} mm")

    with t3:
        st.subheader("🌍 Sustainability & Cost")
        st.metric("CO2 Footprint", f"{p[11]:.2f} kg/m³")
        st.metric("Sustainability Index", f"{p[16]:.3f}")
        st.metric("Adjusted Cost", f"${(p[13]*inf):.2f}")

with t4:
    st.header("💡 AI Optimizer")
    t_st = st.number_input("Target CS 28d (MPa)", 20, 80, 40)
    if st.button("Generate Green Mixes"):
        # (كود المحاكاة كما هو موجود سابقاً)
        st.info("Optimizer engine ready for simulation.")

with t5:
    st.header("📖 Technical Documentation & Research Scope")
    st.markdown(f"""
    * **Scientific Database:** This AI engine was developed using a comprehensive **Meta-Analysis** approach, trained on a global database of **400 experimental samples** sourced from diverse, peer-reviewed international research.
    * **Engineering Applicability:** The predictive model is optimized for **Eco-friendly Concrete** (incorporating RCA, RFA, SF, FA, and RHA) with a target compressive strength range of **20 MPa to 80 MPa**.
    * **Model Reliability:** Validated using Random Forest Regression with a correlation coefficient (**R² = {metrics_real['R2']}**) and a Coefficient of Variation (**COV = {metrics_real['COV']}**).
    * **Disclaimer:** This software is a **Decision Support Tool** for research. It does not replace mandatory laboratory trial mixes or structural compliance testing.
    * **Academic Affiliation:** Developed as part of a Master’s Thesis at **Mansoura University, Faculty of Engineering.**
    """)

st.markdown("""
    <div class="footer-text">
        © 2024 Mansoura University - Structural Engineering Department<br>
        AI for Sustainable Construction Materials
    </div>
""", unsafe_allow_html=True)
