import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json

# 1. إعدادات الصفحة والهوية الأكاديمية
st.set_page_config(page_title="Eco-Concrete AI Optimizer", layout="wide")

# تنسيق CSS مخصص لإظهار الشعار والعنوان بشكل متناسق
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #004a99; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .header-container { 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 15px; 
        border: 2px solid #004a99; 
        margin-bottom: 25px;
    }
    .logo-img { width: 100px; margin-right: 25px; }
    .header-text { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# عرض الشعار وعنوان الرسالة (رابط الشعار رسمي لجامعة المنصورة)
st.markdown(f"""
    <div class="header-container">
        <img src="https://upload.wikimedia.org/wikipedia/ar/thumb/0/01/Mansoura_University_logo.png/200px-Mansoura_University_logo.png" class="logo-img">
        <div class="header-text">
            <h2 style="color: #004a99; margin-bottom:5px;">Multi-criteria analysis of eco-efficient concrete from Technical, Environmental and Economic aspects</h2>
            <p style="font-size: 1.2em; margin-bottom:5px;"><b>Prepared by: Aya Mohammed Sanad Aboud</b></p>
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

# باقي الكود كما هو (الموديلات والإرسال للجوجل شيت والنتائج)
@st.cache_resource
def load_assets():
    model = joblib.load('concrete_model .pkl')
    scaler = joblib.load('scaler_weights .pkl')
    return model, scaler
model, scaler = load_assets()

def send_to_sheets(data):
    url = "https://script.google.com/macros/s/AKfycbxv_xvhImquXOtWAF7RbjKW6hMDyxL4LumA8G7LCXAcxFZvp8f-18tl6y0mvMGUtOG1/exec"
    try:
        requests.post(url, json=data, timeout=10)
    except:
        pass

# 4. القائمة الجانبية (كل المدخلات)
with st.sidebar:
    st.header("⚙️ Mix Ingredients (kg/m³)")
    c = st.number_input("Cement", 100, 600, 350)
    w = st.number_input("Water", 100, 250, 175)
    nca = st.number_input("NCA (Natural Coarse)", 500, 1500, 1050)
    nfa = st.number_input("NFA (Natural Fine)", 300, 1000, 750)
    rca = st.slider("RCA Replacement %", 0, 100, 0)
    rfa = st.slider("RFA Replacement %", 0, 100, 0)
    sf = st.number_input("Silica Fume", 0, 100, 0)
    fa = st.number_input("Fly Ash", 0, 200, 0)
    rha = st.number_input("Rice Husk Ash %", 0, 20, 0)
    fib = st.number_input("Nylon Fiber", 0.0, 5.0, 0.0)
    sp = st.number_input("Superplasticizer", 0.0, 15.0, 2.5)
    sz = st.selectbox("Max Agg Size (mm)", [10, 20, 40], index=1)
    sl = st.number_input("Slump Target (mm)", 0, 250, 100)
    den = st.number_input("Density", 2000, 2600, 2400)
    wc = w/c if c > 0 else 0
    inf = st.slider("Price Inflation Index", 0.5, 2.5, 1.0)
    run = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

# 5. عرض النتائج والتبويبات
t1, t2, t3, t4 = st.tabs(["🏗️ Strength Results", "🛡️ Durability", "🌍 LCA & Econ", "💡 AI Optimizer"])

if run:
    inp = np.array([[c, w, nca, nfa, rca, rfa, sf, fa, rha, fib, sp, wc, sz, sl, den]])
    p = model.predict(scaler.transform(inp))[0]
    
    full_data = {
        "c":c,"w":w,"nca":nca,"nfa":nfa,"rca":rca,"rfa":rfa,"sf":sf,"fa":fa,"rha":rha,"fib":fib,"sp":sp,"sz":sz,"sl":sl,"den":den,"wc":wc,
        "p0":p[0],"p1":p[1],"p2":p[2],"p3":p[3],"p4":p[4],"p5":p[5],"p6":p[6],"p7":p[7],
        "p8":p[8],"p9":p[9],"p10":p[10],"p11":p[11],"p12":p[12],"p13":p[13]*inf,"p14":p[14],"p16":p[16]
    }
    send_to_sheets(full_data)

    with t1:
        st.subheader("🎯 Mechanical Strength Profile")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CS 28d", f"{p[1]:.2f} MPa", "± 2.34")
        col2.metric("CS 7d", f"{p[0]:.2f} MPa")
        col3.metric("CS 90d", f"{p[2]:.2f} MPa")
        col4.metric("Split Tensile", f"{p[3]:.2f} MPa")
        
        st.divider()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(['7 Days', '28 Days', '90 Days'], [p[0], p[1], p[2]], marker='s', color='#004a99', linewidth=3)
        ax.set_ylabel("Strength (MPa)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with t2:
        st.subheader("🛡️ Durability & Physical Indices")
        d1, d2, d3 = st.columns(3)
        d1.metric("Elastic Modulus", f"{p[5]:.2f} GPa")
        d2.metric("Water Absorption", f"{p[6]:.2f} %")
        d3.metric("UPV Speed", f"{p[7]:.2f} km/s")
        st.divider()
        st.write(f"**Flexural Strength:** {p[4]:.2f} MPa | **Carbonation Depth:** {p[9]:.2f} mm")
        st.write(f"**Drying Shrinkage:** {p[8]:.0f} µε | **Chloride Permeability:** {p[10]:.0f} Coul.")

    with t3:
        st.subheader("🌍 Environmental & Economic LCA")
        e1, e2, e3 = st.columns(3)
        e1.metric("CO2 Footprint", f"{p[11]:.2f} kg/m³")
        e2.metric("Sustainability Index", f"{p[16]:.3f}")
        e3.metric("Adjusted Cost", f"${(p[13]*inf):.2f}")
        st.divider()
        st.write(f"**Energy Consumption:** {p[12]:.2f} MJ | **ACV Value:** {p[14]:.2f}")

with t4:
    st.header("💡 AI-Based Full Mix Optimizer")
    target_st = st.number_input("Enter Target Strength (28d)", 20, 80, 40)
    if st.button("Generate Top 10 Lab-Ready Mixes"):
        sims = []
        for _ in range(3000):
            c_r = np.random.randint(300, 500); w_r = np.random.randint(150, 185)
            nca_r = np.random.randint(900, 1100); nfa_r = np.random.randint(700, 850)
            rca_r = np.random.choice([0, 25, 50, 100]); sf_r = np.random.randint(0, 40)
            fa_r = np.random.randint(0, 100); rha_r = np.random.randint(0, 15)
            fib_r = np.random.uniform(0, 1.2); sp_r = np.random.uniform(2.0, 5.0)
            wc_r = w_r / c_r
            t_inp = np.array([[c_r, w_r, nca_r, nfa_r, rca_r, 0, sf_r, fa_r, rha_r, fib_r, sp_r, wc_r, 20, 100, 2400]])
            p_v = model.predict(scaler.transform(t_inp))[0]
            if abs(p_v[1] - target_st) < 1.5:
                sims.append({
                    'Cement': c_r, 'Water': w_r, 'NCA': nca_r, 'NFA': nfa_r, 'RCA%': rca_r, 
                    'SF': sf_r, 'FA': fa_r, 'RHA%': rha_r, 'Fiber': round(fib_r,2), 
                    'CO2': round(p_v[11],1), 'Strength': round(p_v[1],1)
                })
        if sims:
            st.success("Top 10 Complete Mix Designs Found:")
            st.dataframe(pd.DataFrame(sims).sort_values('CO2').head(10), use_container_width=True)
        else: st.warning("No matches found. Try another target.")
