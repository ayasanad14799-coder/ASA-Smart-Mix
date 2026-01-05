import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json

# 1. إعدادات الصفحة
st.set_page_config(page_title="Eco-Concrete AI Optimizer", layout="wide")

# تنسيق CSS للشعار والـ Legend
st.markdown("""
    <style>
    .header-container { display: flex; align-items: center; justify-content: center; background-color: #f8f9fa; padding: 20px; border-radius: 15px; border: 2px solid #004a99; margin-bottom: 20px; }
    .logo-img { width: 100px; margin-right: 20px; }
    .legend-box { background-color: #e3f2fd; padding: 15px; border-radius: 10px; border: 1px solid #004a99; font-size: 0.9em; }
    </style>
    """, unsafe_allow_html=True)

# الهوية الأكاديمية والشعار
st.markdown(f"""
    <div class="header-container">
        <img src="https://upload.wikimedia.org/wikipedia/ar/thumb/0/01/Mansoura_University_logo.png/200px-Mansoura_University_logo.png" class="logo-img">
        <div style="text-align: center;">
            <h2 style="color: #004a99;">Multi-criteria analysis of eco-efficient concrete from Technical, Environmental and Economic aspects</h2>
            <p><b>Prepared by: Aya Mohammed Sanad Aboud</b> | Supervision: Prof. Ahmed Tahwia & Assoc. Prof. Asser El-Sheikh</p>
            <p>Mansoura University | Faculty of Engineering</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 2. نظام الدخول
if "auth" not in st.session_state: st.session_state.auth = False
if not st.session_state.auth:
    pwd = st.sidebar.text_input("Enter Access Code", type="password")
    if st.sidebar.button("Login"):
        if pwd == "ASA2026": st.session_state.auth = True; st.rerun()
        else: st.error("❌ Invalid Code")
    st.stop()

# 3. تحميل الموديلات
@st.cache_resource
def load_assets():
    model = joblib.load('concrete_model .pkl')
    scaler = joblib.load('scaler_weights .pkl')
    return model, scaler
model, scaler = load_assets()

# دالة إرسال البيانات (الرابط الأخير الصحيح)
def send_to_sheets(data):
    url = "https://script.google.com/macros/s/AKfycbxv_xvhImquXOtWAF7RbjKW6hMDyxL4LumA8G7LCXAcxFZvp8f-18tl6y0mvMGUtOG1/exec"
    try: requests.post(url, json=data, timeout=5)
    except: pass

# 4. القائمة الجانبية (15 مدخل)
with st.sidebar:
    st.header("⚙️ Mix Design Inputs")
    c = st.number_input("Cement", 100, 600, 350)
    w = st.number_input("Water", 100, 250, 175)
    nca = st.number_input("NCA (Natural Coarse)", 500, 1500, 1050)
    nfa = st.number_input("NFA (Natural Fine)", 300, 1000, 750)
    rca = st.slider("RCA %", 0, 100, 0)
    rfa = st.slider("RFA %", 0, 100, 0)
    sf = st.number_input("Silica Fume", 0, 100, 0)
    fa = st.number_input("Fly Ash", 0, 200, 0)
    rha = st.number_input("RHA %", 0, 20, 0)
    fib = st.number_input("Nylon Fiber", 0.0, 5.0, 0.0)
    sp = st.number_input("Superplasticizer", 0.0, 15.0, 2.5)
    sz = st.selectbox("Max Agg Size", [10, 20, 40], index=1)
    sl = st.number_input("Target Slump", 0, 250, 100)
    den = st.number_input("Density", 2000, 2600, 2400)
    wc = w/c if c > 0 else 0
    inf = st.slider("Price Index", 0.5, 2.5, 1.0)
    
    st.markdown("---")
    # الـ Legend (مرشد الاختصارات) في الجنب
    st.markdown("""<div class="legend-box"><b>Key Abbreviations:</b><br>
    <b>NCA/NFA:</b> Natural Aggregates<br>
    <b>RCA/RFA:</b> Recycled Aggregates<br>
    <b>SF/FA/RHA:</b> Mineral Admixtures<br>
    <b>CS:</b> Compressive Strength<br>
    <b>STS/FS:</b> Tensile & Flexural<br>
    <b>UPV:</b> Pulse Velocity (Quality)</div>""", unsafe_allow_html=True)

# 5. التبويبات الرئيسية
t1, t2, t3, t4, t5 = st.tabs(["🏗️ Strength", "🛡️ Durability", "🌍 LCA & Econ", "💡 AI Optimizer", "📖 User Guide"])

run = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

if run:
    inp = np.array([[c, w, nca, nfa, rca, rfa, sf, fa, rha, fib, sp, wc, sz, sl, den]])
    p = model.predict(scaler.transform(inp))[0]
    
    # إرسال 32 متغير للشيت
    full_data = {"c":c,"w":w,"nca":nca,"nfa":nfa,"rca":rca,"rfa":rfa,"sf":sf,"fa":fa,"rha":rha,"fib":fib,"sp":sp,"sz":sz,"sl":sl,"den":den,"wc":wc,
                 "p0":p[0],"p1":p[1],"p2":p[2],"p3":p[3],"p4":p[4],"p5":p[5],"p6":p[6],"p7":p[7],
                 "p8":p[8],"p9":p[9],"p10":p[10],"p11":p[11],"p12":p[12],"p13":p[13]*inf,"p14":p[14],"p16":p[16]}
    send_to_sheets(full_data)

    with t1:
        st.subheader("🎯 Mechanical Properties")
        c1, c2, c3 = st.columns(3)
        c1.metric("CS 28d", f"{p[1]:.2f} MPa")
        c2.metric("Split Tensile", f"{p[3]:.2f} MPa")
        c3.metric("Flexural", f"{p[4]:.2f} MPa")
        fig, ax = plt.subplots(figsize=(8, 3)); ax.plot(['7d', '28d', '90d'], [p[0], p[1], p[2]], marker='o'); st.pyplot(fig)

    with t2:
        st.subheader("🛡️ Durability & Absorption")
        d1, d2, d3 = st.columns(3)
        d1.metric("Absorption", f"{p[6]:.2f} %")
        d2.metric("UPV", f"{p[7]:.2f} km/s")
        d3.metric("Elastic Mod", f"{p[5]:.2f} GPa")
        st.write(f"**Carbonation:** {p[9]:.2f} mm | **Chloride:** {p[10]:.0f} Coul.")

    with t3:
        st.subheader("🌍 Environmental Impact")
        st.metric("CO2 Emissions", f"{p[11]:.2f} kg/m³")
        st.metric("Sustainability Index", f"{p[16]:.3f}")
        st.metric("Total Cost", f"${(p[13]*inf):.2f}")

with t4:
    st.header("💡 AI Optimizer")
    t_st = st.number_input("Target 28d Strength", 20, 80, 40)
    if st.button("Generate Top 10 Eco-Mixes"):
        sims = []
        for _ in range(5000):
            cr=np.random.randint(300,500); wr=np.random.randint(150,185); rcar=np.random.choice([0,50,100]); sfr=np.random.randint(0,40)
            t_in = np.array([[cr, wr, 1050, 750, rcar, 0, sfr, 0, 0, 0, 3.5, wr/cr, 20, 100, 2400]])
            pv = model.predict(scaler.transform(t_in))[0]
            if abs(pv[1]-t_st) < 2.0: sims.append({'Cement':cr, 'Water':wr, 'RCA%':rcar, 'SF':sfr, 'CO2':pv[11], 'Strength':pv[1]})
        if sims: st.dataframe(pd.DataFrame(sims).sort_values('CO2').head(10))
        else: st.warning("No matches.")

with t5:
    st.header("📖 User Guide & Legend")
    st.info("1. Enter your mix proportions in the sidebar.\n2. Click 'Run Analysis' to see predictions and log data to Google Sheets.\n3. Use the Optimizer to find the greenest mix for a specific strength.")
    st.write("### Terminology Legend:")
    st.write("- **p0 to p2**: Compressive strength at different ages.\n- **p11**: CO2 footprint (Lower is better).\n- **p16**: Sustainability Index (Higher is better).")
