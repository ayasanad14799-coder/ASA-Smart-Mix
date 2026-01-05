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
    .header-box { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 2px solid #004a99; text-align: center; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown(f"""
    <div class="header-box">
        <h2 style="color: #004a99; margin-bottom:10px;">Multi-criteria analysis of eco-efficient concrete from Technical, Environmental and Economic aspects</h2>
        <p style="font-size: 1.2em;"><b>Prepared by: Aya Mohammed Sanad Aboud</b></p>
        <p style="color: #666;">Supervision: Prof. Ahmed Tahwia & Assoc. Prof. Asser El-Sheikh | Mansoura University</p>
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
    model = joblib.load('concrete_model .pkl')
    scaler = joblib.load('scaler_weights .pkl')
    return model, scaler
model, scaler = load_assets()

# --- دالة إرسال البيانات (تم وضع رابطك الصحيح هنا) ---
def send_to_sheets(data):
    url = "https://script.google.com/macros/s/AKfycbxuQLsHy5spA0BBFasF88JGaKf5JTXrw3vXU67hIBl4xsmhFfHBW3zubuwVbh49EQuWdg/exec"
    try:
        requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})
    except:
        pass

# 4. القائمة الجانبية (كل المدخلات الـ 15)
with st.sidebar:
    st.header("⚙️ Full Mix Ingredients (kg/m³)")
    cement = st.number_input("Cement", 100, 600, 350)
    water = st.number_input("Water", 100, 250, 175)
    nca = st.number_input("Natural Coarse Agg (NCA)", 500, 1500, 1050)
    nfa = st.number_input("Natural Fine Agg (NFA)", 300, 1000, 750)
    rca_p = st.slider("RCA Replacement (%)", 0, 100, 0)
    rfa_p = st.slider("RFA Replacement (%)", 0, 100, 0)
    sf = st.number_input("Silica Fume", 0, 100, 0)
    fa = st.number_input("Fly Ash", 0, 200, 0)
    rha = st.number_input("Rice Husk Ash (RHA %)", 0, 20, 0)
    fiber = st.number_input("Nylon Fiber", 0.0, 5.0, 0.0)
    sp = st.number_input("Superplasticizer (SP)", 0.0, 15.0, 2.5)
    agg_size = st.selectbox("Max Agg Size (mm)", [10, 20, 40], index=1)
    slump_target = st.number_input("Target Slump (mm)", 0, 250, 100)
    density = st.number_input("Target Density", 2000, 2600, 2400)
    
    wc = water/cement if cement > 0 else 0
    st.divider()
    inflation = st.slider("Market Price Index (Inflation)", 0.5, 2.5, 1.0)
    run_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

# 5. التبويبات لعرض النتائج
t1, t2, t3, t4, t5 = st.tabs(["🏗️ Strength", "🛡️ Durability", "🌍 LCA & Economics", "💡 AI Optimizer", "📖 Guide"])

if run_btn:
    input_arr = np.array([[cement, water, nca, nfa, rca_p, rfa_p, sf, fa, rha, fiber, sp, wc, agg_size, slump_target, density]])
    p = model.predict(scaler.transform(input_arr))[0]
    
    # إرسال الـ 32 متغير للشيت
    full_data = {
        "c":cement,"w":water,"nca":nca,"nfa":nfa,"rca":rca_p,"rfa":rfa_p,"sf":sf,"fa":fa,"rha":rha,"fib":fiber,"sp":sp,"sz":agg_size,"sl":slump_target,"den":density,"wc":wc,
        "p0":p[0],"p1":p[1],"p2":p[2],"p3":p[3],"p4":p[4],"p5":p[5],"p6":p[6],"p7":p[7],
        "p8":p[8],"p9":p[9],"p10":p[10],"p11":p[11],"p12":p[12],"p13":p[13]*inflation,"p14":p[14],"p16":p[16]
    }
    send_to_sheets(full_data)

    with t1:
        st.subheader("Mechanical Strength Profile")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CS 28d", f"{p[1]:.2f} MPa", "± 2.34")
        c2.metric("CS 7d", f"{p[0]:.2f} MPa")
        c3.metric("CS 90d", f"{p[2]:.2f} MPa")
        c4.metric("Split Tensile", f"{p[3]:.2f} MPa")
        
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(['7 Days', '28 Days', '90 Days'], [p[0], p[1], p[2]], marker='s', color='#004a99', linewidth=3)
        ax.set_title("Strength Development")
        st.pyplot(fig)

    with t2:
        st.subheader("Durability & Physical Properties")
        d1, d2, d3 = st.columns(3)
        d1.metric("Elastic Modulus", f"{p[5]:.2f} GPa")
        d2.metric("Absorption", f"{p[6]:.2f} %")
        d3.metric("UPV", f"{p[7]:.2f} km/s")
        st.write(f"**Chloride Permeability:** {p[10]:.0f} Coulombs | **Carbonation:** {p[9]:.2f} mm")

    with t3:
        st.subheader("LCA & Economic Impact")
        e1, e2, e3 = st.columns(3)
        e1.metric("CO2 Footprint", f"{p[11]:.2f} kg/m³")
        e2.metric("Sustainability Index", f"{p[16]:.3f}")
        e3.metric("Cost ($)", f"{(p[13]*inflation):.2f}")

with t4:
    st.header("💡 AI-Based Eco-Mix Design")
    target_req = st.number_input("Target Strength (MPa)", 20, 80, 40)
    if st.button("Generate Top 10 Eco-Mixes"):
        sim_results = []
        for _ in range(3000):
            c_r = np.random.randint(250, 550); rca_r = np.random.choice([0, 25, 50, 100]); sf_r = np.random.randint(0, 50)
            fa_r = np.random.randint(0, 100); rha_r = np.random.randint(0, 15)
            test_v = np.array([[c_r, 175, 1050, 750, rca_r, 0, sf_r, fa_r, rha_r, 0, 3.5, 175/c_r, 20, 100, 2400]])
            p_v = model.predict(scaler.transform(test_v))[0]
            if abs(p_v[1] - target_req) < 2.5:
                sim_results.append({'Cement': c_r, 'RCA%': rca_r, 'SF': sf_r, 'FA': fa_r, 'RHA%': rha_r, 'CO2': round(p_v[11],2), 'Strength': round(p_v[1],2)})
        if sim_results:
            st.dataframe(pd.DataFrame(sim_results).sort_values('CO2').head(10), use_container_width=True)
        else: st.warning("No matches found. Try another target.")

with t5:
    st.header("📖 User Manual")
    st.info("All experimental results are logged to the cloud database. Units: kg/m³, MPa, kg CO2-eq.")
