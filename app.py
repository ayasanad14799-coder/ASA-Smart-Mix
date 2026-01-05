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

# 4. القائمة الجانبية (كل المدخلات الـ 15 دون استثناء)
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
    run_btn = st.button("🚀 Run Multi-Criteria Analysis", type="primary", use_container_width=True)

# 5. التبويبات لعرض النتائج الـ 17
t1, t2, t3, t4, t5 = st.tabs(["🏗️ Strength Results", "🛡️ Durability", "🌍 LCA & Economics", "💡 AI Optimizer", "📖 User Guide"])

if run_btn:
    # مصفوفة الـ 15 مدخل للموديل
    input_arr = np.array([[cement, water, nca, nfa, rca_p, rfa_p, sf, fa, rha, fiber, sp, wc, agg_size, slump_target, density]])
    preds = model.predict(scaler.transform(input_arr))[0]
    
    # ربط جوجل شيت الشامل (تسجيل الـ 32 متغير)
    # الرابط مأخوذ من صورتك المرفقة
    sheet_url = "https://script.google.com/macros/s/AKfycbxuQLsHy5spA0BBFasF88JGaKf5JTXrw3vXU67hIBl4xsmhFfHBW3zubuwVbh49EQuWdg/exec"
    full_data = {
        "c":cement,"w":water,"nca":nca,"nfa":nfa,"rca":rca_p,"rfa":rfa_p,"sf":sf,"fa":fa,"rha":rha,"fib":fiber,"sp":sp,"sz":agg_size,"sl":slump_target,"den":density,"wc":wc,
        "p0":preds[0],"p1":preds[1],"p2":preds[2],"p3":preds[3],"p4":preds[4],"p5":preds[5],"p6":preds[6],"p7":preds[7],
        "p8":preds[8],"p9":preds[9],"p10":preds[10],"p11":preds[11],"p12":preds[12],"p13":preds[13]*inflation,"p14":preds[14],"p16":preds[16]
    }
    try: requests.post(sheet_url, data=json.dumps(full_data))
    except: pass

    with t1:
        st.subheader("🎯 Mechanical Strength Profile")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CS 28d", f"{preds[1]:.2f} MPa", "± 2.34")
        col2.metric("CS 7d", f"{preds[0]:.2f} MPa")
        col3.metric("CS 90d", f"{preds[2]:.2f} MPa")
        col4.metric("Split Tensile (STS)", f"{preds[3]:.2f} MPa")
        
        st.markdown("### 📈 Strength Development Over Time")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(['7 Days', '28 Days', '90 Days'], [preds[0], preds[1], preds[2]], marker='s', markersize=8, color='#004a99', linewidth=3)
        ax.set_ylabel("Strength (MPa)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with t2:
        st.subheader("🛡️ Durability & Physical Indices")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Elastic Modulus", f"{preds[5]:.2f} GPa")
        d2.metric("Water Absorption", f"{preds[6]:.2f} %")
        d3.metric("UPV", f"{preds[7]:.2f} km/s")
        d4.metric("Flexural (FS)", f"{preds[4]:.2f} MPa")
        
        st.divider()
        k1, k2, k3 = st.columns(3)
        k1.metric("Drying Shrinkage", f"{preds[8]:.0f} µε")
        k2.metric("Carbonation Depth", f"{preds[9]:.2f} mm")
        k3.metric("Chloride Perm.", f"{preds[10]:.0f} Coul.")

    with t3:
        st.subheader("🌍 Environmental Impact & Economic LCA")
        e1, e2, e3 = st.columns(3)
        e1.metric("CO2 Footprint", f"{preds[11]:.2f} kg/m³")
        e2.metric("Energy Consumption", f"{preds[12]:.2f} MJ")
        e3.metric("Sustainability Index", f"{preds[16]:.3f}")
        
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Base Predicted Cost", f"${preds[13]:.2f}")
        m2.metric("Adjusted Market Cost", f"${(preds[13]*inflation):.2f}", delta=f"{((inflation-1)*100):.1f}%")
        m3.metric("ACV Value", f"{preds[14]:.2f}")

with t4:
    st.header("💡 AI-Based Eco-Mix Design")
    target_req = st.number_input("Input Required 28d Strength (MPa)", 20, 60, 35)
    if st.button("Suggest Optimal Eco-Mix"):
        with st.spinner("Simulating 1000 smart combinations..."):
            sim_results = []
            for _ in range(1000):
                c_rand = np.random.randint(300, 480)
                rca_rand = np.random.choice([0, 25, 50, 75, 100])
                rha_rand = np.random.randint(0, 15)
                wc_rand = 175 / c_rand
                test_v = np.array([[c_rand, 175, 1050, 750, rca_rand, 0, 10, 0, rha_rand, 0, 3.0, wc_rand, 20, 100, 2400]])
                p_v = model.predict(scaler.transform(test_v))[0]
                if abs(p_v[1] - target_req) < 1.2:
                    sim_results.append({'Cement': c_rand, 'RCA%': rca_rand, 'RHA%': rha_rand, 'CO2': p_v[11], 'Pred_CS': p_v[1]})
            if sim_results:
                best_eco = pd.DataFrame(sim_results).sort_values('CO2').iloc[0]
                st.success("✅ Optimal Green Mix Identified!")
                st.table(best_eco)
            else: st.warning("Try a different strength target.")

with t5:
    st.header("📖 Professional User Manual")
    st.markdown("""
    * **Data Logging:** Every 'Run' automatically records 32 parameters into the cloud research database.
    * **Units:** All materials in **kg/m³**, Strength in **MPa**, and CO2 in **kg/m³**.
    * **LCA Integration:** This tool evaluates sustainability based on Global Warming Potential (GWP) and Energy consumption.
    """)

st.caption("© 2026 | Sustainable Concrete Decision Support System | Mansoura University")
