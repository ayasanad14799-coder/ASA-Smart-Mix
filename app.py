import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. إعدادات الصفحة والهوية الأكاديمية
st.set_page_config(page_title="Eco-Concrete AI Optimizer", layout="wide", page_icon="🏗️")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #004a99; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .header-container { display: flex; align-items: center; justify-content: center; background-color: #f8f9fa; padding: 20px; border-radius: 15px; border: 2px solid #004a99; margin-bottom: 25px; }
    .footer-text { text-align: center; color: #666; font-size: 0.85em; margin-top: 50px; padding: 20px; border-top: 1px solid #eee; }
    .doc-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-right: 5px solid #004a99; margin-bottom: 10px; }
    .login-title { color: #004a99; text-align: center; font-weight: bold; margin-top: 10px; margin-bottom: 20px; font-size: 24px; }
    </style>
    """, unsafe_allow_html=True)

# 2. نظام الدخول الآمن (واجهة الدخول بشعارين واسم المشروع)
if "auth" not in st.session_state: st.session_state.auth = False
if not st.session_state.auth:
    col_l, col_mid, col_r = st.columns([1, 2, 1])
    with col_mid:
        # عرض الشعارات جنباً إلى جنب (الجامعة والكلية)
        logo_col1, logo_col2 = st.columns(2)
        with logo_col1:
            st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-Smart-Mix/main/LOGO.png", width=140)
        with logo_col2:
            st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-Smart-Mix/main/OIP.jfif", width=140)
        
        st.markdown("<h2 class='login-title'>ASA Smart Mix: AI-Based Eco-Concrete Optimizer</h2>", unsafe_allow_html=True)
        
        with st.form("login"):
            st.markdown("<p style='text-align: center; color: #555;'>Enter the professional access code to unlock the engine</p>", unsafe_allow_html=True)
            pwd = st.text_input("Access Code", type="password")
            if st.form_submit_button("🚀 Unlock Engine"):
                if pwd == "ASA2026": 
                    st.session_state.auth = True
                    st.rerun()
                else: st.error("❌ Access Denied: Invalid Code")
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
    st.error("Error: Model files not found. Check repository files.")
    st.stop()

metrics_real = {"R2": 0.9557, "RMSE": 2.91, "COV": "6.16%"}

# 4. الهيدر الأكاديمي الداخلي (يظهر بعد الدخول)
st.markdown(f"""
    <div class="header-container">
        <img src="https://raw.githubusercontent.com/ayasanad14799-coder/ASA-Smart-Mix/main/LOGO.png" style="width: 120px; margin-right: 25px;">
        <div style="text-align: center;">
            <h2 style="color: #004a99; margin-bottom:5px;">Multi-criteria analysis of eco-efficient concrete from Technical, Environmental and Economic aspects</h2>
            <p style="font-size: 1.1em; margin-bottom:5px;"><b>Prepared by: Aya Mohammed Sanad Aboud</b></p>
            <p style="color: #666; margin-bottom:5px;">Supervision: <b>Prof. Ahmed Tahwia</b> & <b>Assoc. Prof. Asser El-Sheikh</b></p>
            <p style="color: #004a99;"><b>Mansoura University | Faculty of Engineering</b></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 5. القائمة الجانبية (Sidebar)
with st.sidebar:
    st.header("⚙️ Mix Ingredients")
    c = st.number_input("Cement (kg)", 100, 600, 350)
    w = st.number_input("Water (kg)", 100, 250, 175)
    nca = st.number_input("Natural Coarse Agg. (kg)", 500, 1500, 1050)
    nfa = st.number_input("Natural Fine Agg. (kg)", 300, 1000, 750)
    rca = st.slider("RCA Replacement %", 0, 100, 0)
    rfa = st.slider("RFA Replacement %", 0, 100, 0)
    sf = st.number_input("Silica Fume (kg)", 0, 100, 0)
    fa = st.number_input("Fly Ash (kg)", 0, 200, 0)
    rha = st.number_input("Rice Husk Ash %", 0, 20, 0)
    fib = st.number_input("Nylon Fiber (kg)", 0.0, 5.0, 0.0)
    sp = st.number_input("Superplasticizer (kg)", 0.0, 15.0, 2.5)
    sz = st.selectbox("Max Agg Size (mm)", [10, 20, 40], index=1)
    sl = st.number_input("Target Slump (mm)", 0, 250, 100)
    den = st.number_input("Density (kg/m³)", 2000, 2600, 2400)
    wc = w/c if c > 0 else 0
    inf = st.slider("Price Inflation Index", 0.5, 2.5, 1.0)
    run_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

# 6. التبويبات (Tabs)
t1, t2, t3, t4, t5 = st.tabs(["🏗️ Strength", "🛡️ Durability", "🌍 LCA & Econ", "💡 AI Optimizer", "📖 Technical Docs"])

if run_btn:
    if c <= 0: st.error("Please enter a valid Cement content.")
    else:
        # الحسابات والتنبؤ
        inp = np.array([[c, w, nca, nfa, rca, rfa, sf, fa, rha, fib, sp, wc, sz, sl, den]])
        p = model.predict(scaler.transform(inp))[0]
        
        with t1:
            st.subheader("🎯 Predictive Performance")
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy (R²)", f"{metrics_real['R2']*100:.2f}%")
            m2.metric("Mean Error", f"{metrics_real['RMSE']} MPa")
            m3.metric("COV", metrics_real['COV'])
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CS 28d", f"{p[1]:.2f} MPa"); c2.metric("CS 7d", f"{p[0]:.2f} MPa")
            c3.metric("CS 90d", f"{p[2]:.2f} MPa"); c4.metric("Split Tensile", f"{p[3]:.2f} MPa")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(['7d', '28d', '90d'], [p[0], p[1], p[2]], marker='o', color='#004a99')
            st.pyplot(fig)
        with t2:
            st.subheader("🛡️ Durability & Physical Indices")
            st.write(f"**Elastic Modulus:** {p[5]:.2f} GPa | **Water Absorption:** {p[6]:.2f} % | **UPV:** {p[7]:.2f} km/s")
        with t3:
            st.subheader("🌍 Environmental & Economic LCA")
            st.metric("CO2 Footprint", f"{p[11]:.2f} kg/m³"); st.metric("Sust. Index", f"{p[16]:.3f}"); st.metric("Cost", f"${(p[13]*inf):.2f}")

with t4:
    st.header("💡 AI-Based Mix Optimizer")
    t_st = st.number_input("Target Strength (28d) - MPa", 20, 80, 40)
    if st.button("Generate Top Green Mixes"):
        sims = []
        for _ in range(5000):
            cr, wr = np.random.randint(300, 500), np.random.randint(150, 190)
            rca_r = np.random.choice([0, 25, 50, 100])
            sf_fixed = 20
            t_in = np.array([[cr, wr, 1050, 750, rca_r, 0, sf_fixed, 0, 0, 0, 2.5, wr/cr, 20, 100, 2400]])
            pv = model.predict(scaler.transform(t_in))[0]
            if abs(pv[1] - t_st) < 3.0:
                sims.append({
                    'Cement': cr, 'Water': wr, 'W/C': round(wr/cr, 2), 'RCA %': rca_r, 
                    'Silica Fume': sf_fixed, 'Strength': round(pv[1], 1), 'CO2': round(pv[11], 1), 'Sust. Index': round(pv[16], 3)
                })
        if sims:
            st.success("Top matching mixes sorted by lowest CO2:")
            st.dataframe(pd.DataFrame(sims).sort_values('CO2').head(10), use_container_width=True)
        else: st.warning("No matches found.")

with t5:
    st.header("📖 Technical Documentation")
    st.markdown(f"<div class='doc-card'><b>Algorithm:</b> Random Forest | <b>Database:</b> 400 Samples | <b>Reliability (COV):</b> {metrics_real['COV']}</div>", unsafe_allow_html=True)
    st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-Smart-Mix/main/scatter_accuracy.png", caption="Model Validation Scatter Plot", use_container_width=True)
    st.divider()
    st.subheader("⚠️ Disclaimer")
    st.warning("This AI tool is for research purposes. Actual laboratory testing is required for structural use.")

st.markdown("""<div class="footer-text">© 2026 Mansoura University - Structural Engineering Dept.<br>Prepared by: Aya Mohammed Sanad</div>""", unsafe_allow_html=True)
