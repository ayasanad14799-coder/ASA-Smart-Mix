import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# 1. إعدادات الصفحة والهوية الأكاديمية
st.set_page_config(page_title="Eco-Concrete AI Optimizer", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #004a99; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .header-container { display: flex; align-items: center; justify-content: center; background-color: #f8f9fa; padding: 20px; border-radius: 15px; border: 2px solid #004a99; margin-bottom: 25px; }
    .footer-text { text-align: center; color: #666; font-size: 0.85em; margin-top: 50px; padding: 20px; border-top: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

# 2. نظام الدخول الآمن
if "auth" not in st.session_state: st.session_state.auth = False
if not st.session_state.auth:
    col_l, col_mid, col_r = st.columns([1, 2, 1])
    with col_mid:
        # شعار الجامعة في صفحة الدخول
        st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-Smart-Mix/main/LOGO.png", width=150)
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
    st.error("Error: Model files not found. Check GitHub repository.")
    st.stop()

# المقاييس الحقيقية المستخرجة من التدريب
metrics_real = {"R2": 0.9557, "RMSE": 2.91, "COV": "6.16%"}

# 4. الهيدر الأكاديمي (يظهر بعد الدخول)
logo_url = "https://raw.githubusercontent.com/ayasanad14799-coder/ASA-Smart-Mix/main/LOGO.png"
st.markdown(f"""
    <div class="header-container">
        <img src="{logo_url}" style="width: 130px; margin-right: 25px;">
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
    # منع الحساب إذا كانت البيانات غير منطقية (Error Handling)
    if c <= 0:
        st.error("Please enter a valid Cement content.")
    else:
        # التحذيرات الهندسية
        if wc < 0.25 or wc > 0.65:
            st.sidebar.warning(f"⚠️ W/C Ratio Alert: {wc:.2f} is outside standard limits.")

        inp = np.array([[c, w, nca, nfa, rca, rfa, sf, fa, rha, fib, sp, wc, sz, sl, den]])
        p = model.predict(scaler.transform(inp))[0]
        
        with t1:
            st.subheader("🎯 Predictive Performance")
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy (R²)", f"{metrics_real['R2']*100:.2f}%")
            m2.metric("Mean Error", f"{metrics_real['RMSE']} MPa")
            m3.metric("COV (Stability)", metrics_real['COV'])
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CS 28d", f"{p[1]:.2f} MPa")
            c2.metric("CS 7d", f"{p[0]:.2f} MPa")
            c3.metric("CS 90d", f"{p[2]:.2f} MPa")
            c4.metric("Split Tensile", f"{p[3]:.2f} MPa")
            
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(['7 Days', '28 Days', '90 Days'], [p[0], p[1], p[2]], marker='o', color='#004a99')
            ax.set_ylabel("Strength (MPa)")
            st.pyplot(fig)

        with t2:
            st.subheader("🛡️ Durability & Physical Indices")
            d1, d2, d3 = st.columns(3)
            d1.metric("Elastic Modulus", f"{p[5]:.2f} GPa")
            d2.metric("Water Absorption", f"{p[6]:.2f} %")
            d3.metric("UPV Speed", f"{p[7]:.2f} km/s")
            st.info(f"**Flexural Strength:** {p[4]:.2f} MPa | **Carbonation Depth:** {p[9]:.2f} mm")

        with t3:
            st.subheader("🌍 Environmental & Economic LCA")
            e1, e2, e3 = st.columns(3)
            e1.metric("CO2 Footprint", f"{p[11]:.2f} kg/m³")
            e2.metric("Sustainability Index", f"{p[16]:.3f}")
            e3.metric("Adjusted Cost", f"${(p[13]*inf):.2f}")

with t4:
    st.header("💡 AI-Based Mix Optimizer")
    st.write("Find the most eco-friendly mix for your target strength:")
    t_st = st.number_input("Enter Target Strength (28d) - MPa", 20, 80, 40)
    if st.button("Generate Top Green Mixes"):
        sims = []
        for _ in range(5000):
            cr, wr = np.random.randint(300, 500), np.random.randint(150, 190)
            rca_r = np.random.choice([0, 25, 50, 100])
            t_in = np.array([[cr, wr, 1050, 750, rca_r, 0, 20, 0, 0, 0, 2.5, wr/cr, 20, 100, 2400]])
            pv = model.predict(scaler.transform(t_in))[0]
            if abs(pv[1] - t_st) < 3.0:
                sims.append({'Cement': cr, 'Water': wr, 'RCA%': rca_r, 'CO2': round(pv[11], 1), 'Strength': round(pv[1], 1)})
        if sims:
            st.success("Top matching mixes sorted by lowest CO2:")
            st.dataframe(pd.DataFrame(sims).sort_values('CO2').head(10), use_container_width=True)
        else: st.warning("No matches found. Try adjusting the target strength.")

with t5:
    st.header("📖 Technical Documentation")
    st.markdown(f"""
    * **Algorithm:** Random Forest Regression (Multi-output Architecture).
    * **Database:** Global Meta-Dataset (400 Samples from various international peer-reviewed journals).
    * **Applicability Domain:** Optimized for Eco-friendly concrete mixes with strengths between **20 MPa and 80 MPa**.
    * **Robustness:** Validated with a COV of **{metrics_real['COV']}**, showing high stability across different raw material sources.
    """)
    st.divider()
    
    # عرض مخطط التشتت الحقيقي الخاص بكِ
    st.subheader("📊 Model Accuracy (Scatter Plot)")
    scatter_url = "https://raw.githubusercontent.com/ayasanad14799-coder/ASA-Smart-Mix/main/scatter_accuracy.png"
    st.image(scatter_url, caption="Predicted vs. Actual Compressive Strength (Lab Validation Data)", use_container_width=True)
    
    st.divider()
    st.subheader("⚠️ Disclaimer")
    st.warning("This AI tool is for research and preliminary design purposes. Laboratory trial mixes are mandatory for structural applications and compliance with local building codes.")

st.markdown("""<div class="footer-text">© 2024 Mansoura University - Structural Engineering Department<br>Prepared by: Aya Mohammed Sanad</div>""", unsafe_allow_html=True)
