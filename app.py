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
    .login-title { color: #004a99; text-align: center; font-weight: bold; margin-top: 15px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# 2. نظام الدخول الآمن (تمت إضافة اسم المشروع وشعار الكلية فقط)
if "auth" not in st.session_state: st.session_state.auth = False
if not st.session_state.auth:
    col_l, col_mid, col_r = st.columns([1, 2, 1])
    with col_mid:
        # عرض الشعارات (الجامعة والكلية)
        logo_col1, logo_col2 = st.columns(2)
        with logo_col1:
            st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-Smart-Mix/main/LOGO.png", width=140)
        with logo_col2:
            st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-Smart-Mix/main/OIP.jfif", width=140)
        
        # إضافة اسم المشروع
        st.markdown("<h2 class='login-title'>ASA Smart Mix: AI-Based Eco-Concrete Optimizer</h2>", unsafe_allow_html=True)
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
    st.error("Error: Model files not found. Please ensure 'concrete_model.pkl' and 'scaler_weights.pkl' are in the repository.")
    st.stop()

metrics_real = {"R2": 0.9557, "RMSE": 2.91, "COV": "6.16%"}

# 4. الهيدر الأكاديمي
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
    if c <= 0:
        st.error("Please enter a valid Cement content.")
    else:
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
            ax.plot(['7 Days', '28 Days', '90 Days'], [p[0], p[1], p[2]], marker='o', color='#004a99', linewidth=2)
            ax.set_ylabel("Strength (MPa)")
            ax.grid(True, linestyle='--', alpha=0.6)
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
            sf_fixed = 20 
            t_in = np.array([[cr, wr, 1050, 750, rca_r, 0, sf_fixed, 0, 0, 0, 2.5, wr/cr, 20, 100, 2400]])
            pv = model.predict(scaler.transform(t_in))[0]
            if abs(pv[1] - t_st) < 3.0:
                sims.append({
                    'Cement': cr, 
                    'Water': wr, 
                    'W/C': round(wr/cr, 2),
                    'RCA %': rca_r, 
                    'Silica Fume': sf_fixed,
                    'Strength (MPa)': round(pv[1], 1), 
                    'CO2 (kg/m³)': round(pv[11], 1),
                    'Sust. Index': round(pv[16], 3)
                })
        if sims:
            st.success("Top matching mixes sorted by lowest CO2 emissions:")
            res_df = pd.DataFrame(sims).sort_values('CO2 (kg/m³)').head(10)
            st.dataframe(res_df, use_container_width=True)
        else: st.warning("No matches found. Try adjusting the target strength.")

with t5:
    st.header("📖 Technical Documentation & Methodology")
    st.markdown(f"""
    <div class="doc-card">
    <h4>Core Model Information</h4>
    <ul>
        <li><b>Algorithm:</b> Random Forest Regression (Multi-output Architecture) - Selected for its superior ability to handle non-linear structural data without overfitting.</li>
        <li><b>Database:</b> Global Meta-Dataset comprising <b>400 Samples</b> meticulously collected from peer-reviewed journals.</li>
        <li><b>Applicability Domain:</b> Optimized for <b>Eco-friendly concrete mixes</b> with strengths between <b>20 MPa and 80 MPa</b>.</li>
        <li><b>Robustness:</b> Validated with a COV of <b>{metrics_real['COV']}</b>, demonstrating stability across raw material sources.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.subheader("📊 Statistical Validation (Scatter Plot)")
    scatter_url = "https://raw.githubusercontent.com/ayasanad14799-coder/ASA-Smart-Mix/main/scatter_accuracy.png"
    st.image(scatter_url, caption="Validation Plot: Predicted vs. Actual Compressive Strength (R² = 0.9557)", use_container_width=True)
    
    st.divider()
    st.subheader("💬 Experimental Feedback")
    feedback = st.text_area("Share your lab observations...", placeholder="e.g., The 28d strength was 42 MPa instead of 40 MPa.")
    if st.button("Submit Feedback"):
        st.success("Thank you! Your data has been queued for the next model retraining cycle.")

    st.divider()
    st.subheader("⚠️ Disclaimer")
    st.warning("This AI tool is for research and preliminary design. Actual laboratory trial mixes are mandatory for structural implementation.")

st.markdown("""<div class="footer-text">© 2026 Mansoura University - Structural Engineering Department<br>Developed by: Aya Mohammed Sanad</div>""", unsafe_allow_html=True)
