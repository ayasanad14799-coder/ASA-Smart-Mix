import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. إعدادات الصفحة والأمان
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if st.session_state["password_correct"]:
        return True
    st.title("🔒 ASA Smart-Concrete Secure Portal")
    placeholder = st.empty()
    with placeholder.form("login"):
        password = st.text_input("Access Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if password == "ASA2026":
                st.session_state["password_correct"] = True
                placeholder.empty()
                st.rerun()
                return True
            else:
                st.error("❌ Invalid Access Code")
                return False
    return False

if check_password():
    st.set_page_config(page_title="ASA Smart-Concrete AI", layout="wide")

    # 2. تحميل الموديل والسكيلر (الملفات الجديدة)
    @st.cache_resource
    def load_assets():
        try:
            model = joblib.load('concrete_model.pkl')
            scaler = joblib.load('scaler_weights.pkl')
            return model, scaler
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
            return None, None

    model, scaler = load_assets()

    if model is not None:
        st.title("🏗️ ASA Smart Design & Sustainability Analysis")
        st.markdown("---")

        # 3. المدخلات الجانبية (15 برامتر)
        st.sidebar.header("🛠️ Mix Parameters")
        c = st.sidebar.number_input("Cement (kg/m³)", 400.0)
        w = st.sidebar.number_input("Water (kg/m³)", 165.0)
        nca = st.sidebar.number_input("NCA (kg/m³)", 1100.0)
        nfa = st.sidebar.number_input("NFA (kg/m³)", 700.0)
        rca = st.sidebar.slider("RCA Replacement %", 0, 100, 25)
        rfa = st.sidebar.slider("RFA Replacement %", 0, 100, 0)
        sf = st.sidebar.number_input("Silica Fume (kg/m³)", 0.0)
        fa = st.sidebar.number_input("Fly Ash (kg/m³)", 0.0)
        rha = st.sidebar.slider("RHA Replacement %", 0, 20, 0)
        nylon = st.sidebar.number_input("Nylon Fiber (kg/m³)", 0.0, step=0.1)
        sp = st.sidebar.number_input("Superplasticizer (kg/m³)", 4.0)
        
        w_c = w/c if c != 0 else 0
        msa = st.sidebar.selectbox("Max Agg Size (mm)", [10, 20])
        slump = st.sidebar.number_input("Target Slump (mm)", 100.0)
        dens = st.sidebar.number_input("Fresh Density (kg/m³)", 2400.0)

        # 4. زر التنبؤ والنتائج
        tab1, tab2, tab3 = st.tabs(["💪 Strength Prediction", "💧 Durability", "🌍 Sustainability & Cost"])

        if st.sidebar.button("🚀 Run Comprehensive AI Analysis", use_container_width=True):
            inputs = np.array([[c, w, nca, nfa, rca, rfa, sf, fa, rha, nylon, sp, w_c, msa, slump, dens]])
            scaled_inputs = scaler.transform(inputs)
            # الموديل سيعطي 17 مخرجاً بناءً على تدريب الكولاب
            prediction = model.predict(scaled_inputs)[0]

            # عرض النتائج الحقيقية من الموديل (الترتيب حسب ملف CSV)
            with tab1:
                st.subheader("📊 Mechanical Strength (Model Outputs)")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("CS (28d)", f"{prediction[0]:.2f} MPa")
                m2.metric("CS (90d)", f"{prediction[1]:.2f} MPa")
                m3.metric("STS", f"{prediction[2]:.2f} MPa")
                m4.metric("FS", f"{prediction[3]:.2f} MPa")

                # رسم بياني تفاعلي للمقاومة
                chart_data = pd.DataFrame({
                    'Metric': ['CS 28d', 'CS 90d', 'STS', 'FS'],
                    'Value (MPa)': [prediction[0], prediction[1], prediction[2], prediction[3]]
                })
                st.bar_chart(chart_data, x='Metric', y='Value (MPa)')

            with tab2:
                st.subheader("💧 Durability Performance")
                d1, d2 = st.columns(2)
                # استخدام المخرجات التالية من الموديل
                d1.metric("Water Absorption", f"{prediction[4]:.2f} %")
                d2.metric("Chloride Permeability", f"{prediction[5]:.1f} Coulombs")

            with tab3:
                st.subheader("🌍 Sustainability & Impact")
                s1, s2, s3 = st.columns(3)
                # استخدام آخر مخرجات الموديل (CO2, Energy, Cost)
                s1.metric("CO2 Footprint", f"{prediction[6]:.2f} kg/m³")
                s2.metric("Energy Demand", f"{prediction[7]:.1f} MJ/m³")
                s3.metric("Estimated Cost", f"${prediction[8]:.2f}")
        else:
            st.info("👈 Please adjust parameters and click 'Run Analysis'.")
