import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. إعدادات الصفحة
st.set_page_config(page_title="ASA Smart-Concrete AI", layout="wide", page_icon="🏗️")

# دالة التحقق من كلمة المرور
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
    # 2. تحميل الموديل والسكيلر (الملفات الجديدة الناتجة من التدريب المطور)
    @st.cache_resource
    def load_assets():
        try:
            model = joblib.load('concrete_model.pkl')
            scaler = joblib.load('scaler_weights.pkl')
            return model, scaler
        except Exception as e:
            st.error(f"⚠️ Error loading assets: {e}")
            return None, None

    model, scaler = load_assets()

    if model is not None:
        st.title("🏗️ ASA Smart Design & Sustainability Analysis")
        st.info("AI-Driven Prediction based on Multi-Target Random Forest")
        st.markdown("---")

        # 3. الشريط الجانبي - المدخلات الـ 15 حسب ترتيب ملف CSV
        st.sidebar.header("🛠️ Mix Design Parameters")
        
        c = st.sidebar.number_input("Cement (kg/m³)", min_value=0.0, value=400.0)
        w = st.sidebar.number_input("Water (kg/m³)", min_value=0.0, value=160.0)
        nca = st.sidebar.number_input("Natural Coarse Agg (kg/m³)", min_value=0.0, value=1150.0)
        nfa = st.sidebar.number_input("Natural Fine Agg (kg/m³)", min_value=0.0, value=750.0)
        rca = st.sidebar.slider("RCA Replacement %", 0, 100, 0)
        rfa = st.sidebar.slider("RFA Replacement %", 0, 100, 0)
        sf = st.sidebar.number_input("Silica Fume (kg/m³)", min_value=0.0, value=0.0)
        fa = st.sidebar.number_input("Fly Ash (kg/m³)", min_value=0.0, value=0.0)
        rha = st.sidebar.slider("RHA Replacement %", 0, 20, 0)
        nylon = st.sidebar.number_input("Nylon Fiber (kg/m³)", min_value=0.0, value=0.0, step=0.1)
        sp = st.sidebar.number_input("Superplasticizer (kg/m³)", min_value=0.0, value=4.0)
        
        w_c = w/c if c != 0 else 0
        msa = st.sidebar.selectbox("Max Agg Size (mm)", [10, 20], index=1)
        slump = st.sidebar.number_input("Target Slump (mm)", min_value=0.0, value=160.0)
        dens = st.sidebar.number_input("Fresh Density (kg/m³)", min_value=0.0, value=2450.0)

        # 4. واجهة عرض النتائج المقسمة لتبويبات
        tab1, tab2, tab3 = st.tabs(["💪 Mechanical Properties", "💧 Durability", "🌍 Sustainability & Cost"])

        if st.sidebar.button("🚀 Run AI Analysis", use_container_width=True):
            # تجميع المدخلات في مصفوفة
            raw_inputs = np.array([[c, w, nca, nfa, rca, rfa, sf, fa, rha, nylon, sp, w_c, msa, slump, dens]])
            
            # المعايرة باستخدام السكيلر الجديد
            scaled_inputs = scaler.transform(raw_inputs)
            
            # التنبؤ (سيخرج 17 مخرجاً حسب ترتيب ملف CSV)
            prediction = model.predict(scaled_inputs)[0]

            with tab1:
                st.subheader("📊 Mechanical Strengths")
                col1, col2, col3 = st.columns(3)
                # CS_28 هو المخرج رقم 2 (Index 1)
                col1.metric("CS (28 Days)", f"{prediction[1]:.2f} MPa")
                # CS_90 هو المخرج رقم 3 (Index 2)
                col2.metric("CS (90 Days)", f"{prediction[2]:.2f} MPa")
                # EM هو المخرج رقم 6 (Index 5)
                col3.metric("Elastic Modulus (EM)", f"{prediction[5]:.2f} GPa")

                st.markdown("---")
                col4, col5 = st.columns(2)
                # STS هو المخرج رقم 4 (Index 3)
                col4.metric("Split Tensile (STS)", f"{prediction[3]:.2f} MPa")
                # FS هو المخرج رقم 5 (Index 4)
                col5.metric("Flexural Strength (FS)", f"{prediction[4]:.2f} MPa")

                # رسم بياني للمقارنة الميكانيكية
                chart_data = pd.DataFrame({
                    'Metric': ['CS 28d', 'CS 90d', 'STS', 'FS'],
                    'Value (MPa)': [prediction[1], prediction[2], prediction[3], prediction[4]]
                })
                st.bar_chart(chart_data, x='Metric', y='Value (MPa)')

            with tab2:
                st.subheader("💧 Durability & Physical Tests")
                d1, d2, d3 = st.columns(3)
                # Water_Abs = Index 6
                d1.metric("Water Absorption", f"{prediction[6]:.2f} %")
                # UPV = Index 7
                d2.metric("UPV", f"{prediction[7]:.2f} km/s")
                # Cl_Perm = Index 10
                d3.metric("Chloride Permeability", f"{prediction[10]:.0f} Coulombs")

            with tab3:
                st.subheader("🌍 Eco-Impact & Economics")
                # إضافة معامل تعديل الأسعار
                st.markdown("### 💰 Smart Cost Adjuster")
                multiplier = st.number_input("Market Price Factor (Current Inflation)", value=1.0, step=0.1)
                
                # CO2 = Index 11, Energy = Index 12, Cost = Index 13
                s1, s2, s3 = st.columns(3)
                s1.metric("CO2 Footprint", f"{prediction[11]:.2f} kg/m³")
                s2.metric("Energy Demand", f"{prediction[12]:.0f} MJ/m³")
                s3.metric("Adjusted Cost", f"${prediction[13] * multiplier:.2f}", delta=f"Base: ${prediction[13]:.1f}")
                
                st.success("✅ Prediction based on calibrated historical data successful.")
        else:
            st.warning("👈 Please enter mix proportions and click 'Run AI Analysis'")
