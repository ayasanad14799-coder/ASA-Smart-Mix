import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. إعدادات الأمان والواجهة
st.set_page_config(page_title="ASA Smart-Concrete AI", layout="wide", page_icon="🏗️")

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if st.session_state["password_correct"]:
        return True
    
    st.title("🔒 ASA Smart-Concrete Secure Portal")
    st.markdown("Scientific Research Tool for Advanced Concrete Optimization")
    
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
    # 2. تحميل الأصول البرمجية (الموديل والسكيلر)
    @st.cache_resource
    def load_assets():
        try:
            model = joblib.load('concrete_model.pkl')
            scaler = joblib.load('scaler_weights.pkl')
            return model, scaler
        except Exception as e:
            st.error(f"⚠️ Error loading model files: {e}")
            return None, None

    model, scaler = load_assets()

    if model is not None:
        st.title("🏗️ ASA Smart Design & Sustainability Tool")
        st.info("Direct AI Output - Zero External Calculations Added")
        st.markdown("---")

        # 3. الشريط الجانبي - الـ 15 مدخل (Inputs)
        st.sidebar.header("🛠️ Mix Design Parameters")
        
        c = st.sidebar.number_input("Cement (kg/m³)", min_value=0.0, value=400.0)
        w = st.sidebar.number_input("Water (kg/m³)", min_value=0.0, value=165.0)
        nca = st.sidebar.number_input("NCA (kg/m³)", min_value=0.0, value=1100.0)
        nfa = st.sidebar.number_input("NFA (kg/m³)", min_value=0.0, value=700.0)
        rca = st.sidebar.slider("RCA Replacement %", 0, 100, 25)
        rfa = st.sidebar.slider("RFA Replacement %", 0, 100, 0)
        sf = st.sidebar.number_input("Silica Fume (kg/m³)", min_value=0.0, value=0.0)
        fa = st.sidebar.number_input("Fly Ash (kg/m³)", min_value=0.0, value=0.0)
        rha = st.sidebar.slider("RHA Replacement %", 0, 20, 0)
        nylon = st.sidebar.number_input("Nylon Fiber (kg/m³)", min_value=0.0, value=0.0, step=0.1)
        sp = st.sidebar.number_input("Superplasticizer (kg/m³)", min_value=0.0, value=4.0)
        
        # W/C كمدخل رقم 12
        w_c = w/c if c != 0 else 0
        
        msa = st.sidebar.selectbox("Max Agg Size (mm)", [10, 20])
        slump = st.sidebar.number_input("Target Slump (mm)", min_value=0.0, value=100.0)
        dens = st.sidebar.number_input("Fresh Density (kg/m³)", min_value=0.0, value=2400.0)

        # 4. التنبؤ وعرض المخرجات الـ 17 (Outputs)
        tab1, tab2, tab3 = st.tabs(["💪 Mechanical Properties", "💧 Durability Indicators", "🌍 Eco-Impact & Cost"])

        if st.sidebar.button("🚀 Run Comprehensive AI Analysis", use_container_width=True):
            # تجميع البيانات للتحويل
            inputs = np.array([[c, w, nca, nfa, rca, rfa, sf, fa, rha, nylon, sp, w_c, msa, slump, dens]])
            scaled_inputs = scaler.transform(inputs)
            
            # التنبؤ المباشر من الموديل
            prediction = model.predict(scaled_inputs)[0]

            # عرض النتائج بناءً على ترتيب المخرجات في ملف الـ CSV
            with tab1:
                st.subheader("📊 Mechanical Strength Prediction")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("CS (28 Days)", f"{prediction[0]:.2f} MPa")
                col2.metric("CS (90 Days)", f"{prediction[1]:.2f} MPa")
                col3.metric("Split Tensile", f"{prediction[2]:.2f} MPa")
                col4.metric("Flexural Strength", f"{prediction[3]:.2f} MPa")

                # رسم بياني للمقارنة البصرية
                chart_data = pd.DataFrame({
                    'Metric': ['CS 28d', 'CS 90d', 'STS', 'FS'],
                    'Value (MPa)': [prediction[0], prediction[1], prediction[2], prediction[3]]
                })
                st.bar_chart(chart_data, x='Metric', y='Value (MPa)')

            with tab2:
                st.subheader("💧 Durability & Microstructure")
                d1, d2, d3 = st.columns(3)
                d1.metric("Water Absorption", f"{prediction[4]:.2f} %")
                d2.metric("Chloride Permeability", f"{prediction[5]:.1f} C")
                # عرض المخرجات الإضافية إذا وجدت في الترتيب
                try:
                    d3.metric("Porosity", f"{prediction[9]:.2f} %")
                except:
                    st.info("Standard Durability Metrics Displayed.")

            with tab3:
                st.subheader("🌍 Sustainability & Economic Impact")
                s1, s2, s3 = st.columns(3)
                s1.metric("CO2 Footprint", f"{prediction[6]:.2f} kg/m³")
                s2.metric("Energy Demand", f"{prediction[7]:.1f} MJ/m³")
                s3.metric("Production Cost", f"${prediction[8]:.2f}")
                
                st.success("✅ Prediction completed using Multi-target Random Forest Regressor.")
        else:
            st.warning("👈 Input your mix design and click the analysis button.")
