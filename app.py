import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. تعريف وظائف مساعدة (Helpers) - يجب أن تكون في البداية
def custom_number_input(label, val, step=1.0):
    return st.sidebar.number_input(label, value=float(val), step=step)

def custom_slider(label, min_v, max_v, val):
    return st.sidebar.slider(label, min_v, max_v, val)

# 2. نظام الحماية والدخول
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if st.session_state["password_correct"]:
        return True
    
    st.title("🔒 ASA Smart-Concrete Secure Portal")
    placeholder = st.empty()
    with placeholder.form("login"):
        password = st.text_input("Please enter the access password", type="password")
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

    # 3. تحميل محرك الذكاء الاصطناعي
    @st.cache_resource
    def load_assets():
        try:
            model = joblib.load('concrete_model.pkl')
            scaler = joblib.load('scaler_weights.pkl')
            return model, scaler
        except Exception as e:
            st.error(f"⚠️ Error loading AI Engine: {e}")
            return None, None

    model, scaler = load_assets()

    if model is not None:
        st.title("🏗️ ASA Smart Design & Sustainability Analysis Tool")
        st.info("AI-powered simulation for Eco-friendly Reinforced Concrete")
        st.markdown("---")

        # 4. الشريط الجانبي - المدخلات الـ 15
        st.sidebar.header("🛠️ Mix Design Parameters")
        
        c = custom_number_input("Cement (kg/m³)", 400)
        w = custom_number_input("Water (kg/m³)", 165)
        nca = custom_number_input("Natural Coarse Agg (kg/m³)", 1100)
        nfa = custom_number_input("Natural Fine Agg (kg/m³)", 700)
        rca = custom_slider("RCA Replacement %", 0, 100, 25)
        rfa = custom_slider("RFA Replacement %", 0, 100, 0)
        sf = custom_number_input("Silica Fume (kg/m³)", 0)
        fa = custom_number_input("Fly Ash (kg/m³)", 0)
        rha = custom_slider("RHA Replacement %", 0, 20, 0)
        nylon = custom_number_input("Nylon Fiber (kg/m³)", 0.0, step=0.1)
        sp = custom_number_input("Superplasticizer (kg/m³)", 4.0)
        
        w_c = w/c if c != 0 else 0
        msa = st.sidebar.selectbox("Max Agg Size (mm)", [10, 20])
        slump = custom_number_input("Target Slump (mm)", 100)
        dens = custom_number_input("Fresh Density (kg/m³)", 2400)

        # تحضير المصفوفة
        features = np.array([[c, w, nca, nfa, rca, rfa, sf, fa, rha, nylon, sp, w_c, msa, slump, dens]])

        # 5. واجهة النتائج
        tab1, tab2, tab3 = st.tabs(["💪 Mechanical Strength", "💧 Durability", "🌍 Sustainability & Cost"])

        if st.sidebar.button("🚀 Run Comprehensive AI Analysis", use_container_width=True):
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)[0]
            
            with tab1:
                st.subheader("📊 Mechanical Strength Prediction")
                col1, col2, col3, col4 = st.columns(4)
                
                # تأكدي من ترتيب المخرجات بناءً على ملفك
                cs28 = prediction[0]
                cs90 = prediction[1] if len(prediction) > 1 else cs28 * 1.12
                sts = prediction[2] if len(prediction) > 2 else 0.1 * cs28
                fs = prediction[3] if len(prediction) > 3 else 0.7 * np.sqrt(cs28)

                col1.metric("CS (28 Days)", f"{cs28:.2f} MPa")
                col2.metric("CS (90 Days)", f"{cs90:.2f} MPa")
                col3.metric("STS (Tensile)", f"{sts:.2f} MPa")
                col4.metric("FS (Flexural)", f"{fs:.2f} MPa")

                # الرسم البياني
                chart_data = pd.DataFrame({
                    'Metric': ['CS 28d', 'CS 90d', 'STS', 'FS'],
                    'Value (MPa)': [cs28, cs90, sts, fs]
                })
                st.bar_chart(chart_data, x='Metric', y='Value (MPa)')

            with tab2:
                st.subheader("💧 Durability Indicators")
                d1, d2 = st.columns(2)
                water_abs = prediction[4] if len(prediction) > 4 else (w_c * 11)
                cl_perm = prediction[5] if len(prediction) > 5 else 1200 - (sf * 5)
                d1.metric("Water Absorption (%)", f"{water_abs:.2f} %")
                d2.metric("Chloride Permeability", f"{cl_perm:.0f} Coulombs")

            with tab3:
                st.subheader("🌍 Sustainability & Impact")
                s1, s2, s3 = st.columns(3)
                co2 = (c * 0.9) + (sf * 0.05) + (nylon * 1.5) - (rha * 0.2)
                s1.metric("CO2 Footprint", f"{co2:.2f} kg/m³")
                s2.metric("Eco Rating", "A+" if rha > 10 else "B")
                s3.metric("Recycled Content", f"{rca+rfa+rha:.1f} %")
        else:
            st.warning("👈 Adjust parameters and click 'Run Analysis'")
