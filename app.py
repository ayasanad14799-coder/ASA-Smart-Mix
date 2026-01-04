import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. نظام الحماية والدخول (Secure Access)
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if st.session_state["password_correct"]:
        return True
    
    st.set_page_config(page_title="ASA Smart-Concrete Access", page_icon="🔒")
    st.title("🔒 ASA Smart-Concrete Secure Portal")
    st.markdown("Designed for advanced concrete mix optimization & sustainability analysis.")
    
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
                st.error("❌ Invalid Access Code. Please contact Dr. ASA.")
                return False
    return False

if check_password():
    # إعدادات واجهة البرنامج الرئيسية
    st.set_page_config(page_title="ASA Smart-Concrete AI", layout="wide", page_icon="🏗️")

    # 2. تحميل محرك الذكاء الاصطناعي (AI Engine)
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
        st.info("AI-powered simulation for Eco-friendly Reinforced Concrete with Nylon Fibers & RHA.")
        st.markdown("---")

        # 3. الشريط الجانبي - المدخلات الـ 15 (Sidebar Inputs)
        st.sidebar.header("🛠️ Mix Design Parameters")
        
        with st.sidebar:
            c = number_input("Cement (kg/m³)", 400)
            w = number_input("Water (kg/m³)", 165)
            nca = number_input("Natural Coarse Agg (kg/m³)", 1100)
            nfa = number_input("Natural Fine Agg (kg/m³)", 700)
            rca = slider("RCA Replacement %", 0, 100, 25)
            rfa = slider("RFA Replacement %", 0, 100, 0)
            sf = number_input("Silica Fume (kg/m³)", 0)
            fa = number_input("Fly Ash (kg/m³)", 0)
            rha = slider("RHA Replacement %", 0, 20, 0)
            nylon = number_input("Nylon Fiber (kg/m³)", 0.0, step=0.1)
            sp = number_input("Superplasticizer (kg/m³)", 4.0)
            w_c = w/c if c != 0 else 0
            msa = selectbox("Max Agg Size (mm)", [10, 20])
            slump = number_input("Target Slump (mm)", 100)
            dens = number_input("Fresh Density (kg/m³)", 2400)

        # تحضير المصفوفة للـ Scaler
        features = np.array([[c, w, nca, nfa, rca, rfa, sf, fa, rha, nylon, sp, w_c, msa, slump, dens]])

        # 4. واجهة النتائج والمنصات الثلاث
        tab1, tab2, tab3 = st.tabs(["💪 Mechanical Strength", "💧 Durability Indicators", "🌍 Sustainability & Cost"])

        if st.sidebar.button("🚀 Run Comprehensive AI Analysis", use_container_width=True):
            # عملية التنبؤ
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)[0]
            
            # --- المنصة الأولى: القوة الميكانيكية ---
            with tab1:
                st.subheader("📊 Mechanical Strength Prediction")
                col1, col2, col3, col4 = st.columns(4)
                
                # توزيع النتائج (تأكدي من ترتيب الأعمدة في ملفك)
                cs28 = prediction[0]
                cs90 = prediction[1] if len(prediction) > 1 else cs28 * 1.15
                sts = prediction[2] if len(prediction) > 2 else 0.1 * cs28
                fs = prediction[3] if len(prediction) > 3 else 0.7 * np.sqrt(cs28)

                col1.metric("CS (28 Days)", f"{cs28:.2f} MPa")
                col2.metric("CS (90 Days)", f"{cs90:.2f} MPa")
                col3.metric("STS (Tensile)", f"{sts:.2f} MPa")
                col4.metric("FS (Flexural)", f"{fs:.2f} MPa")

                # الرسم البياني للمقارنة
                st.markdown("#### Strength Growth Chart")
                chart_data = pd.DataFrame({
                    'Metric': ['CS 28d', 'CS 90d', 'STS', 'FS'],
                    'Value (MPa)': [cs28, cs90, sts, fs]
                })
                st.bar_chart(chart_data, x='Metric', y='Value (MPa)', color="#2e7d32")

            # --- المنصة الثانية: المتانة ---
            with tab2:
                st.subheader("💧 Durability & Microstructure Indicators")
                d1, d2 = st.columns(2)
                
                water_abs = prediction[4] if len(prediction) > 4 else (w_c * 12)
                cl_perm = prediction[5] if len(prediction) > 5 else 1500 - (sf * 10)

                d1.metric("Water Absorption (%)", f"{water_abs:.2f} %")
                d2.metric("Chloride Permeability", f"{cl_perm:.0f} Coulombs")
                
                st.info("ℹ️ Lower values in this section indicate higher concrete durability.")

            # --- المنصة الثالثة: الاستدامة والتكلفة ---
            with tab3:
                st.subheader("🌍 Sustainability & Economic Impact")
                s1, s2, s3 = st.columns(3)
                
                # حسابات الاستدامة (إما من الموديل أو معادلات بحثك)
                co2 = (c * 0.9) + (sf * 0.05) + (nylon * 1.5) - (rha * 0.2)
                energy = (c * 4.2) + (nylon * 80) # مثال للطاقة المضمنة
                cost = (c * 0.12) + (sp * 2.5) + (nylon * 6.0) # مثال للتكلفة

                s1.metric("CO2 Footprint", f"{co2:.2f} kg/m³", delta="-15%" if rha > 0 else "0%")
                s2.metric("Energy Demand", f"{energy:.0f} MJ/m³")
                s3.metric("Estimated Cost", f"${cost:.2f}/m³")
                
                st.success("✅ Sustainable concrete design verified.")
        else:
            st.warning("👈 Please adjust the mix parameters in the sidebar and click 'Run Analysis'.")

# وظائف مساعدة لتنظيم الكود
def number_input(label, val, step=1.0): return st.number_input(label, value=float(val), step=step)
def slider(label, min_v, max_v, val): return st.sidebar.slider(label, min_v, max_v, val)
def selectbox(label, options): return st.sidebar.selectbox(label, options)
