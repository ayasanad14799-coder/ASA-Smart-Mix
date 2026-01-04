import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. إعدادات الخصوصية والأمان
def check_password():
    """Returns True if the user had the correct password."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    st.title("🔒 ASA Secure Access")
    placeholder = st.empty()
    
    with placeholder.form("login"):
        password = st.text_input("Please enter the access password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if password == "ASA2026":
                st.session_state["password_correct"] = True
                placeholder.empty() # حذف نموذج الدخول بعد النجاح
                return True
            else:
                st.error("❌ Wrong password. Please try again.")
                return False
    return False

# لا يبدأ البرنامج إلا إذا كانت كلمة السر صحيحة
if check_password():
    
    # إعداد واجهة البرنامج بعد تسجيل الدخول
    st.set_page_config(page_title="ASA Smart Concrete AI", layout="wide")

    # 2. تحميل الموديل والسكيلر
    @st.cache_resource
    def load_assets():
        try:
            with open('concrete_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('scaler_weights.pkl', 'rb') as f:
                scaler = pickle.load(f)
            return model, scaler
        except Exception as e:
            st.error(f"Error loading model files: {e}")
            return None, None

    model, scaler = load_assets()

    if model is not None:
        st.title("🏗️ ASA Smart Design & Sustainability Tool")
        st.success("Welcome, Dr. ASA! The AI engine is ready.")
        st.markdown("---")

        # 3. واجهة المدخلات
        st.sidebar.title("🛠️ Mix Design Parameters")
        
        cement = st.sidebar.number_input("Cement (kg/m3)", value=400)
        water = st.sidebar.number_input("Water (kg/m3)", value=165)
        nca = st.sidebar.number_input("NCA (kg/m3)", value=1150)
        nfa = st.sidebar.number_input("NFA (kg/m3)", value=750)
        rca_p = st.sidebar.slider("RCA Replacement %", 0, 100, 25)
        rfa_p = st.sidebar.slider("RFA Replacement %", 0, 100, 0)
        sf = st.sidebar.number_input("Silica Fume (kg/m3)", value=0)
        fa = st.sidebar.number_input("Fly Ash (kg/m3)", value=0)
        rha_p = st.sidebar.slider("RHA %", 0, 20, 0)
        nylon = st.sidebar.number_input("Nylon Fiber (kg/m3)", value=0.0)
        sp = st.sidebar.number_input("Superplasticizer (kg/m3)", value=4.0)
        
        w_c = water/cement if cement != 0 else 0
        agg_size = st.sidebar.selectbox("Max Aggregate Size (mm)", [10, 20])
        slump = st.sidebar.number_input("Slump (mm)", value=100)
        density = st.sidebar.number_input("Fresh Density (kg/m3)", value=2400)

        # تجميع المدخلات
        input_data = np.array([[cement, water, nca, nfa, rca_p, rfa_p, sf, fa, rha_p, nylon, sp, w_c, agg_size, slump, density]])

        # 4. النتائج
        tab1, tab2, tab3 = st.tabs(["💪 Mechanical Strength", "💧 Durability", "🌍 Sustainability"])

        if st.sidebar.button("Run AI Prediction"):
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)[0]

            with tab1:
                st.subheader("Predicted Strength")
                st.metric("Compressive Strength (28d)", f"{prediction:.2f} MPa")
                st.bar_chart(pd.DataFrame({'Property': ['CS'], 'Value': [prediction]}))

            with tab2:
                st.subheader("Durability Indicators")
                st.info(f"W/C Ratio: {w_c:.2f}")

            with tab3:
                st.subheader("Sustainability Impact")
                co2 = (cement * 0.9) + (nylon * 1.5)
                st.success(f"Estimated CO2: {co2:.2f} kg/m3")
