import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. إعدادات الخصوصية (كلمة السر)
def check_password():
    def password_guessed():
        if st.session_state["password"] == "ASA2026":  # كلمة السر الخاصة بكِ
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 ASA Secure Access")
        st.text_input("Please enter the access password", type="password", on_change=password_guessed, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔒 ASA Secure Access")
        st.text_input("Please enter the access password", type="password", on_change=password_guessed, key="password")
        st.error("❌ Wrong password. Please try again.")
        return False
    else:
        return True

# لا يبدأ البرنامج إلا إذا كانت كلمة السر صحيحة
if check_password():
    
    # إعداد واجهة البرنامج بعد تسجيل الدخول
    st.set_page_config(page_title="ASA Smart Concrete AI", layout="wide")

    # 2. تحميل الموديل والسكيلر بطريقة آمنة (Binary Mode)
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
        st.markdown("---")

        # 3. واجهة المستخدم في الشريط الجانبي (Inputs)
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
        
        # حسابات تلقائية تعتمد على المدخلات
        w_c = water/cement if cement != 0 else 0
        agg_size = st.sidebar.selectbox("Max Aggregate Size (mm)", [10, 20])
        slump = st.sidebar.number_input("Slump (mm)", value=100)
        density = st.sidebar.number_input("Fresh Density (kg/m3)", value=2400)

        # تجميع المدخلات في مصفوفة (يجب أن تكون 15 قيمة بنفس ترتيب الـ Scaler)
        input_data = np.array([[cement, water, nca, nfa, rca_p, rfa_p, sf, fa, rha_p, nylon, sp, w_c, agg_size, slump, density]])

        # 4. تقسيم النتائج إلى منصات (Tabs)
        tab1, tab2, tab3 = st.tabs(["💪 Mechanical Strength", "💧 Durability", "🌍 Sustainability & Cost"])

        if st.sidebar.button("Run AI Prediction"):
            # تحويل البيانات باستخدام السكيلر ثم التنبؤ
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)[0]

            with tab1:
                st.subheader("Results: Mechanical Properties")
                col1, col2, col3 = st.columns(3)
                col1.metric("Compressive Strength (28d)", f"{prediction:.2f} MPa")
                # معادلات تقريبية بناءً على الأبحاث لزيادة محتوى الموقع
                col2.metric("Tensile Strength (STS)", f"{(0.1 * prediction):.2f} MPa")
                col3.metric("Flexural Strength (FS)", f"{(0.7 * np.sqrt(prediction)):.2f} MPa")
                
                # رسم بياني توضيحي بسيط
                chart_data = pd.DataFrame({'Property': ['CS', 'STS', 'FS'], 'Value': [prediction, 0.1*prediction, 0.7*np.sqrt(prediction)]})
                st.bar_chart(data=chart_data, x='Property', y='Value')

            with tab2:
                st.subheader("Results: Durability Indicators")
                st.write("Estimated based on mix proportions:")
                water_abs = (w_c * 10) + (rca_p * 0.05)
                st.info(f"Predicted Water Absorption: **{water_abs:.2f} %**")

            with tab3:
                st.subheader("Environmental & Cost Analysis")
                # حساب انبعاثات الكربون (أرقام تقريبية من بحثك)
                co2_impact = (cement * 0.9) + (sf * 0.05) + (nylon * 1.5) - (rha_p * 0.2)
                # حساب التكلفة
                total_cost = (cement * 0.15) + (sp * 2.0) + (nylon * 5.0)
                
                c1, c2 = st.columns(2)
                c1.success(f"CO2 Footprint: {co2_impact:.2f} kg/m3")
                c2.warning(f"Estimated Cost: ${total_cost:.2f} per m3")

    else:
        st.error("Model files are missing or corrupted. Please upload them again to GitHub.")
