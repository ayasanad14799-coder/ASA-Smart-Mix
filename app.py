import streamlit as st
import joblib # استخدمنا joblib بدل pickle
import numpy as np
import pandas as pd

# 1. إعدادات الخصوصية
def check_password():
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
                placeholder.empty()
                return True
            else:
                st.error("❌ Wrong password")
                return False
    return False

if check_password():
    st.set_page_config(page_title="ASA Smart Concrete AI", layout="wide")

    # 2. تحميل الموديل باستخدام joblib (أكثر استقراراً)
    @st.cache_resource
    def load_assets():
        try:
            # استخدام joblib مباشرة لقراءة الملفات
            model = joblib.load('concrete_model.pkl')
            scaler = joblib.load('scaler_weights.pkl')
            return model, scaler
        except Exception as e:
            st.error(f"Error: {e}")
            return None, None

    model, scaler = load_assets()

    if model is not None:
        st.title("🏗️ ASA Smart Design & Sustainability Tool")
        # ... بقية الكود كما هو ...
        st.sidebar.title("🛠️ Parameters")
        cement = st.sidebar.number_input("Cement", value=400)
        # (بقية المدخلات)
        if st.sidebar.button("Predict"):
            # تجميع الـ 15 مدخل بنفس الترتيب
            # ملاحظة: تأكدي أن عدد المدخلات هنا يطابق الـ 15 عمود
            inputs = np.array([[cement, 165, 1150, 750, 25, 0, 0, 0, 0, 0.0, 4.0, 0.4, 10, 100, 2400]])
            scaled = scaler.transform(inputs)
            res = model.predict(scaled)[0]
            st.metric("Strength", f"{res:.2f} MPa")
