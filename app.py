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
    .login-title { color: #004a99; text-align: center; font-weight: bold; margin-top: 10px; margin-bottom: 20px; font-size: 24px; }
    </style>
    """, unsafe_allow_html=True)

# 2. نظام الدخول الآمن (واجهة الدخول بشعارين واسم المشروع)
if "auth" not in st.session_state: st.session_state.auth = False
if not st.session_state.auth:
    col_l, col_mid, col_r = st.columns([1, 2, 1])
    with col_mid:
        # عرض الشعارات جنباً إلى جنب (الجامعة والكلية)
        logo_col1, logo_col2 = st.columns(2)
        with logo_col1:
            st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-Smart-Mix/main/LOGO.png", width=140)
        with logo_col2:
            st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-Smart-Mix/main/OIP.jfif", width=140)
        
        st.markdown("<h2 class='login-title'>ASA Smart Mix: AI-Based Eco-Concrete Optimizer</h2>", unsafe_allow_html=True)
        
        with st.form("login"):
            st.markdown("<p style='text-align: center; color: #555;'>Enter the professional access code to unlock the engine</p>", unsafe_allow_html=True)
            pwd = st.text_input("Access Code", type="password")
            if st.form_submit_button("🚀 Unlock Engine"):
                if pwd == "ASA2026": 
                    st.session_state.auth = True
                    st.rerun()
                else: st.error("❌ Access Denied: Invalid Code")
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
    st.error("Error: Model files not found. Check repository files.")
    st.stop()

metrics_real = {"R2": 0.9557, "RMSE": 2.91, "COV": "6.16%"}

# 4. الهيدر الأكاديمي الداخلي (يظهر بعد الدخول)
st.markdown(f"""
    <div class="header-container">
        <img src="https://raw.githubusercontent.com/ayasanad14799-coder/ASA-Smart-Mix/main/LOGO.png" style="width: 120px; margin-right: 25px;">
        <div style="text-align: center;">
            <h2 style="color: #004a99; margin-bottom:5px;">Multi-criteria analysis of eco-efficient concrete from Technical, Environmental and Economic aspects</h2>
            <p style="font-size: 1.1em; margin-bottom:5px;"><b>Prepared by: Aya Mohammed Sanad Aboud</b></p>
            <p style="color: #666; margin-bottom:5px;">Supervision: <b>Prof. Ahmed Tahwia</b> & <b>Assoc. Prof. Asser El-Sheikh</b></p>
            <p style="color: #004a9
