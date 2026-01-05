import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json

# 1. الإعدادات العامة والهوية
st.set_page_config(page_title="Eco-Concrete AI Optimizer", layout="wide")

def send_to_sheets(data):
    # الرابط الجديد الخاص بكِ
    url = "https://script.google.com/macros/s/AKfycbxv_xvhImquXOtWAF7RbjKW6hMDyxL4LumA8G7LCXAcxFZvp8f-18tl6y0mvMGUtOG1/exec"
    try:
        # إرسال البيانات بصيغة JSON لضمان دقة الـ 32 عمود
        requests.post(url, json=data, timeout=10)
    except:
        pass

st.markdown("""
    <div style="background-color: white; padding: 20px; border-radius: 15px; border: 2px solid #004a99; text-align: center;">
        <h2 style="color: #004a99;">Multi-criteria analysis of eco-efficient concrete</h2>
        <p><b>Prepared by: Aya Mohammed Sanad Aboud</b> | Mansoura University</p>
    </div>
    """, unsafe_allow_html=True)

# 2. حماية البرنامج
if "auth" not in st.session_state: st.session_state.auth = False
if not st.session_state.auth:
    pwd = st.sidebar.text_input("Enter Access Code", type="password")
    if st.sidebar.button("Login"):
        if pwd == "ASA2026": st.session_state.auth = True; st.rerun()
    st.stop()

# 3. تحميل الموديلات
@st.cache_resource
def load_assets():
    model = joblib.load('concrete_model .pkl')
    scaler = joblib.load('scaler_weights .pkl')
    return model, scaler
model, scaler = load_assets()

# 4. القائمة الجانبية (15 مدخل كاملين)
with st.sidebar:
    st.header("⚙️ Mix Ingredients (kg/m³)")
    c = st.number_input("Cement", 100, 600, 350)
    w = st.number_input("Water", 100, 250, 175)
    nca = st.number_input("NCA (Sinn)", 500, 1500, 1050)
    nfa = st.number_input("NFA (Sand)", 300, 1000, 750)
    rca = st.slider("RCA %", 0, 100, 0)
    rfa = st.slider("RFA %", 0, 100, 0)
    sf = st.number_input("Silica Fume", 0, 100, 0)
    fa = st.number_input("Fly Ash", 0, 200, 0)
    rha = st.number_input("Rice Husk Ash %", 0, 20, 0)
    fib = st.number_input("Nylon Fiber", 0.0, 5.0, 0.0)
    sp = st.number_input("Superplasticizer", 0.0, 15.0, 2.5)
    sz = st.selectbox("Max Agg Size", [10, 20, 40], index=1)
    sl = st.number_input("Slump Target", 0, 250, 100)
    den = st.number_input("Density", 2000, 2600, 2400)
    wc = w/c if c > 0 else 0
    inf = st.slider("Inflation Index", 0.5, 2.5, 1.0)
    run = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

# 5. التبويبات
t1, t2, t3, t4 = st.tabs(["🏗️ Results", "🛡️ Durability", "🌍 LCA", "💡 AI Optimizer"])

if run:
    # التنبؤ
    inp = np.array([[c, w, nca, nfa, rca, rfa, sf, fa, rha, fib, sp, wc, sz, sl, den]])
    p = model.predict(scaler.transform(inp))[0]
    
    # تجهيز الـ 32 متغير للإرسال (مطابق لأسماء السكريبت)
    full_data = {
        "c":c,"w":w,"nca":nca,"nfa":nfa,"rca":rca,"rfa":rfa,"sf":sf,"fa":fa,"rha":rha,"fib":fib,"sp":sp,"sz":sz,"sl":sl,"den":den,"wc":wc,
        "p0":p[0],"p1":p[1],"p2":p[2],"p3":p[3],"p4":p[4],"p5":p[5],"p6":p[6],"p7":p[7],
        "p8":p[8],"p9":p[9],"p10":p[10],"p11":p[11],"p12":p[12],"p13":p[13]*inf,"p14":p[14],"p16":p[16]
    }
    send_to_sheets(full_data)

    with t1:
        st.subheader("Mechanical Strength Profile")
        col1, col2, col3 = st.columns(3)
        col1.metric("CS 28d", f"{p[1]:.2f} MPa")
        col2.metric("STS", f"{p[3]:.2f} MPa")
        col3.metric("FS", f"{p[4]:.2f} MPa")
        
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(['7d', '28d', '90d'], [p[0], p[1], p[2]], marker='o', color='#004a99')
        st.pyplot(fig)

with t4:
    st.header("💡 AI-Based Full Mix Optimizer")
    target_strength = st.number_input("Enter Target Strength (28d)", 20, 75, 40)
    if
