import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. إعدادات الصفحة
st.set_page_config(page_title="Eco-Efficient Concrete AI | Mansoura University", layout="wide")

# تنسيق المظهر
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #004a99; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .header-box { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 2px solid #004a99; text-align: center; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# 2. الهوية الأكاديمية
col_logo, col_title = st.columns([1, 4])
with col_logo:
    # شعار جامعة المنصورة
    st.image("https://upload.wikimedia.org/wikipedia/ar/thumb/0/01/Mansoura_University_logo.png/200px-Mansoura_University_logo.png", width=130)

with col_title:
    st.markdown(f"""
    <div class="header-box">
        <h2 style="color: #004a99; margin-bottom:0;">Multi-criteria analysis of eco-efficient concrete</h2>
        <p style="color: #555; font-size: 1.1em;">Technical, Environmental and Economic Aspects</p>
        <p style="margin-top:10px;"><b>By: Aya Mohammed Sanad Aboud</b></p>
        <p style="font-size: 0.9em; color: #666;">Under Supervision of: <b>Prof. Ahmed Tahwia</b> & <b>Assoc. Prof. Asser El-Sheikh</b></p>
    </div>
    """, unsafe_allow_html=True)

# بوابة الدخول
if "auth" not in st.session_state: st.session_state.auth = False
if not st.session_state.auth:
    with st.container():
        pwd = st.text_input("Access Password", type="password")
        if st.button("Login"):
            if pwd == "ASA2026": 
                st.session_state.auth = True
                st.rerun()
            else: st.error("Wrong Password")
    st.stop()

# 3. تحميل الموديلات
@st.cache_resource
def load_assets():
    model = joblib.load('concrete_model .pkl')
    scaler = joblib.load('scaler_weights .pkl')
    return model, scaler

model, scaler = load_assets()

# --- واجهة الإدخال ---
tab1, tab2, tab3 = st.tabs(["🚀 AI Prediction Engine", "📊 Statistical Validation", "📚 Research Database"])

with tab1:
    st.subheader("🛠️ Concrete Mix Proportions (kg/m³)")
    c1, c2, c3 = st.columns(3)
    with c1:
        cement = st.number_input("Cement", 100, 600, 350)
        water = st.number_input("Water", 100, 250, 175)
        nca = st.number_input("Natural Coarse Agg (NCA)", 500, 1500, 1050)
    with c2:
        nfa = st.number_input("Natural Fine Agg (NFA)", 300, 1000, 750)
        rca_p = st.slider("RCA Replacement (%)", 0, 100, 0)
        sf = st.number_input("Silica Fume", 0, 100, 0)
    with c3:
        fa = st.number_input("Fly Ash", 0, 200, 0)
        sp = st.number_input("Superplasticizer", 0.0, 15.0, 2.5)
        density = st.number_input("Target Density", 2000, 2600, 2400)

    # حساب الـ W/C والتحضير للتنبؤ (15 مدخلاً)
    wc = water/cement if cement > 0 else 0
    # ترتيب المدخلات كما تدرب الموديل
    input_arr = np.array([[cement, water, nca, nfa, rca_p, 0, sf, fa, 0, 0, sp, wc, 20, 100, density]])
    
    if st.button("Predict Strength"):
        scaled = scaler.transform(input_arr)
        preds = model.predict(scaled)[0]
        mae = 2.34 # هامش الخطأ الذي استخرجناه من كولاب

        st.divider()
        res1, res2, res3 = st.columns(3)
        res1.metric("CS 28d (MPa)", f"{preds[1]:.2f}", delta=f"± {mae}")
        res2.metric("CS 90d (MPa)", f"{preds[2]:.2f}")
        res3.metric("STS (MPa)", f"{preds[3]:.2f}")

        # رسم منحنى النمو الزمني
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(['7d', '28d', '90d'], [preds[0], preds[1], preds[2]], marker='o', color='#004a99')
        ax.set_title("Compressive Strength Development")
        st.pyplot(fig)

with tab2:
    st.markdown("### 📈 Accuracy Analysis")
    st.write("The model was validated using 400 experimental data points.")
    st.image("https://via.placeholder.com/600x400?text=Insert+Scatter+Plot+From+Colab") # هنا ترفعين صورة الـ Scatter Plot من كولاب
    st.info("Verified R-Squared: 0.941 | Mean Absolute Error: 2.34 MPa")

with tab3:
    st.subheader("📚 Dataset Sample")
    df_sample = pd.read_csv('Database_Inputs jimini2.csv', sep=';')
    st.dataframe(df_sample.head(20))
