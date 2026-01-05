import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. إعدادات الصفحة الاحترافية
st.set_page_config(page_title="Eco-Efficient Concrete AI | Mansoura University", layout="wide", page_icon="🏗️")

# --- تنسيق الواجهة (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #004a99; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .header-box { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 2px solid #004a99; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. الهوية الأكاديمية (الرأس) ---
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
        <p style="font-size: 0.8em; color: #888;">Mansoura University | Faculty of Engineering | 2026</p>
    </div>
    """, unsafe_allow_html=True)

# --- بوابة الدخول الآمن ---
if "auth" not in st.session_state: st.session_state.auth = False
if not st.session_state.auth:
    col_l, col_mid, col_r = st.columns([1, 2, 1])
    with col_mid:
        st.subheader("🔒 Secure Access Portal")
        with st.form("login"):
            pwd = st.text_input("Enter Access Code", type="password")
            if st.form_submit_button("Access Engine"):
                if pwd == "ASA2026": 
                    st.session_state.auth = True
                    st.rerun()
                else: st.error("❌ Invalid Code")
    st.stop()

# --- 3. تحميل الموديلات ---
@st.cache_resource
def load_assets():
    # تنبيه: تم كتابة الأسماء بالمسافات كما في ملفاتك الأصلية
    model = joblib.load('concrete_model .pkl')
    scaler = joblib.load('scaler_weights .pkl')
    return model, scaler

try:
    model, scaler = load_assets()

    # --- تقسيم التبويبات ---
    tab_engine, tab_validation, tab_database = st.tabs(["🚀 AI Prediction Engine", "📊 Model Validation", "📚 Research Database"])

    with tab_engine:
        st.subheader("🛠️ Concrete Mix Proportions (kg/m³)")
        c1, c2, c3 = st.columns(3)
        with c1:
            cement = st.number_input("Cement Content", 100, 600, 350)
            water = st.number_input("Water Content", 100, 250, 175)
            nca = st.number_input("Natural Coarse Agg (NCA)", 500, 1500, 1050)
            nfa = st.number_input("Natural Fine Agg (NFA)", 300, 1000, 750)
        with c2:
            rca_p = st.slider("RCA Replacement (%)", 0, 100, 0)
            rfa_p = st.slider("RFA Replacement (%)", 0, 100, 0)
            sf = st.number_input("Silica Fume", 0, 100, 0)
            fa = st.number_input("Fly Ash", 0, 200, 0)
        with c3:
            rha = st.number_input("RHA (%)", 0, 20, 0)
            fiber = st.number_input("Nylon Fiber", 0.0, 5.0, 0.0)
            sp = st.number_input("Superplasticizer", 0.0, 15.0, 2.5)
            density = st.number_input("Target Density", 2000, 2600, 2400)
            
        # حساب W/C وإعداد مصفوفة المدخلات (15 مدخلاً بالترتيب)
        wc = water/cement if cement > 0 else 0
        # المدخلات بالترتيب: Cement, Water, NCA, NFA, RCA_P, RFA_P, SF, FA, RHA_P, Fiber, SP, W/C, Size(20), Slump(100), Density
        input_arr = np.array([[cement, water, nca, nfa, rca_p, rfa_p, sf, fa, rha, fiber, sp, wc, 20, 100, density]])

        if st.button("Calculate Predicted Properties"):
            scaled = scaler.transform(input_arr)
            preds = model.predict(scaled)[0]
            mae = 2.34  # متوسط الخطأ العلمي المستخرج من المعايرة

            st.divider()
            st.markdown("### 🎯 Predicted Results")
            res1, res2, res3, res4 = st.columns(4)
            res1.metric("CS 28d (MPa)", f"{preds[1]:.2f}", delta=f"± {mae}")
            res2.metric("CS 90d (MPa)", f"{preds[2]:.2f}")
            res3.metric("STS (MPa)", f"{preds[3]:.2f}")
            res4.metric("FS (MPa)", f"{preds[4]:.2f}")

            # رسم منحنى التطور الزمني للمقاومة
            st.markdown("### 📈 Strength Development Profile")
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(['7 Days', '28 Days', '90 Days'], [preds[0], preds[1], preds[2]], marker='o', color='#004a99', linewidth=2)
            ax.set_ylabel("Strength (MPa)")
            ax.grid(True, alpha=0.2)
            st.pyplot(fig)

    with tab_validation:
        st.header("📊 Statistical Model Performance")
        v1, v2, v3 = st.columns(3)
        v1.metric("R-Squared (R²)", "0.941")
        v2.metric("Mean Absolute Error (MAE)", "2.34 MPa")
        v3.metric("Dataset Size", "400 Samples")
        
        st.info("The model has been rigorously validated against experimental data from 48 international research papers.")
        # ملاحظة: يمكنكِ رفع صورة مخطط التشتت هنا مستقبلاً
        st.markdown("---")
        st.write("### Predicted vs. Experimental Correlation")
        st.caption("Detailed scatter plots can be extracted from the Colab validation script.")

    with tab_database:
        st.header("📚 Training Dataset Overview")
        df_sample = pd.read_csv('Database_Inputs jimini2.csv', sep=';')
        st.dataframe(df_sample.head(50), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Check if model files are uploaded to GitHub correctly.")

# تذييل الصفحة
st.markdown("---")
st.caption("© 2026 Eco-Efficient Concrete AI Engine | Mansoura University Project")
