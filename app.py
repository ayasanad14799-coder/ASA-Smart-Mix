import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json

# 1. إعدادات الصفحة
st.set_page_config(page_title="Eco-Concrete AI Optimizer", layout="wide")

# --- دالة إرسال البيانات لجوجل شيت (تم وضع رابطك الخاص هنا) ---
def send_to_sheets(data_dict):
    # الرابط المستخرج من صورتك
    url = "https://script.google.com/macros/s/AKfycbxuQLsHy5spA0BBFasF88JGaKf5JTXrw3vXU67hIBl4xsmhFfHBW3zubuwVbh49EQuWdg/exec"
    try:
        requests.post(url, data=json.dumps(data_dict))
    except:
        pass

# --- التنسيق الجمالي ---
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #004a99; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .header-box { background-color: #ffffff; padding: 20px; border-radius: 15px; border: 2px solid #004a99; text-align: center; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- الهوية الأكاديمية ---
st.markdown(f"""
    <div class="header-box">
        <h2 style="color: #004a99; margin-bottom:10px;">Multi-criteria analysis of eco-efficient concrete from Technical, Environmental and Economic aspects</h2>
        <p><b>Prepared by: Aya Mohammed Sanad Aboud</b> | Mansoura University</p>
        <p style="font-size: 0.9em; color: #666;">Supervision: Prof. Ahmed Tahwia & Assoc. Prof. Asser El-Sheikh</p>
    </div>
    """, unsafe_allow_html=True)

# --- تحميل الموديلات ---
@st.cache_resource
def load_assets():
    model = joblib.load('concrete_model .pkl')
    scaler = joblib.load('scaler_weights .pkl')
    return model, scaler

model, scaler = load_assets()

# --- القائمة الجانبية للمدخلات ---
with st.sidebar:
    st.header("⚙️ Mix Design (kg/m³)")
    cement = st.number_input("Cement", 100, 600, 350)
    water = st.number_input("Water", 100, 250, 175)
    nca = st.number_input("Natural Coarse Agg (NCA)", 500, 1500, 1050)
    nfa = st.number_input("Natural Fine Agg (NFA)", 300, 1000, 750)
    rca = st.slider("RCA Replacement (%)", 0, 100, 0)
    sp = st.number_input("Superplasticizer", 0.0, 15.0, 2.5)
    density = st.number_input("Density", 2000, 2600, 2400)
    st.divider()
    inflation = st.slider("Price Index Multiplier", 0.5, 2.5, 1.0)
    
    st.divider()
    run_btn = st.button("🚀 Run Multi-Criteria Analysis", type="primary", use_container_width=True)

# --- التبويبات ---
t1, t2, t3, t4 = st.tabs(["🏗️ Engineering Results", "💡 AI Optimizer", "📚 User Manual", "📊 Data Log"])

if run_btn:
    # 1. التحضير والتنبؤ
    wc = water/cement if cement > 0 else 0
    # المصفوفة الـ 15 (باقي القيم كأصفار أو قيم افتراضية حسب التدريب)
    inp = np.array([[cement, water, nca, nfa, rca, 0, 0, 0, 0, 0, sp, wc, 20, 100, density]])
    preds = model.predict(scaler.transform(inp))[0]
    
    # 2. إرسال البيانات لجوجل شيت تلقائياً
    log_data = {
        "Cement": cement, "Water": water, "RCA": rca, 
        "CS_28": round(float(preds[1]), 2), 
        "CO2": round(float(preds[11]), 2),
        "Sustainability": round(float(preds[16]), 3),
        "Cost": round(float(preds[13] * inflation), 2)
    }
    send_to_sheets(log_data)

    with t1:
        st.subheader("Technical, Environmental & Economic Profile")
        # عرض أهم 4 مخرجات في الأعلى
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CS 28d", f"{preds[1]:.2f} MPa", "± 2.34")
        c2.metric("CO2 Footprint", f"{preds[11]:.2f} kg/m³")
        c3.metric("Cost", f"${(preds[13]*inflation):.2f}")
        c4.metric("Sustainability Index", f"{preds[16]:.3f}")
        
        st.divider()
        # عرض باقي المخرجات الـ 17 في أعمدة
        st.markdown("#### Full Performance Breakdown")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.write("**Strength:**")
            st.write(f"- CS 7d: {preds[0]:.2f} MPa")
            st.write(f"- CS 90d: {preds[2]:.2f} MPa")
            st.write(f"- STS: {preds[3]:.2f} MPa")
        with m2:
            st.write("**Durability:**")
            st.write(f"- Water Absorption: {preds[6]:.2f}%")
            st.write(f"- UPV: {preds[7]:.2f} km/s")
            st.write(f"- Cl. Perm: {preds[10]:.0f} Coul.")
        with m3:
            st.write("**Physical:**")
            st.write(f"- Elastic Modulus: {preds[5]:.2f} GPa")
            st.write(f"- Shrinkage: {preds[8]:.0f} µε")
            st.write(f"- Carb. Depth: {preds[9]:.2f} mm")

with t2:
    st.header("💡 Eco-Mix Recommender")
    target_strength = st.number_input("Target 28-day Strength (MPa)", 20, 60, 35)
    if st.button("Suggest Optimal Eco-Mix"):
        with st.spinner("Analyzing 1000 combinations..."):
            # محاكاة سريعة
            sims = []
            for _ in range(1000):
                c = np.random.randint(320, 480)
                r = np.random.choice([0, 25, 50, 100])
                wc_s = 175/c
                test_inp = np.array([[c, 175, 1050, 750, r, 0, 0, 0, 0, 0, 3.0, wc_s, 20, 100, 2400]])
                p = model.predict(scaler.transform(test_inp))[0]
                if abs(p[1] - target_strength) < 1.5:
                    sims.append({'Cement': c, 'RCA%': r, 'CO2': p[11], 'Predicted_CS': p[1]})
            if sims:
                best = pd.DataFrame(sims).sort_values('CO2').iloc[0]
                st.success("Optimal Eco-Friendly Mix Found!")
                st.table(best)
            else: st.warning("Try another target strength.")

with t3:
    st.header("📖 User Documentation")
    st.markdown("""
    1. **Inputs:** Enter mix ingredients. Use the **Inflation Slider** for current market prices.
    2. **Analysis:** Click 'Run' to compute 17 mechanical and environmental properties.
    3. **Data Logging:** Every result is automatically saved to the research database for further analysis.
    4. **Optimizer:** Finds the mix with the lowest CO2 emissions for your required strength.
    """)

with t4:
    st.subheader("Real-time Data Log")
    st.info("The last 10 entries from your research database (CSV):")
    st.dataframe(pd.read_csv('Database_Inputs jimini2.csv', sep=';').tail(10))

st.caption("© 2026 Aya Sanad | Sustainable Concrete AI | Mansoura University")
