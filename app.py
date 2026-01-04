import streamlit as st
import pickle
import numpy as np

# 1. إعدادات الخصوصية (كلمة السر)
def check_password():
    def password_guessed():
        if st.session_state["password"] == "ASA2026":  # يمكنك تغيير كلمة السر هنا
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
    
    # --- بداية البرنامج الأصلي ---
    st.set_page_config(page_title="ASA Smart Concrete AI", layout="wide")

    @st.cache_resource
    def load_assets():
        with open('concrete_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler_weights.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler

    try:
        model, scaler = load_assets()
        
        st.sidebar.title("🛠️ ASA Input Parameters")
        # (بقية المدخلات كما هي في الكود السابق)
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

        input_data = np.array([[cement, water, nca, nfa, rca_p, rfa_p, sf, fa, rha_p, nylon, sp, w_c, agg_size, slump, density]])

        st.title("🏗️ ASA Smart Design & Sustainability Tool")
        tab1, tab2, tab3 = st.tabs(["💪 Strength", "💧 Durability", "🌍 Eco & Cost"])

        if st.sidebar.button("Predict Results"):
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)[0]

            with tab1:
                st.subheader("Predicted Mechanical Properties")
                st.metric("Compressive Strength (28d)", f"{prediction:.2f} MPa")
            
            with tab3:
                st.subheader("Sustainability Analysis")
                co2 = (cement * 0.9) + (sf * 0.05) + (nylon * 1.5)
                st.info(f"Estimated CO2 Footprint: {co2:.2f} kg/m3")

    except Exception as e:
        st.error(f"Error: {e}")
