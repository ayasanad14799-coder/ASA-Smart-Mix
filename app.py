# ... (جزء الأمان وتحميل الموديل يبقى كما هو)

        if st.sidebar.button("🚀 Run Comprehensive AI Analysis", use_container_width=True):
            inputs = np.array([[c, w, nca, nfa, rca, rfa, sf, fa, rha, nylon, sp, w_c, msa, slump, dens]])
            scaled_inputs = scaler.transform(inputs)
            prediction = model.predict(scaled_inputs)[0]

            with tab1:
                st.subheader("📊 Mechanical Strength Prediction")
                col1, col2, col3 = st.columns(3)
                # الربط الصحيح بناءً على ملفك الجديد
                col1.metric("CS (28 Days)", f"{prediction[1]:.2f} MPa")
                col2.metric("CS (90 Days)", f"{prediction[2]:.2f} MPa")
                col3.metric("Modulus of Elasticity (EM)", f"{prediction[5]:.2f} GPa")

                st.markdown("---")
                col4, col5 = st.columns(2)
                col4.metric("Split Tensile (STS)", f"{prediction[3]:.2f} MPa")
                col5.metric("Flexural (FS)", f"{prediction[4]:.2f} MPa")

                # رسم بياني للقوة
                chart_data = pd.DataFrame({
                    'Metric': ['CS 28d', 'CS 90d', 'STS', 'FS'],
                    'Value (MPa)': [prediction[1], prediction[2], prediction[3], prediction[4]]
                })
                st.bar_chart(chart_data, x='Metric', y='Value (MPa)')

            with tab2:
                st.subheader("💧 Durability Indicators")
                d1, d2 = st.columns(2)
                d1.metric("Water Absorption", f"{prediction[6]:.2f} %")
                d2.metric("Chloride Permeability", f"{prediction[10]:.1f} Coulombs")

            with tab3:
                st.subheader("🌍 Sustainability & Cost")
                # التكلفة هي Index 13 في ملفك
                base_cost = prediction[13]
                
                st.write("🔧 **Market Price Adjuster**")
                multiplier = st.number_input("Inflation Factor", value=1.0)
                
                s1, s2, s3 = st.columns(3)
                s1.metric("CO2 Footprint", f"{prediction[11]:.2f} kg/m³")
                s2.metric("Energy", f"{prediction[12]:.1f} MJ/m³")
                s3.metric("Final Cost", f"${base_cost * multiplier:.2f}")
