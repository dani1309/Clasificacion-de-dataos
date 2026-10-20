import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Predicción de Abandono Bancario (Credit Card Churn)",
    page_icon="💳",
    layout="wide"
)

def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d['Avg_Ticket_Size'] = d['Total_Trans_Amt'] / np.maximum(d['Total_Trans_Ct'], 1)
    d['Inactivity_Contact_Ratio'] = d['Months_Inactive_12_mon'] * d['Contacts_Count_12_mon']
    d['Products_Per_Year'] = d['Total_Relationship_Count'] / (d['Months_on_book'] / 12.0)
    return d

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "modelo_churn_pipeline.joblib")
    return joblib.load(model_path)

bundle = load_model()
pipeline = bundle["pipeline"]
test_metrics = bundle["test_metrics"]
best_params = bundle["best_params"]

st.title("💳 Sistema Inteligente de Predicción de Abandono de Clientes (Credit Card Churn)")
st.markdown(
    """
    **Proyecto Final de Minería de Datos — Metodología CRISP-DM**  
    Esta aplicación interactiva utiliza el **`Pipeline` de Machine Learning ganador (XGBoost Optimizado + SMOTE + Preprocesador)** para estimar en tiempo real la probabilidad de que un cliente cancele su tarjeta de crédito (*Attrited Customer*).
    """
)

with st.expander("📊 Ver Desempeño del Modelo en Producción (Conjunto de Prueba 30%)", expanded=False):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("F1-Score", f"{test_metrics['F1-Score']*100:.2f}%")
    c2.metric("Recall (Sensibilidad)", f"{test_metrics['Recall']*100:.2f}%")
    c3.metric("Precision", f"{test_metrics['Precision']*100:.2f}%")
    c4.metric("Accuracy", f"{test_metrics['Accuracy']*100:.2f}%")
    c5.metric("ROC-AUC", f"{test_metrics['ROC-AUC']*100:.2f}%")
    st.caption(f"Método de optimización ganador: **{bundle['optimization_method']}** | Hiperparámetros: `{best_params}`")

st.divider()

tab1, tab2 = st.tabs(["👤 Predicción Individual de Cliente", "📂 Predicción Masiva por Lote (CSV)"])

with tab1:
    st.subheader("Ingrese el Perfil Demográfico y Financiero del Cliente")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 1. Datos Demográficos")
        customer_age = st.slider("Edad del Cliente (Customer_Age)", 18, 85, 46)
        gender = st.selectbox("Género (Gender)", ["F", "M"])
        dependent_count = st.slider("Número de Dependientes (Dependent_count)", 0, 6, 2)
        education_level = st.selectbox(
            "Nivel Educativo (Education_Level)",
            ["Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate"],
            index=3
        )
        marital_status = st.selectbox(
            "Estado Civil (Marital_Status)",
            ["Married", "Single", "Divorced"],
            index=0
        )
        income_category = st.selectbox(
            "Categoría de Ingresos Anuales (Income_Category)",
            ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"],
            index=0
        )

    with col2:
        st.markdown("##### 2. Relación con el Banco")
        card_category = st.selectbox(
            "Categoría de Tarjeta (Card_Category)",
            ["Blue", "Silver", "Gold", "Platinum"],
            index=0
        )
        months_on_book = st.slider("Antigüedad en el Banco en Meses (Months_on_book)", 12, 60, 36)
        total_rel_count = st.slider("Total de Productos Contratados (Total_Relationship_Count)", 1, 6, 3)
        months_inactive = st.slider("Meses Inactivo en Últimos 12 Meses (Months_Inactive_12_mon)", 0, 6, 2)
        contacts_count = st.slider("Contactos al Banco en Últimos 12 Meses (Contacts_Count_12_mon)", 0, 6, 2)
        credit_limit = st.number_input("Cupo / Límite de Crédito USD (Credit_Limit)", min_value=1000.0, max_value=40000.0, value=4500.0, step=500.0)

    with col3:
        st.markdown("##### 3. Comportamiento Transaccional")
        total_revolving_bal = st.number_input("Saldo Rotativo Total USD (Total_Revolving_Bal)", min_value=0, max_value=3000, value=1100, step=100)
        total_amt_chng = st.number_input("Cambio en Monto Q4 vs Q1 (Total_Amt_Chng_Q4_Q1)", min_value=0.0, max_value=3.5, value=0.75, step=0.05)
        total_trans_amt = st.number_input("Monto Total Transado 12m USD (Total_Trans_Amt)", min_value=100, max_value=25000, value=3800, step=200)
        total_trans_ct = st.slider("Cantidad de Transacciones 12m (Total_Trans_Ct)", 1, 150, 62)
        total_ct_chng = st.number_input("Cambio en Frecuencia Q4 vs Q1 (Total_Ct_Chng_Q4_Q1)", min_value=0.0, max_value=3.8, value=0.70, step=0.05)
        avg_util_ratio = st.slider("Ratio Promedio de Utilización (Avg_Utilization_Ratio)", 0.0, 1.0, round(min(total_revolving_bal / max(credit_limit, 1.0), 1.0), 3), step=0.01)

    if st.button("🔍 Evaluar Riesgo de Abandono (Churn)", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "Customer_Age": customer_age,
            "Gender": gender,
            "Dependent_count": dependent_count,
            "Education_Level": education_level,
            "Marital_Status": marital_status,
            "Income_Category": income_category,
            "Card_Category": card_category,
            "Months_on_book": months_on_book,
            "Total_Relationship_Count": total_rel_count,
            "Months_Inactive_12_mon": months_inactive,
            "Contacts_Count_12_mon": contacts_count,
            "Credit_Limit": credit_limit,
            "Total_Revolving_Bal": total_revolving_bal,
            "Total_Amt_Chng_Q4_Q1": total_amt_chng,
            "Total_Trans_Amt": total_trans_amt,
            "Total_Trans_Ct": total_trans_ct,
            "Total_Ct_Chng_Q4_Q1": total_ct_chng,
            "Avg_Utilization_Ratio": avg_util_ratio
        }])

        input_fe = add_engineered_features(input_df)
        pred_class = int(pipeline.predict(input_fe)[0])
        pred_prob = float(pipeline.predict_proba(input_fe)[0, 1])

        st.divider()
        r1, r2 = st.columns([1, 2])
        with r1:
            if pred_class == 1:
                st.error("### 🚨 ALERTA: ALTO RIESGO DE ABANDONO (Attrited Customer)")
            else:
                st.success("### ✅ CLIENTE ESTABLE / RETENIDO (Existing Customer)")
            st.metric("Probabilidad Estimada de Fuga", f"{pred_prob*100:.2f}%")
            st.progress(min(max(pred_prob, 0.0), 1.0))

        with r2:
            st.markdown("#### 💡 Diagnóstico e Indicadores de Ingeniería de Características")
            st.write({
                "Ticket Promedio por Transacción (USD)": round(float(input_fe["Avg_Ticket_Size"].iloc[0]), 2),
                "Índice de Fricción / Inactividad": int(input_fe["Inactivity_Contact_Ratio"].iloc[0]),
                "Productos Contratados por Año": round(float(input_fe["Products_Per_Year"].iloc[0]), 2)
            })
            if pred_prob >= 0.50:
                st.warning(
                    "**Acción Comercial Recomendada:** Asignar de inmediato al equipo de Retención Proactiva. "
                    "Ofrecer incentivos de uso transaccional (cashback/puntos), exoneración de cuota de manejo "
                    "y seguimiento prioritario a sus últimos contactos con servicio al cliente."
                )
            else:
                st.info(
                    "**Acción Comercial Recomendada:** Mantener en campañas de fidelización estándar y evaluar "
                    "oportunidades de venta cruzada (*cross-selling*) según su capacidad crediticia."
                )

with tab2:
    st.subheader("Carga de Archivo CSV para Predicción por Lote")
    st.markdown("Suba un archivo CSV con las columnas del dataset limpio (`credit_card_churn_limpio.csv`) para obtener las predicciones masivas.")
    uploaded_file = st.file_uploader("Seleccione un archivo CSV", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        raw_cols = bundle["raw_columns"]
        missing_c = [c for c in raw_cols if c not in batch_df.columns]
        if missing_c:
            st.error(f"Faltan las siguientes columnas requeridas en el CSV: {missing_c}")
        else:
            batch_fe = add_engineered_features(batch_df[raw_cols])
            batch_preds = pipeline.predict(batch_fe)
            batch_probs = pipeline.predict_proba(batch_fe)[:, 1]
            res_df = batch_df.copy()
            res_df["Prediccion_Churn"] = np.where(batch_preds == 1, "Attrited Customer", "Existing Customer")
            res_df["Probabilidad_Churn"] = np.round(batch_probs, 4)
            st.success(f"Se procesaron exitosamente {len(res_df)} clientes. Clientes en riesgo detectados: {(batch_preds == 1).sum()} ({(batch_preds == 1).mean()*100:.2f}%)")
            st.dataframe(res_df.head(50), use_container_width=True)
            csv_bytes = res_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar Resultados con Predicciones (CSV)", data=csv_bytes, file_name="predicciones_churn_lote.csv", mime="text/csv")
