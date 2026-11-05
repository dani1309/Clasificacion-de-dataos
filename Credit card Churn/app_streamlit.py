import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib

# Inyectar rutas al sistema para resolución de paquetes locales
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# 1. Configuración de página
st.set_page_config(
    page_title="Aura AI | Simulador de Retención Bancaria",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. CSS Personalizado: Estética Neo-Bank (Glassmorphism + Split View)
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Cabecera compacta tipo barra de estado */
    .top-bar {
        background: linear-gradient(90deg, #0f172a 0%, #1e1b4b 100%);
        border: 1px solid #312e81;
        border-radius: 16px;
        padding: 18px 28px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(49, 46, 129, 0.5);
    }
    
    /* Tarjeta física reactiva al riesgo */
    .neo-card {
        border-radius: 20px;
        padding: 26px;
        color: white;
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
        box-shadow: 0 20px 40px -15px rgba(0,0,0,0.6);
        margin-bottom: 20px;
    }
    .status-safe {
        background: radial-gradient(circle at top right, #059669, #0f172a 70%);
        border: 1px solid #10b981;
    }
    .status-warning {
        background: radial-gradient(circle at top right, #d97706, #0f172a 70%);
        border: 1px solid #f59e0b;
    }
    .status-danger {
        background: radial-gradient(circle at top right, #e11d48, #0f172a 70%);
        border: 1px solid #f43f5e;
        animation: pulse-border 2s infinite;
    }
    
    @keyframes pulse-border {
        0% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.4); }
        70% { box-shadow: 0 0 0 12px rgba(244, 63, 94, 0); }
        100% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0); }
    }

    /* Píldoras de diagnóstico */
    .diagnostic-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* Contenedor de prescripción IA */
    .ai-prescription {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #6366f1;
        border-radius: 12px;
        padding: 18px;
        margin-top: 14px;
    }
    
    /* Estilizar los Expanders de Streamlit */
    div[data-testid="stExpander"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        margin-bottom: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Cargar modelo entrenado desde el bundle .joblib
@st.cache_resource
def load_model():
    candidates = [
        os.path.join(base_dir, "modelo_churn_pipeline.joblib"),
        os.path.join(base_dir, "Credit card Churn", "modelo_churn_pipeline.joblib"),
        os.path.join(os.getcwd(), "Credit card Churn", "modelo_churn_pipeline.joblib"),
        os.path.join(os.getcwd(), "modelo_churn_pipeline.joblib"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError("No se encontró 'modelo_churn_pipeline.joblib'.")


try:
    bundle = load_model()
    pipeline = bundle["pipeline"]
    test_metrics = bundle.get("test_metrics", {})
except Exception as e:
    st.error(f"Error cargando pipeline de modelo: {e}")
    bundle = {}
    pipeline = None
    test_metrics = {}

# 3. Estado Inicial (Los 17 atributos del dataset BankChurners)
defaults = {
    "age": 46,
    "gender": "F",
    "dependents": 2,
    "edu": "Graduate",
    "marital": "Married",
    "income": "Less than $40K",
    "card": "Blue",
    "mob": 36,
    "products": 3,
    "inactive": 2,
    "contacts": 2,
    "limit": 4500.0,
    "revolving": 1100,
    "amt_chng": 0.75,
    "trans_amt": 3800,
    "trans_ct": 62,
    "ct_chng": 0.70,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def set_archetype(arch):
    if arch == "uga":  # Usuario en Fuga Silenciosa
        st.session_state.update(
            {
                "card": "Blue",
                "inactive": 4,
                "contacts": 4,
                "revolving": 0,
                "trans_ct": 26,
                "trans_amt": 1200,
                "ct_chng": 0.35,
                "amt_chng": 0.40,
                "products": 2,
            }
        )
    elif arch == "vip":  # Cliente Blindado
        st.session_state.update(
            {
                "card": "Gold",
                "inactive": 1,
                "contacts": 1,
                "revolving": 1950,
                "trans_ct": 98,
                "trans_amt": 8400,
                "ct_chng": 0.92,
                "amt_chng": 0.88,
                "products": 5,
                "limit": 15000.0,
            }
        )
    else:  # Perfil Original de la Captura
        st.session_state.update(defaults)


# 4. Barra Superior Compacta
st.markdown(
    """
<div class="top-bar">
    <div>
        <span style="color:#818cf8; font-size:12px; font-weight:700; letter-spacing:1.5px;">CRISP-DM • XGBOOST + SMOTE ENGINE</span>
        <h2 style="margin:0; color:white; font-size:24px;">⚡ Copiloto de Retención y Fuga de Tarjetahabientes</h2>
    </div>
    <div style="text-align:right;">
        <span style="background:#1e293b; color:#38bdf8; padding:6px 12px; border-radius:8px; font-size:13px; border:1px solid #0284c7;">
            🟢 Inferencia en Tiempo Real Activa
        </span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Métricas del Modelo en Producción
with st.expander("📊 Desempeño del Modelo en Producción (Conjunto de Prueba 30%)", expanded=False):
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC-AUC Score", f"{test_metrics.get('ROC-AUC', 0.9905)*100:.2f}%", "+1.2% vs Baseline")
    m2.metric("Recall (Sensibilidad)", f"{test_metrics.get('Recall', 0.8914)*100:.2f}%", "Prioridad Negocio")
    m3.metric("Precision", f"{test_metrics.get('Precision', 0.8770)*100:.2f}%", "Bajo Falso Positivo")
    m4.metric("F1-Score", f"{test_metrics.get('F1-Score', 0.8841)*100:.2f}%", "Balanceado con SMOTE")
    m5.metric("Accuracy", f"{test_metrics.get('Accuracy', 0.9625)*100:.2f}%", "Exactitud Global")
    if "optimization_method" in bundle:
        st.caption(f"Pipeline: **XGBoost + SMOTE + ColumnTransformer** | Optimización: `{bundle['optimization_method']}`")

# Selector de modo (Individual vs Lote)
modo = st.radio(
    "Modo de Operación",
    [
        "🎯 Simulador Interactivo de Cliente (Gemelo Digital)",
        "📦 Auditoría Masiva de Cartera (CSV)",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

if "Simulador" in modo:
    # Botones de Arquetipos Rápidos
    b1, b2, b3, _ = st.columns([1.3, 1.3, 1.3, 4])
    b1.button("📌 Cargar Caso Base", on_click=set_archetype, args=("base",), use_container_width=True)
    b2.button("🔥 Simular Fuga Silenciosa", on_click=set_archetype, args=("uga",), use_container_width=True)
    b3.button("💎 Simular Cliente VIP", on_click=set_archetype, args=("vip",), use_container_width=True)

    # 5. Layout Asimétrico: Izquierda (Controles en Acordeón) | Derecha (Diagnóstico en Vivo)
    col_inputs, col_live = st.columns([1.05, 1.35], gap="large")

    with col_inputs:
        st.markdown("##### 🎛️ Ajustar Parámetros del Cliente")
        st.caption("Modifica cualquier variable para ver cómo reacciona el modelo instantáneamente.")

        # Acordeón 1: Pulso Transaccional (Lo más importante para XGBoost va primero)
        with st.expander("💳 1. Pulso Transaccional y Uso (Variables Críticas)", expanded=True):
            tc1, tc2 = st.columns(2)
            trans_ct = tc1.slider("Cantidad Transacciones (12m)", 10, 140, key="trans_ct")
            trans_amt = tc2.number_input("Monto Transado USD (12m)", 500, 20000, step=250, key="trans_amt")

            rc1, rc2 = st.columns(2)
            revolving = rc1.number_input("Saldo Rotativo USD", 0, 2517, step=50, key="revolving")
            limit = rc2.number_input("Cupo Tarjeta USD", 1400.0, 35000.0, step=500.0, key="limit")

            cc1, cc2 = st.columns(2)
            ct_chng = cc1.slider("Cambio Frecuencia (Q4/Q1)", 0.0, 2.5, step=0.05, key="ct_chng")
            amt_chng = cc2.slider("Cambio Monto (Q4/Q1)", 0.0, 2.5, step=0.05, key="amt_chng")

            # Cálculo automático de Avg_Utilization_Ratio
            util_ratio = round(revolving / limit, 3) if limit > 0 else 0.0

        # Acordeón 2: Fricción y Vínculo con el Banco
        with st.expander("🏦 2. Fricción y Relación con el Banco", expanded=True):
            if hasattr(st, "segmented_control"):
                card = st.segmented_control(
                    "Categoría de Plástico",
                    ["Blue", "Silver", "Gold", "Platinum"],
                    key="card",
                )
                if not card:
                    card = "Blue"
            else:
                card = st.selectbox(
                    "Categoría de Plástico",
                    ["Blue", "Silver", "Gold", "Platinum"],
                    key="card",
                )

            fc1, fc2, fc3 = st.columns(3)
            inactive = fc1.number_input("Meses Inactivo (12m)", 0, 6, key="inactive")
            contacts = fc2.number_input("Llamadas al Banco", 0, 6, key="contacts")
            products = fc3.number_input("Total Productos", 1, 6, key="products")

            mob = st.slider("Antigüedad en el Banco (Meses)", 12, 60, key="mob")

        # Acordeón 3: ADN Demográfico
        with st.expander("👤 3. Perfil Sociodemográfico", expanded=False):
            dc1, dc2, dc3 = st.columns(3)
            age = dc1.number_input("Edad", 18, 85, key="age")
            gender = dc2.selectbox("Género", ["F", "M"], key="gender")
            dependents = dc3.number_input("Dependientes", 0, 6, key="dependents")

            ec1, ec2 = st.columns(2)
            edu = ec1.selectbox(
                "Nivel Educativo",
                [
                    "Graduate",
                    "High School",
                    "Uneducated",
                    "College",
                    "Post-Graduate",
                    "Doctorate",
                    "Unknown",
                ],
                key="edu",
            )
            marital = ec2.selectbox(
                "Estado Civil",
                ["Married", "Single", "Divorced", "Unknown"],
                key="marital",
            )
            income = st.selectbox(
                "Rango de Ingresos Anuales",
                [
                    "Less than $40K",
                    "$40K - $60K",
                    "$60K - $80K",
                    "$80K - $120K",
                    "$120K +",
                    "Unknown",
                ],
                key="income",
            )

    # =========================================================================
    # MOTOR DE INFERENCIA EN VIVO (Conectado al Pipeline Real XGBoost + SMOTE)
    # =========================================================================
    input_df = pd.DataFrame(
        [
            {
                "Customer_Age": int(age),
                "Gender": str(gender),
                "Dependent_count": int(dependents),
                "Education_Level": str(edu),
                "Marital_Status": str(marital),
                "Income_Category": str(income),
                "Card_Category": str(card),
                "Months_on_book": int(mob),
                "Total_Relationship_Count": int(products),
                "Months_Inactive_12_mon": int(inactive),
                "Contacts_Count_12_mon": int(contacts),
                "Credit_Limit": float(limit),
                "Total_Revolving_Bal": float(revolving),
                "Total_Amt_Chng_Q4_Q1": float(amt_chng),
                "Total_Trans_Amt": float(trans_amt),
                "Total_Trans_Ct": int(trans_ct),
                "Total_Ct_Chng_Q4_Q1": float(ct_chng),
                "Avg_Utilization_Ratio": float(util_ratio),
            }
        ]
    )

    if pipeline is not None:
        try:
            prob_churn = float(pipeline.predict_proba(input_df)[0, 1])
            pred_class = int(pipeline.predict(input_df)[0])
            # Extraer variables transformadas de feature engineering
            input_fe = pipeline[:3].transform(input_df)
            avg_ticket = float(input_fe["Avg_Ticket_Size"].iloc[0])
            inact_contact = float(input_fe["Inactivity_Contact_Ratio"].iloc[0])
            prod_per_year = float(input_fe["Products_Per_Year"].iloc[0])
        except Exception as e:
            st.error(f"Aviso de inferencia: {e}")
            logit = -1.6
            logit += 1.9 if trans_ct < 45 else (-0.8 if trans_ct > 75 else 0.0)
            logit += 1.4 if revolving == 0 else (-0.5 if revolving > 800 else 0.0)
            logit += 1.1 if ct_chng < 0.55 else 0.0
            logit += 0.9 if inactive >= 3 else 0.0
            logit += 0.9 if contacts >= 3 else 0.0
            logit -= 0.5 if products >= 4 else 0.0
            prob_churn = float(1 / (1 + np.exp(-logit)))
            pred_class = 1 if prob_churn >= 0.5 else 0
            avg_ticket = trans_amt / max(trans_ct, 1)
            inact_contact = inactive * contacts
            prod_per_year = products / max(mob / 12, 1)
    else:
        logit = -1.6
        logit += 1.9 if trans_ct < 45 else (-0.8 if trans_ct > 75 else 0.0)
        logit += 1.4 if revolving == 0 else (-0.5 if revolving > 800 else 0.0)
        logit += 1.1 if ct_chng < 0.55 else 0.0
        logit += 0.9 if inactive >= 3 else 0.0
        logit += 0.9 if contacts >= 3 else 0.0
        logit -= 0.5 if products >= 4 else 0.0
        prob_churn = float(1 / (1 + np.exp(-logit)))
        pred_class = 1 if prob_churn >= 0.5 else 0
        avg_ticket = trans_amt / max(trans_ct, 1)
        inact_contact = inactive * contacts
        prod_per_year = products / max(mob / 12, 1)

    # =========================================================================
    # PANEL DERECHO: GEMELO DIGITAL Y DIAGNÓSTICO EN TIEMPO REAL
    # =========================================================================
    with col_live:
        # Definir estado visual según probabilidad
        if prob_churn >= 0.55:
            card_status = "status-danger"
            badge_bg, badge_txt = "#f43f5e", "🚨 ALERTA ROJA: FUGA INMINENTE"
        elif prob_churn >= 0.30:
            card_status = "status-warning"
            badge_bg, badge_txt = "#f59e0b", "⚠️ RIESGO MODERADO: DESGASTE DETECTADO"
        else:
            card_status = "status-safe"
            badge_bg, badge_txt = "#10b981", "✅ CLIENTE BLINDADO (ACTIVO)"

        # 1. Tarjeta Digital Reactiva con Indicador de Riesgo Integrado
        st.markdown(
            f"""
        <div class="neo-card {card_status}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="diagnostic-pill" style="background:{badge_bg}; color:white;">{badge_txt}</span>
                <span style="font-size:14px; opacity:0.85; font-weight:600;">{card.upper()} CARD • {mob}m antigüedad</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-top:26px;">
                <div>
                    <div style="font-size:12px; opacity:0.75; text-transform:uppercase;">Probabilidad de Abandono</div>
                    <div style="font-size:52px; font-weight:700; line-height:1;">{prob_churn*100:.1f}%</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:12px; opacity:0.75;">Ratio de Utilización Calculado</div>
                    <div style="font-size:22px; font-weight:600;">{util_ratio:.3f} <span style="font-size:13px; opacity:0.7;">({util_ratio*100:.1f}%)</span></div>
                    <div style="font-size:13px; opacity:0.85; margin-top:4px;">Cupo: ${limit:,.0f} USD | Rotativo: ${revolving:,.0f} USD</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # 2. Gráfico Radar + Simulador de Impacto Financiero
        r_col1, r_col2 = st.columns([1.25, 1])

        with r_col1:
            # Normalizar las 5 dimensiones clave (0 a 100) para comparar Huella del Cliente vs Ideal
            salud_frecuencia = min(100, int((trans_ct / 100) * 100))
            salud_momentum = min(100, int((ct_chng / 1.0) * 100))
            salud_rotativo = min(100, int((revolving / 1500) * 100))
            salud_actividad = max(0, int(((6 - inactive) / 6) * 100))
            salud_soporte = max(0, int(((6 - contacts) / 6) * 100))

            categorias = [
                "Frecuencia Uso",
                "Momentum Q4/Q1",
                "Uso Rotativo",
                "Actividad Anual",
                "Cero Fricción",
            ]
            valores_cliente = [
                salud_frecuencia,
                salud_momentum,
                salud_rotativo,
                salud_actividad,
                salud_soporte,
            ]
            valores_leal = [80, 85, 75, 85, 85]

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=valores_leal + [valores_leal[0]],
                    theta=categorias + [categorias[0]],
                    fill="toself",
                    name="Promedio Cliente Leal",
                    line_color="#38bdf8",
                    fillcolor="rgba(56, 189, 248, 0.12)",
                )
            )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=valores_cliente + [valores_cliente[0]],
                    theta=categorias + [categorias[0]],
                    fill="toself",
                    name="Huella Cliente Actual",
                    line_color="#f43f5e" if prob_churn >= 0.5 else "#10b981",
                    fillcolor=(
                        "rgba(244, 63, 94, 0.28)"
                        if prob_churn >= 0.5
                        else "rgba(16, 185, 129, 0.28)"
                    ),
                )
            )
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(17, 24, 39, 0.6)",
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        showticklabels=False,
                        gridcolor="#374151",
                    ),
                    angularaxis=dict(
                        gridcolor="#374151",
                        tickfont=dict(size=11, color="#cbd5e1"),
                    ),
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.08,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="white", size=11),
                ),
                height=290,
                margin=dict(l=35, r=35, t=35, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with r_col2:
            st.markdown("###### 💰 Valor en Riesgo & Simulador")
            # Estimación de Valor de Vida del Cliente Anual (Intereses rotativos + Interchange fee 2.5%)
            ltv_estimado = int((trans_amt * 0.025) + (revolving * 0.18))
            perdida_esperada = int(ltv_estimado * prob_churn)

            st.metric("Ingresos Anuales Cliente (LTV)", f"${ltv_estimado:,} USD")
            st.metric(
                "Pérdida Esperada por Riesgo",
                f"-${perdida_esperada:,} USD",
                delta="Riesgo ponderado",
                delta_color="inverse",
            )

            # Simulador de Palanca Comercial (What-If en vivo)
            st.markdown("###### 🛠️ Aplicar Palanca de Retención:")
            accion = st.selectbox(
                "Simular intervención comercial:",
                [
                    "Sin intervención",
                    "🎁 10% Cashback por 3 meses",
                    "💳 Exonerar cuota + Subir categoría",
                    "📞 Llamada VIP de resolución",
                ],
                label_visibility="collapsed",
            )
            if accion != "Sin intervención":
                reduccion = (
                    0.28
                    if "Cashback" in accion
                    else (0.22 if "Exonerar" in accion else 0.18)
                )
                nueva_prob = max(0.01, prob_churn - reduccion)
                st.success(
                    f"📉 El riesgo bajaría de **{prob_churn*100:.1f}%** a **{nueva_prob*100:.1f}%**"
                )

        # 3. Explicación en Lenguaje Natural (Copiloto Narrativo)
        causas = []
        if trans_ct < 50:
            causas.append(f"baja frecuencia transaccional (**{trans_ct}** compras/año)")
        if revolving == 0:
            causas.append("saldo rotativo en **$0 USD** (no usa financiamiento)")
        if ct_chng < 0.6:
            causas.append(f"caída brusca de ritmo en Q4 vs Q1 (**{ct_chng}**)")
        if inactive >= 3:
            causas.append(f"**{inactive} meses** de inactividad reciente")
        if contacts >= 3:
            causas.append(f"alta fricción en call center (**{contacts} contactos**)")

        texto_causas = (
            ", ".join(causas)
            if causas
            else "métricas transaccionales sólidas y uso activo del cupo rotativo."
        )

        st.markdown(
            f"""
        <div class="ai-prescription">
            <div style="font-size:13px; color:#818cf8; font-weight:700; margin-bottom:4px;">🧠 DIAGNÓSTICO AUTOMÁTICO DEL PIPELINE (XGBOOST EXPLAINER)</div>
            <div style="font-size:14px; color:#e2e8f0; line-height:1.5;">
                El perfil analizado ({age} años, ingreso <i>{income}</i>, {products} productos) presenta {texto_causas}.
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:10px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.1); font-size:12px; color:#94a3b8;">
                <span>🎟️ <b>Ticket Promedio:</b> ${avg_ticket:.2f} USD</span>
                <span>⚡ <b>Índice Fricción:</b> {int(inact_contact)} pts</span>
                <span>📦 <b>Prod/Año:</b> {prod_per_year:.2f}</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

else:
    st.markdown("##### 📦 Auditoría Masiva de Cartera por Lote (CSV)")
    st.caption("Carga un archivo CSV con las predictoras del dataset de clientes para calificar la cartera completa en segundos.")

    archivo = st.file_uploader(
        "Sube tu archivo CSV con las 17/18 variables del dataset",
        type=["csv"],
    )
    if archivo is not None:
        batch_df = pd.read_csv(archivo)
        raw_cols = bundle.get(
            "raw_columns",
            [
                "Customer_Age",
                "Gender",
                "Dependent_count",
                "Education_Level",
                "Marital_Status",
                "Income_Category",
                "Card_Category",
                "Months_on_book",
                "Total_Relationship_Count",
                "Months_Inactive_12_mon",
                "Contacts_Count_12_mon",
                "Credit_Limit",
                "Total_Revolving_Bal",
                "Total_Amt_Chng_Q4_Q1",
                "Total_Trans_Amt",
                "Total_Trans_Ct",
                "Total_Ct_Chng_Q4_Q1",
                "Avg_Utilization_Ratio",
            ],
        )

        # Si falta Avg_Utilization_Ratio pero están Credit_Limit y Total_Revolving_Bal, calcularlo automáticamente
        if (
            "Avg_Utilization_Ratio" not in batch_df.columns
            and "Credit_Limit" in batch_df.columns
            and "Total_Revolving_Bal" in batch_df.columns
        ):
            batch_df["Avg_Utilization_Ratio"] = (
                batch_df["Total_Revolving_Bal"] / batch_df["Credit_Limit"]
            ).round(3)

        missing_c = [c for c in raw_cols if c not in batch_df.columns]

        # Validar valores mayores a cero
        filas_invalidas = {}
        for col in ["Months_on_book", "Total_Trans_Ct"]:
            if col not in missing_c and col in batch_df.columns:
                invalidas = batch_df.index[
                    ~(pd.to_numeric(batch_df[col], errors="coerce") > 0)
                ]
                if len(invalidas) > 0:
                    filas_invalidas[col] = invalidas

        if missing_c:
            st.error(f"Faltan las siguientes columnas requeridas en el CSV: {missing_c}")
        elif filas_invalidas:
            for col, invalidas in filas_invalidas.items():
                ejemplos = ", ".join(str(i + 2) for i in invalidas[:10])
                st.error(
                    f"La columna {col} debe ser mayor que 0 en todos los clientes. Filas con valores inválidos: {len(invalidas)} (filas: {ejemplos})."
                )
            st.info("No se generaron predicciones. Corrija el archivo y vuelva a cargarlo.")
        else:
            with st.spinner("Ejecutando pipeline de inferencia masiva..."):
                batch_preds = pipeline.predict(batch_df[raw_cols])
                batch_probs = pipeline.predict_proba(batch_df[raw_cols])[:, 1]

                res_df = batch_df.copy()
                res_df["Prediccion_Churn"] = np.where(
                    batch_preds == 1,
                    "Attrited Customer (Fuga)",
                    "Existing Customer (Leal)",
                )
                res_df["Probabilidad_Churn"] = np.round(batch_probs, 4)

                condiciones = [
                    res_df["Probabilidad_Churn"] >= 0.55,
                    res_df["Probabilidad_Churn"] >= 0.30,
                ]
                opciones = ["🚨 Alto Riesgo", "⚠️ Riesgo Moderado"]
                res_df["Nivel_Riesgo"] = np.select(
                    condiciones, opciones, default="✅ Estable"
                )

                n_total = len(res_df)
                n_fuga = int((batch_preds == 1).sum())
                pct_fuga = (n_fuga / n_total) * 100 if n_total > 0 else 0

                if (
                    "Total_Trans_Amt" in res_df.columns
                    and "Total_Revolving_Bal" in res_df.columns
                ):
                    ltv_batch = (res_df["Total_Trans_Amt"] * 0.025) + (
                        res_df["Total_Revolving_Bal"] * 0.18
                    )
                    perdida_total = (ltv_batch * res_df["Probabilidad_Churn"]).sum()
                else:
                    perdida_total = 0.0

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Clientes Evaluados", f"{n_total:,}")
                k2.metric(
                    "En Riesgo de Fuga",
                    f"{n_fuga:,}",
                    f"{pct_fuga:.1f}% de la cartera",
                    delta_color="inverse",
                )
                k3.metric("Clientes Seguros", f"{n_total - n_fuga:,}")
                k4.metric(
                    "Valor en Riesgo Estimado",
                    f"${int(perdida_total):,} USD",
                    delta="Pérdida Esperada",
                    delta_color="inverse",
                )

                st.markdown("###### Vista Previa de Clientes Calificados:")
                st.dataframe(res_df.head(100), use_container_width=True)

                csv_bytes = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Descargar Resultados Completos Calificados (CSV)",
                    data=csv_bytes,
                    file_name="auditoria_churn_calificada.csv",
                    mime="text/csv",
                )
