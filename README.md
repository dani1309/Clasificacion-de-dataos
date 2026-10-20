# 💳 Proyecto Final de Minería de Datos — Metodología CRISP-DM
## Predicción de Abandono de Clientes de Tarjetas de Crédito (*Credit Card Churn*)

Este repositorio contiene el desarrollo integral del **Proyecto Final de Minería de Datos**, aplicando todas las fases de la metodología **CRISP-DM** sobre el conjunto de datos real de abandono bancario (`credit_card_churn.csv` — 10,127 registros y 23 variables originales).

---

## 📂 Estructura de Entregables en el Repositorio

Todos los entregables principales se encuentran dentro de la carpeta [`Credit card Chum/`](./Credit%20card%20Chum/):

| Entregable Requerido | Archivo en el Repositorio | Descripción |
| :--- | :--- | :--- |
| **1. Documentación Metodología CRISP-DM** | [`METODOLOGIA - CRISP DM (8).docx`](./METODOLOGIA%20-%20CRISP%20DM%20(8).docx) / [`Credit card Chum/INFORME_METODOLOGIA_CRISP_DM_CHURN.docx`](./Credit%20card%20Chum/INFORME_METODOLOGIA_CRISP_DM_CHURN.docx) | Informe completo con todas las fases de CRISP-DM y las 8 tablas de la plantilla oficial diligenciadas. |
| **2. Notebook 01: Preparación de Datos + Profiling** | [`Credit card Chum/01_Preparacion_de_Datos.ipynb`](./Credit%20card%20Chum/01_Preparacion_de_Datos.ipynb) | Limpieza de datos, eliminación de *Data Leakage*, tratamiento de nulos con regla del 15%, atípicos, correlaciones de redundancia/irrelevancia y generación de Pandas Profiling. |
| **Reporte Pandas Profiling** | [`Credit card Chum/reporte_profiling_churn.html`](./Credit%20card%20Chum/reporte_profiling_churn.html) | Reporte interactivo HTML generado con `ydata-profiling`. |
| **Dataset Depurado** | [`Credit card Chum/credit_card_churn_limpio.csv`](./Credit%20card%20Chum/credit_card_churn_limpio.csv) | Dataset limpio (10,120 registros × 19 columnas) resultante del Notebook 01. |
| **3. Notebook 02: Modelado Predictivo Avanzado** | [`Credit card Chum/02_Modelado_Predictivo_Avanzado.ipynb`](./Credit%20card%20Chum/02_Modelado_Predictivo_Avanzado.ipynb) | Ingeniería de características, partición 70/30, balanceo SMOTE sólo en el 70%, validación cruzada 10-Fold de **7 modelos** (4 supervisados + 3 ensambles), 5 métricas de calidad, análisis estadístico **ANOVA + Tukey HSD**, hiperparametrización (**GridSearchCV + Optimización Bayesiana Optuna**) y exportación del `Pipeline`. |
| **Pipeline Entrenado** | [`Credit card Chum/modelo_churn_pipeline.joblib`](./Credit%20card%20Chum/modelo_churn_pipeline.joblib) | Objeto serializado con `ColumnTransformer` + `SMOTE` + `XGBClassifier` optimizado. |
| **4. Despliegue con Interfaz Gráfica Streamlit** | [`Credit card Chum/03_Despliegue_Streamlit.ipynb`](./Credit%20card%20Chum/03_Despliegue_Streamlit.ipynb) y [`Credit card Chum/app_streamlit.py`](./Credit%20card%20Chum/app_streamlit.py) | Notebook de validación de inferencia en datos futuros y aplicación web interactiva en Streamlit para predicción individual y por lote (CSV). |

---

## 🚀 Cómo Ejecutar la Interfaz Gráfica en Streamlit

1. Instalar las dependencias necesarias:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost scipy statsmodels optuna ydata-profiling streamlit joblib matplotlib seaborn
   ```
2. Ejecutar la aplicación desde la carpeta `Credit card Chum`:
   ```bash
   cd "Credit card Chum"
   streamlit run app_streamlit.py
   ```
