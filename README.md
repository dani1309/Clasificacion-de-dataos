# 💳 Proyecto Final de Minería de Datos, Metodología CRISP-DM
## Predicción de Abandono de Clientes de Tarjetas de Crédito (*Credit Card Churn*)

Este repositorio contiene el desarrollo integral del **Proyecto Final de Minería de Datos**, aplicando todas las fases de la metodología CRISP-DM sobre el conjunto de datos real de abandono bancario (`credit_card_churn.csv`, 10,127 registros y 23 variables originales).

---

## 📂 Estructura de Entregables en el Repositorio

Todos los entregables principales se encuentran dentro de la carpeta [`Credit card Churn/`](./Credit%20card%20Churn/):

| Entregable Requerido | Archivo en el Repositorio | Descripción |
| :--- | :--- | :--- |
| **1. Documentación Metodología CRISP-DM** | [`Credit card Churn/INFORME_METODOLOGIA_CRISP_DM_CHURN.docx`](./Credit%20card%20Churn/INFORME_METODOLOGIA_CRISP_DM_CHURN.docx) | Informe completo con todas las fases de CRISP-DM y las 8 tablas de la plantilla oficial diligenciadas. |
| **2. Notebook 01: Preparación de Datos + Profiling** | [`Credit card Churn/01_Preparacion_de_Datos.ipynb`](./Credit%20card%20Churn/01_Preparacion_de_Datos.ipynb) | Limpieza de datos, eliminación de *Data Leakage*, tratamiento de nulos con regla del 15%, atípicos, correlaciones de redundancia/irrelevancia y generación de Pandas Profiling. |
| **Reporte Pandas Profiling** | [`Credit card Churn/reporte_profiling_churn.html`](./Credit%20card%20Churn/reporte_profiling_churn.html) | Reporte interactivo HTML generado con `ydata-profiling`. |
| **Dataset Depurado** | [`Credit card Churn/credit_card_churn_limpio.csv`](./Credit%20card%20Churn/credit_card_churn_limpio.csv) | Dataset limpio (10,120 registros × 19 columnas) resultante del Notebook 01. |
| **3. Notebook 02: Modelado Predictivo Avanzado** | [`Credit card Churn/02_Modelado_Predictivo_Avanzado.ipynb`](./Credit%20card%20Churn/02_Modelado_Predictivo_Avanzado.ipynb) | Ingeniería de características, partición 70/30, balanceo SMOTE sólo en el 70%, validación cruzada 10-Fold de **7 modelos** (Supervisados: Árbol de Decisión, KNN, Red Neuronal MLP, SVM; Ensambles: Random Forest, XGBoost, Stacking), 5 métricas de calidad, análisis estadístico **ANOVA + Tukey HSD**, hiperparametrización (**GridSearchCV + Optimización Bayesiana Optuna**) y exportación del `Pipeline`. |
| **Pipeline Entrenado** | [`Credit card Churn/modelo_churn_pipeline.joblib`](./Credit%20card%20Churn/modelo_churn_pipeline.joblib) | `Pipeline` serializado que recibe las 18 predictoras crudas e incluye el reemplazo de `Unknown` por `NaN`, la imputación por moda de las categóricas, la ingeniería de características, el `ColumnTransformer`, `SMOTE` y el `XGBClassifier` optimizado. |
| **Funciones del Pipeline** | [`Credit card Churn/churn_features.py`](./Credit%20card%20Churn/churn_features.py) | Contiene `unknown_to_nan` y `add_engineered_features`. El modelo guardado depende de este archivo, por lo que debe mantenerse junto al `.joblib`. |
| **4. Despliegue con Interfaz Gráfica Streamlit** | [`Credit card Churn/03_Despliegue_Streamlit.ipynb`](./Credit%20card%20Churn/03_Despliegue_Streamlit.ipynb) y [`Credit card Churn/app_streamlit.py`](./Credit%20card%20Churn/app_streamlit.py) | Notebook de validación de inferencia en datos futuros y aplicación web interactiva en Streamlit para predicción individual y por lote (CSV). |

---

## 🚀 Cómo Ejecutar la Interfaz Gráfica en Streamlit

El entorno del proyecto es **Python 3.11**. El modelo guardado debe cargarse con las versiones exactas de `requirements.txt`, porque no es compatible con otras versiones de las librerías.

1. Instalar las dependencias desde la raíz del repositorio:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecutar la aplicación desde la carpeta `Credit card Churn`:
   ```bash
   cd "Credit card Churn"
   streamlit run app_streamlit.py
   ```
