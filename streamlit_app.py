import os
import sys
import runpy

# Directorio raíz del repositorio
root_dir = os.path.dirname(os.path.abspath(__file__))
churn_dir = os.path.join(root_dir, "Credit card Churn")

# Inyectar rutas al sistema para que todos los módulos y el .joblib se resuelvan al 100%
if churn_dir not in sys.path:
    sys.path.insert(0, churn_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Ejecutar la aplicación principal
script_path = os.path.join(churn_dir, "app_streamlit.py")
runpy.run_path(script_path, run_name="__main__")
