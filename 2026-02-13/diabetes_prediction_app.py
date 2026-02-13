"""
🏥 Aplicación de Predicción de Diabetes
========================================
Aplicación Streamlit para consultar predicciones del modelo de diabetes
desplegado con MLflow.

Para ejecutar:
    streamlit run diabetes_prediction_app.py

Asegúrate de que el servidor MLflow esté corriendo:
    mlflow models serve -m "models:/diabetes-prediction-model/1" --port 5001 --env-manager=local
"""

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="🏥 Predicción de Diabetes",
    page_icon="🏥",
    layout="wide"
)

# URL del servidor MLflow
MLFLOW_SERVER_URL = "http://127.0.0.1:5001/invocations"

# Información sobre las características con valores REALES (no normalizados)
# Incluye media y desviación estándar para normalización
FEATURE_INFO = {
    "age": {
        "name": "Edad",
        "description": "Edad del paciente en años",
        "unit": "años",
        "min": 20,
        "max": 80,
        "default": 50,
        "mean": 48.5,  # Media aproximada del dataset
        "std": 13.0    # Desviación estándar aproximada
    },
    "sex": {
        "name": "Sexo",
        "description": "Sexo biológico del paciente",
        "unit": "",
        "options": ["Femenino", "Masculino"],
        "default": 0,
        "mean": 0.5,
        "std": 0.5
    },
    "bmi": {
        "name": "IMC (Índice de Masa Corporal)",
        "description": "Peso(kg) / Altura(m)²",
        "unit": "kg/m²",
        "min": 18.0,
        "max": 45.0,
        "default": 26.0,
        "mean": 26.4,
        "std": 4.4
    },
    "bp": {
        "name": "Presión Arterial Media",
        "description": "Presión arterial media en reposo",
        "unit": "mmHg",
        "min": 60,
        "max": 130,
        "default": 95,
        "mean": 94.6,
        "std": 13.8
    },
    "s1": {
        "name": "Colesterol Total (TC)",
        "description": "Nivel total de colesterol en sangre",
        "unit": "mg/dL",
        "min": 100,
        "max": 350,
        "default": 200,
        "mean": 199.0,
        "std": 35.0
    },
    "s2": {
        "name": "Colesterol LDL",
        "description": "Colesterol 'malo' - Lipoproteína de baja densidad",
        "unit": "mg/dL",
        "min": 50,
        "max": 250,
        "default": 115,
        "mean": 115.0,
        "std": 30.0
    },
    "s3": {
        "name": "Colesterol HDL",
        "description": "Colesterol 'bueno' - Lipoproteína de alta densidad",
        "unit": "mg/dL",
        "min": 20,
        "max": 100,
        "default": 50,
        "mean": 52.0,
        "std": 13.0
    },
    "s4": {
        "name": "Ratio TC/HDL",
        "description": "Proporción Colesterol Total / HDL",
        "unit": "",
        "min": 1.5,
        "max": 8.0,
        "default": 4.0,
        "mean": 4.0,
        "std": 1.3
    },
    "s5": {
        "name": "Triglicéridos",
        "description": "Nivel de triglicéridos en sangre",
        "unit": "mg/dL",
        "min": 50,
        "max": 400,
        "default": 150,
        "mean": 150.0,
        "std": 80.0
    },
    "s6": {
        "name": "Glucosa en Ayunas",
        "description": "Nivel de glucosa en sangre en ayunas",
        "unit": "mg/dL",
        "min": 70,
        "max": 200,
        "default": 100,
        "mean": 91.0,
        "std": 11.5
    }
}


def normalize_value(value, feature_key):
    """
    Normaliza un valor real a la escala del modelo.
    Usa estandarización: (valor - media) / desviación_estándar
    """
    info = FEATURE_INFO[feature_key]
    
    # Para triglicéridos, aplicamos log antes de normalizar (como en el dataset original)
    if feature_key == "s5":
        value = np.log(value)
        # Ajustar media y std para el log
        mean = np.log(150)  # log de la media
        std = 0.5
        return (value - mean) / std
    
    return (value - info["mean"]) / info["std"]


def normalize_patient_data(raw_data):
    """
    Convierte los datos del paciente en valores reales a valores normalizados
    para enviar al modelo.
    """
    normalized = {}
    
    for feature, value in raw_data.items():
        # Normalizar usando media y desviación estándar
        normalized[feature] = normalize_value(value, feature)
    
    # Escalar para que coincida con el rango del dataset de sklearn
    # El dataset original tiene los valores escalados aproximadamente entre -0.15 y 0.15
    scale_factor = 0.05  # Factor de escala aproximado
    
    for key in normalized:
        normalized[key] = normalized[key] * scale_factor
    
    return normalized


def get_risk_level(prediction):
    """Determina el nivel de riesgo basado en la predicción"""
    if prediction < 100:
        return "🟢 Bajo", "success"
    elif prediction < 200:
        return "🟡 Moderado", "warning"
    else:
        return "🔴 Alto", "error"


def make_prediction(patient_data):
    """Envía los datos al servidor MLflow y obtiene la predicción"""
    try:
        # Normalizar los datos antes de enviar
        normalized_data = normalize_patient_data(patient_data)
        
        # Formato para MLflow serving
        payload = {
            "dataframe_split": {
                "columns": list(normalized_data.keys()),
                "data": [list(normalized_data.values())]
            }
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            MLFLOW_SERVER_URL,
            data=json.dumps(payload),
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            # MLflow devuelve una lista de predicciones
            prediction = result["predictions"][0] if "predictions" in result else result[0]
            return {"success": True, "prediction": prediction, "normalized_data": normalized_data}
        else:
            return {"success": False, "error": f"Error del servidor: {response.status_code} - {response.text}"}
            
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "⚠️ No se puede conectar al servidor MLflow. Asegúrate de que el servidor esté corriendo en el puerto 5001."}
    except Exception as e:
        return {"success": False, "error": f"Error: {str(e)}"}


# Título principal
st.title("🏥 Sistema de Predicción de Progresión de Diabetes")
st.markdown("""
Esta aplicación utiliza un modelo de Machine Learning desplegado con **MLflow** 
para predecir la progresión de diabetes en pacientes basándose en 10 características clínicas.

**Los datos se introducen en unidades médicas reales** y se normalizan automáticamente.
""")

# Verificar conexión con el servidor
st.sidebar.header("🔗 Estado del Servidor")
try:
    response = requests.get("http://127.0.0.1:5001/health", timeout=2)
    st.sidebar.success("✅ Servidor MLflow conectado")
except:
    st.sidebar.error("❌ Servidor MLflow no disponible")
    st.sidebar.code("""
# Para iniciar el servidor:
mlflow models serve \\
  -m "models:/diabetes-prediction-model/1" \\
  --port 5001 \\
  --env-manager=local
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### ℹ️ Valores de Referencia
| Indicador | Normal | Elevado |
|-----------|--------|---------|
| IMC | 18.5-25 | >30 |
| Presión | <100 | >110 |
| Glucosa | 70-100 | >126 |
| Colesterol | <200 | >240 |
| LDL | <100 | >160 |
| HDL | >40 | <40 (bajo) |
| Triglicéridos | <150 | >200 |
""")

# Crear dos columnas
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Datos del Paciente")
    st.markdown("*Introduce los valores en unidades médicas estándar*")
    
    # Crear inputs para cada característica
    patient_data = {}
    
    # Información del paciente básica
    st.subheader("👤 Información Básica")
    basic_col1, basic_col2, basic_col3 = st.columns(3)
    
    with basic_col1:
        patient_data["age"] = st.number_input(
            "🎂 Edad (años)",
            min_value=20,
            max_value=80,
            value=50,
            step=1,
            help="Edad del paciente en años"
        )
    
    with basic_col2:
        sex_option = st.selectbox(
            "⚧ Sexo",
            options=["Femenino", "Masculino"],
            index=0,
            help="Sexo biológico del paciente"
        )
        patient_data["sex"] = 1 if sex_option == "Masculino" else 0
    
    with basic_col3:
        patient_data["bmi"] = st.number_input(
            "⚖️ IMC (kg/m²)",
            min_value=18.0,
            max_value=45.0,
            value=26.0,
            step=0.1,
            format="%.1f",
            help="Índice de Masa Corporal = Peso(kg) / Altura(m)²"
        )
    
    # Signos vitales
    st.subheader("💓 Signos Vitales")
    vital_col1, vital_col2 = st.columns(2)
    
    with vital_col1:
        patient_data["bp"] = st.number_input(
            "🩺 Presión Arterial Media (mmHg)",
            min_value=60,
            max_value=130,
            value=95,
            step=1,
            help="Presión arterial media en reposo"
        )
    
    with vital_col2:
        patient_data["s6"] = st.number_input(
            "🍬 Glucosa en Ayunas (mg/dL)",
            min_value=70,
            max_value=200,
            value=100,
            step=1,
            help="Nivel de glucosa en sangre después de 8h de ayuno"
        )
    
    # Perfil lipídico
    st.subheader("🧪 Perfil Lipídico (Colesterol)")
    lipid_col1, lipid_col2 = st.columns(2)
    
    with lipid_col1:
        patient_data["s1"] = st.number_input(
            "🔴 Colesterol Total (mg/dL)",
            min_value=100,
            max_value=350,
            value=200,
            step=5,
            help="Nivel total de colesterol en sangre (deseable: <200)"
        )
        
        patient_data["s2"] = st.number_input(
            "⚠️ Colesterol LDL (mg/dL)",
            min_value=50,
            max_value=250,
            value=115,
            step=5,
            help="Colesterol 'malo' (óptimo: <100)"
        )
        
        patient_data["s5"] = st.number_input(
            "📊 Triglicéridos (mg/dL)",
            min_value=50,
            max_value=400,
            value=150,
            step=10,
            help="Nivel de triglicéridos (normal: <150)"
        )
    
    with lipid_col2:
        patient_data["s3"] = st.number_input(
            "✅ Colesterol HDL (mg/dL)",
            min_value=20,
            max_value=100,
            value=50,
            step=5,
            help="Colesterol 'bueno' (deseable: >40 hombres, >50 mujeres)"
        )
        
        patient_data["s4"] = st.number_input(
            "📈 Ratio TC/HDL",
            min_value=1.5,
            max_value=8.0,
            value=4.0,
            step=0.1,
            format="%.1f",
            help="Proporción Colesterol Total / HDL (óptimo: <4)"
        )

with col2:
    st.header("🎯 Predicción")
    
    # Botón de predicción
    if st.button("🔮 Obtener Predicción", type="primary", use_container_width=True):
        with st.spinner("Consultando al modelo..."):
            result = make_prediction(patient_data)
        
        if result["success"]:
            prediction = result["prediction"]
            risk_level, risk_type = get_risk_level(prediction)
            
            # Mostrar resultado
            st.metric(
                label="Predicción de Progresión",
                value=f"{prediction:.1f}",
                delta=risk_level
            )
            
            # Barra de progreso visual
            progress_value = min(prediction / 300, 1.0)  # Normalizar a 0-1
            st.progress(progress_value)
            
            # Interpretación
            st.markdown("### 📊 Interpretación")
            if prediction < 100:
                st.success("""
                **Nivel de Riesgo: BAJO**
                
                El paciente muestra indicadores favorables. 
                Se recomienda mantener hábitos saludables y controles regulares.
                """)
            elif prediction < 200:
                st.warning("""
                **Nivel de Riesgo: MODERADO**
                
                El paciente presenta algunos factores de riesgo.
                Se recomienda seguimiento médico más frecuente y 
                ajustes en estilo de vida.
                """)
            else:
                st.error("""
                **Nivel de Riesgo: ALTO**
                
                El paciente muestra indicadores preocupantes.
                Se recomienda atención médica prioritaria y 
                posible intervención terapéutica.
                """)
            
            # Mostrar datos normalizados (para debug/educación)
            with st.expander("🔍 Ver datos normalizados (enviados al modelo)"):
                norm_df = pd.DataFrame([result["normalized_data"]]).T
                norm_df.columns = ["Valor Normalizado"]
                norm_df.index.name = "Característica"
                st.dataframe(norm_df, use_container_width=True)
        else:
            st.error(result["error"])
    
    # Mostrar datos actuales
    st.markdown("---")
    st.markdown("### 📋 Resumen del Paciente")
    
    # Crear un resumen más legible
    summary_data = {
        "Edad": f"{patient_data['age']} años",
        "Sexo": "Masculino" if patient_data['sex'] == 1 else "Femenino",
        "IMC": f"{patient_data['bmi']:.1f} kg/m²",
        "Presión Arterial": f"{patient_data['bp']} mmHg",
        "Glucosa": f"{patient_data['s6']} mg/dL",
        "Colesterol Total": f"{patient_data['s1']} mg/dL",
        "LDL": f"{patient_data['s2']} mg/dL",
        "HDL": f"{patient_data['s3']} mg/dL",
        "Ratio TC/HDL": f"{patient_data['s4']:.1f}",
        "Triglicéridos": f"{patient_data['s5']} mg/dL"
    }
    
    for key, value in summary_data.items():
        st.text(f"{key}: {value}")

# Sección de ejemplos predefinidos
st.markdown("---")
st.header("📌 Ejemplos de Perfiles de Pacientes")
st.markdown("*Haz clic en un ejemplo para ver su predicción*")

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    st.subheader("🟢 Perfil Saludable")
    st.markdown("""
    - **Edad**: 35 años
    - **Sexo**: Femenino
    - **IMC**: 22.5 kg/m²
    - **Presión**: 85 mmHg
    - **Glucosa**: 85 mg/dL
    - **Col. Total**: 180 mg/dL
    - **LDL**: 90 mg/dL
    - **HDL**: 65 mg/dL
    - **Ratio**: 2.8
    - **Triglicéridos**: 100 mg/dL
    """)
    low_risk = {
        "age": 35, "sex": 0, "bmi": 22.5, "bp": 85,
        "s1": 180, "s2": 90, "s3": 65, "s4": 2.8, "s5": 100, "s6": 85
    }
    if st.button("🔮 Probar este perfil", key="low"):
        result = make_prediction(low_risk)
        if result["success"]:
            risk, _ = get_risk_level(result["prediction"])
            st.metric("Predicción", f"{result['prediction']:.1f}", risk)

with example_col2:
    st.subheader("🟡 Perfil con Factores de Riesgo")
    st.markdown("""
    - **Edad**: 55 años
    - **Sexo**: Masculino
    - **IMC**: 28.0 kg/m²
    - **Presión**: 105 mmHg
    - **Glucosa**: 115 mg/dL
    - **Col. Total**: 230 mg/dL
    - **LDL**: 140 mg/dL
    - **HDL**: 42 mg/dL
    - **Ratio**: 5.5
    - **Triglicéridos**: 180 mg/dL
    """)
    mid_risk = {
        "age": 55, "sex": 1, "bmi": 28.0, "bp": 105,
        "s1": 230, "s2": 140, "s3": 42, "s4": 5.5, "s5": 180, "s6": 115
    }
    if st.button("🔮 Probar este perfil", key="mid"):
        result = make_prediction(mid_risk)
        if result["success"]:
            risk, _ = get_risk_level(result["prediction"])
            st.metric("Predicción", f"{result['prediction']:.1f}", risk)

with example_col3:
    st.subheader("🔴 Perfil Alto Riesgo")
    st.markdown("""
    - **Edad**: 65 años
    - **Sexo**: Masculino
    - **IMC**: 35.0 kg/m²
    - **Presión**: 120 mmHg
    - **Glucosa**: 160 mg/dL
    - **Col. Total**: 280 mg/dL
    - **LDL**: 190 mg/dL
    - **HDL**: 32 mg/dL
    - **Ratio**: 7.5
    - **Triglicéridos**: 300 mg/dL
    """)
    high_risk = {
        "age": 65, "sex": 1, "bmi": 35.0, "bp": 120,
        "s1": 280, "s2": 190, "s3": 32, "s4": 7.5, "s5": 300, "s6": 160
    }
    if st.button("🔮 Probar este perfil", key="high"):
        result = make_prediction(high_risk)
        if result["success"]:
            risk, _ = get_risk_level(result["prediction"])
            st.metric("Predicción", f"{result['prediction']:.1f}", risk)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🏥 Sistema de Predicción de Diabetes | Powered by MLflow & Streamlit</p>
    <p>⚠️ Esta herramienta es solo para fines educativos y no debe usarse para diagnósticos médicos reales.</p>
</div>
""", unsafe_allow_html=True)
