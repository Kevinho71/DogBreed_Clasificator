import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# =====================
# CARGA DE CONFIGURACIÓN Y MODELOS
# =====================

# Carga variables de entorno
load_dotenv()
API_GEMINI = os.getenv("API_GEMINI")

# Directorio base del proyecto
BASE_DIR = Path(__file__).parent.parent

# Rutas a artefactos (pueden sobrescribirse en .env)
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "modelo_entrenado.h5"))
CLASS_INDICES_PATH = Path(os.getenv("CLASS_INDICES_PATH", BASE_DIR / "models" / "class_indices.pkl"))
TRANSLATIONS_PATH = Path(os.getenv("TRANSLATIONS_PATH", BASE_DIR / "models" / "traducciones.pkl"))

# Función para consultar Gemini
API_KEY = API_GEMINI

def consultar_gemini(raza: str) -> str:
    genai.configure(api_key=API_KEY)
    model_ai = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    Dame una descripción detallada de la raza de perro '{raza}'. Quiero que el texto tenga los siguientes apartados:
    - Tamaño:
    - Color:
    - Personalidad:
    - Curiosidades:
    - Inteligencia:
    - Precauciones:
    - Región de origen:
    Sé claro, haz un listado para cada caracteristica y explicala en 1 sola linea y en español.
    Solo genera el nombre del perro y el listado no anadas absolutamente nada mas como "¡Claro! Aquí tienes una descripción detallada del perro, organizada en los apartados que solicitaste:"
    """
    response = model_ai.generate_content(prompt)
    return response.text.strip()

# Carga del modelo TensorFlow
model = tf.keras.models.load_model(str(MODEL_PATH))

# Carga de índices de clase
with open(CLASS_INDICES_PATH, 'rb') as f:
    class_indices = pickle.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

# Carga de traducciones
with open(TRANSLATIONS_PATH, 'rb') as f:
    traducciones = pickle.load(f)

def traducir(nombre_en: str) -> str:
    return traducciones.get(nombre_en, nombre_en)

# =====================
# INTERFAZ STREAMLIT
# =====================

st.set_page_config(page_title="Clasificador de Razas de Perros", page_icon="🐶", layout="centered")
st.title("🐾 Clasificador de Raza de Perro")
st.caption("Sube una imagen de un perro y el modelo te dirá las 5 razas más probables.")

# Selector de archivo de imagen
img_file = st.file_uploader("📷 Subir imagen", type=["jpg", "jpeg", "png"])

if img_file:
    # Muestra imagen
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Preprocesamiento
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    preds = model.predict(img_array)[0]
    top_indices = preds.argsort()[-5:][::-1]

    # Resultados textuales
    st.subheader("🔎 Top 5 razas más probables:")
    for i in top_indices:
        en = idx_to_class[i]
        es = traducir(en)
        st.markdown(f"- **{es}**: `{preds[i]*100:.2f}%`")

    # Gráfico de barras
    st.subheader("📊 Visualización de resultados")
    labels = [traducir(idx_to_class[i]) for i in top_indices]
    scores = [preds[i]*100 for i in top_indices]
    df = pd.DataFrame({"Raza": labels, "Probabilidad (%)": scores})
    st.bar_chart(df.set_index("Raza"))

    # Detalles de la raza principal
    raza_principal = traducir(idx_to_class[top_indices[0]])
    st.subheader(f"📘 Características de la raza: {raza_principal}")
    with st.spinner("Consultando a Gemini..."):
        descripcion = consultar_gemini(raza_principal)
    st.markdown(descripcion)
