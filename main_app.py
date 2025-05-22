import streamlit as st
import threading
import webbrowser
import time

st.set_page_config(page_title="Predicción de Desastres", layout="wide")
st.title("🧭 Sistema de Predicción de Desastres Naturales")

# Menú simple con selectbox nativo
opcion = st.selectbox(
    "Selecciona la aplicación que deseas visualizar:",
    ["🌍 Terremotos y Tsunamis", "🔥 Incendios Forestales"]
)

# Llamar al módulo correspondiente
if opcion == "🌍 Terremotos y Tsunamis":
    import Seismos.app as seismos_app
    seismos_app.main()

elif opcion == "🔥 Incendios Forestales":
    import Incendios.app as incendios_app
    incendios_app.main()
