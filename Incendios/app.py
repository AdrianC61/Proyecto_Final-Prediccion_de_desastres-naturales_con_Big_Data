def main():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import folium
    from folium.plugins import MarkerCluster
    from streamlit_folium import st_folium
    import joblib
    import plotly.express as px
    from datetime import datetime

    # Configuración general
    st.title("🔥 Mapa de Riesgo de Incendios en la Península Ibérica")

    # Carga de modelos
    modelo = joblib.load("Incendios\\modelo_ocurrencia_incendios.pkl")
    @st.cache_resource
    def cargar_modelo_extension():
        return joblib.load("Incendios\\modelo_prediccion_extension_incendio.pkl")

    modelo_extension = cargar_modelo_extension()

    # Carga de datos
    df_mun = pd.read_csv("Incendios\\MUNICIPIOS.csv", encoding='ISO-8859-1', sep=';', on_bad_lines='warn', decimal=',')

    # Filtrar solo península
    provincias_excluir = ['Santa Cruz de Tenerife', 'Las Palmas', 'Illes Balears', 'Ceuta', 'Melilla']
    df_peninsula = df_mun[~df_mun['PROVINCIA'].isin(provincias_excluir)].copy()

    # --- Sidebar ---
    st.sidebar.header("🎯 Filtro de predicción")

    # Rango de probabilidad
    rango_prob = st.sidebar.slider("Selecciona el rango de probabilidad de incendio", 0.0, 1.0, (0.5, 1.0), step=0.01)
    min_prob, max_prob = rango_prob

    # Mes de simulación con nombres
    meses_nombre = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                    'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    mes_nombre = st.sidebar.selectbox("🗓️ Mes de simulación", meses_nombre, index=datetime.now().month - 1)
    mes = meses_nombre.index(mes_nombre) + 1

    # Mostrar solo Catástrofes
    solo_catastrofes = st.sidebar.checkbox("💥 Mostrar solo catástrofes (>500 ha)", value=False)


    # Botón para ejecutar cálculos
    if "df_resultado" not in st.session_state:
        st.session_state.df_resultado = pd.DataFrame()


    # Función para simular clima por zona y mes
    def simular_clima_completo(row):
        zona = row["zona"]
        m = mes  # mes seleccionado por usuario

        # Definir parámetros base por zona y mes
        if zona == "norte":
            clima = {
                1:  {"temp": (6, 2),  "prcp": (3.0,), "wspd": (8, 2), "pres": (1015, 3), "hum": (85, 5)},
                6:  {"temp": (18, 3), "prcp": (2.0,), "wspd": (10, 2), "pres": (1012, 3), "hum": (70, 10)},
                7:  {"temp": (21, 3), "prcp": (1.5,), "wspd": (9, 2),  "pres": (1010, 3), "hum": (65, 10)},
                12: {"temp": (7, 2),  "prcp": (3.5,), "wspd": (7, 2),  "pres": (1016, 3), "hum": (80, 5)},
            }.get(m, {"temp": (15, 3), "prcp": (2.5,), "wspd": (8, 2), "pres": (1013, 3), "hum": (75, 10)})
        else:  # sur
            clima = {
                1:  {"temp": (10, 3), "prcp": (2.0,), "wspd": (10, 2), "pres": (1014, 3), "hum": (75, 10)},
                6:  {"temp": (24, 3), "prcp": (1.0,), "wspd": (12, 3), "pres": (1010, 3), "hum": (55, 10)},
                7:  {"temp": (28, 3), "prcp": (0.8,), "wspd": (14, 3), "pres": (1008, 3), "hum": (45, 10)},
                12: {"temp": (12, 3), "prcp": (2.5,), "wspd": (9, 2),  "pres": (1015, 3), "hum": (70, 10)},
            }.get(m, {"temp": (20, 3), "prcp": (1.5,), "wspd": (11, 3), "pres": (1012, 3), "hum": (60, 10)})

        temp = np.random.normal(*clima["temp"])
        return pd.Series({
            "temp": temp,
            "tmin": temp - np.random.uniform(3, 6),
            "tmax": temp + np.random.uniform(3, 6),
            "prcp": np.random.exponential(*clima["prcp"]),
            "wspd": np.random.normal(*clima["wspd"]),
            "pres": np.random.normal(*clima["pres"]),
            "hum": np.clip(np.random.normal(*clima["hum"]), 20, 100)  # limitar entre 20-100%
        })

    if st.sidebar.button("🔍 Calcular predicciones"):
        # Preparar datos
        df_peninsula = df_peninsula.rename(columns={"LATITUD_ETRS89": "lat", "LONGITUD_ETRS89": "lng"})
        df_peninsula["temp"] = 22
        df_peninsula["tmin"] = 16
        df_peninsula["tmax"] = 28
        df_peninsula["prcp"] = 0.5
        df_peninsula["wspd"] = 5
        df_peninsula["pres"] = 1012
        df_peninsula["hum"] = 35
        df_peninsula["mes"] = mes
        # Año actual (o fijo a 2025 si prefieres)
        anio_actual = datetime.now().year if datetime.now().year >= 2025 else 2025

        # Crear columna de fecha para el primer día del mes
        df_peninsula["fecha_simulada"] = pd.to_datetime({
            "year": anio_actual,
            "month": df_peninsula["mes"],
            "day": 1
        })
        np.random.seed(42)
        n = len(df_peninsula)

        # Clasificar por zona: norte (más frío) y sur (más cálido)
        np.random.seed(42)
        df_peninsula["zona"] = np.where(df_peninsula["lat"] >= 41.5, "norte", "sur")
            
        # Aplicar simulación
        df_peninsula[["temp", "tmin", "tmax", "prcp", "wspd", "pres", "hum"]] = df_peninsula.apply(simular_clima_completo, axis=1)

        # Predicción de ocurrencia
        features_ocurrencia = ['lat', 'lng', 'temp', 'tmin', 'tmax', 'prcp', 'wspd', 'pres', 'hum', 'mes']
        df_peninsula["probabilidad_incendio"] = modelo.predict_proba(df_peninsula[features_ocurrencia])[:, 1]
        df_peninsula["probabilidad"] = np.clip(df_peninsula["probabilidad_incendio"] * 1.8, 0, 1)
        df_peninsula["riesgo_alto"] = (df_peninsula["probabilidad"] > 0.5).astype(int)

        # Filtrar por rango
        df_filtrado = df_peninsula[
            (df_peninsula["probabilidad"] >= min_prob) &
            (df_peninsula["probabilidad"] <= max_prob)
        ].copy()

        # Clasificación catástrofe
        if not df_filtrado.empty:
            features_extension = ['temp', 'tmin', 'tmax', 'prcp', 'wspd', 'pres', 'hum']
            df_filtrado["superficie_predicha"] = modelo_extension.predict(df_filtrado[features_extension])
            df_filtrado["es_catastrofe"] = (df_filtrado["superficie_predicha"] >= 500).astype(int)
        else:
            st.warning(f"No se encontraron incendios con probabilidad entre {min_prob:.2f} y {max_prob:.2f} en el mes de {mes_nombre}.")
        
        # Guardar resultado
        st.session_state.df_resultado = df_filtrado

    # --- Mostrar resultados si existen ---
    df_resultado = st.session_state.df_resultado

    if not df_resultado.empty:

        # --- Estadísticas ---
        st.subheader("📈 Estadísticas generales del filtro aplicado")

        municipios_filtrados = len(df_resultado)
        riesgo_alto = (df_resultado["riesgo_alto"] == 1).sum()
        catastrofes = (df_resultado["es_catastrofe"] == 1).sum()
        total_quemado = (df_resultado["superficie_predicha"]).round(0).astype(int).sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Municipios afectados", municipios_filtrados)
        col2.metric("Con riesgo alto (> 50%)", riesgo_alto)
        col3.metric("Posibles catástrofes", catastrofes)
        col4.metric("Total de posibles ha quemadas", total_quemado)
        
        # --- Mapa ---
        st.subheader("🗺️ Mapa filtrado por riesgo")

        m = folium.Map(location=[40.0, -3.5], zoom_start=6, tiles='CartoDB positron')
        cluster = MarkerCluster().add_to(m)

        if solo_catastrofes:
                df_resultado = df_resultado[df_resultado["es_catastrofe"] == 1]
        
        for _, row in df_resultado.iterrows():
            color = "red" if row.get("es_catastrofe", 0) else "green"
            popup = f"""
            <b>Municipio:</b> {row['NOMBRE_ACTUAL']}<br>
            <b>Provincia:</b> {row['PROVINCIA']}<br>
            <b>Fecha simulada:</b> {row['fecha_simulada'].strftime('%Y-%m-%d')}<br>
            <b>Probabilidad:</b> {row['probabilidad']:.2f}%<br>
            """
            if "superficie_predicha" in row and pd.notna(row["superficie_predicha"]):
                popup += f"<b>Superficie estimada:</b> {row['superficie_predicha']:.0f} ha<br>"
            if "es_catastrofe" in row:
                popup += f"<b>¿Catástrofe?</b> {'Sí' if row['es_catastrofe'] == 1 else 'No'}<br>"
            if "magnitud_incendio" in row and pd.notna(row["magnitud_incendio"]):
                popup += f"<b>Magnitud:</b> {row['magnitud_incendio']:.2f}"

            folium.CircleMarker(
                location=[row["lat"], row["lng"]],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=popup
            ).add_to(cluster)

        st_folium(m, width=1000, height=600)
        st.markdown("<div style='margin-top:-40px;'></div>", unsafe_allow_html=True)

        # --- Tabla ---
        with st.expander("📋 Ver tabla filtrada"):
            # Crear una copia para mostrar con nombres adecuados y valores redondeados/formateados
            tabla_mostrar = df_resultado.copy()

            # Renombrar columnas
            tabla_mostrar = tabla_mostrar.rename(columns={
                "PROVINCIA": "provincia",
                "NOMBRE_ACTUAL": "municipio",
                "probabilidad": "probabilidad",
                "es_catastrofe": "catástrofe",
                "superficie_predicha": "superficie (ha)"
            })

            # Formatear columnas
            tabla_mostrar["probabilidad"] = tabla_mostrar["probabilidad"].round(2)
            if "catástrofe" in tabla_mostrar.columns:
                tabla_mostrar["catástrofe"] = tabla_mostrar["catástrofe"].map({0: "No", 1: "Sí"})
            if "superficie (ha)" in tabla_mostrar.columns:
                tabla_mostrar["superficie (ha)"] = tabla_mostrar["superficie (ha)"].round(0).astype(int)

            # Selección y orden final de columnas a mostrar
            columnas = ["provincia", "municipio", "probabilidad"]
            if "catástrofe" in tabla_mostrar.columns:
                columnas.append("catástrofe")
            if "superficie (ha)" in tabla_mostrar.columns:
                columnas.append("superficie (ha)")

            st.dataframe(tabla_mostrar[columnas].sort_values(by="probabilidad", ascending=False).reset_index(drop=True))



        # --- Gráficos de resumen ---
        st.subheader("📊 Distribución de Riesgo de Incendio")

        # Riesgo alto vs. no alto
        conteo_riesgo = df_resultado["riesgo_alto"].value_counts().rename({0: "Riesgo Bajo/Moderado", 1: "Riesgo Alto"})
        fig_riesgo = px.pie(
            names=conteo_riesgo.index,
            values=conteo_riesgo.values,
            title="Proporción de municipios según nivel de riesgo",
            color_discrete_sequence=px.colors.sequential.Reds
        )
        st.plotly_chart(fig_riesgo, use_container_width=True)

        # Si hay datos de catástrofe
        if "es_catastrofe" in df_resultado.columns and df_resultado["es_catastrofe"].notna().any():
            st.subheader("🚨 Proporción de Catástrofes Detectadas")

            conteo_catastrofe = df_resultado["es_catastrofe"].value_counts().rename({0: "No Catástrofe", 1: "Catástrofe"})
            fig_cat = px.pie(
                names=conteo_catastrofe.index,
                values=conteo_catastrofe.values,
                title="Clasificación de municipios con riesgo como catástrofe",
                color_discrete_sequence=px.colors.sequential.Oranges
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        st.subheader("📉 Histograma de probabilidades de incendio")
        fig_hist = px.histogram(df_resultado, x="probabilidad", nbins=20)
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.info("Haz clic en el botón de la izquierda para calcular predicciones.")
