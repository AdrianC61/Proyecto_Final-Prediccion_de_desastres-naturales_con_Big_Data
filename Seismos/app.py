def main():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    import pydeck as pdk
    from pandas.tseries.offsets import MonthEnd
    import joblib
    import json
    from io import BytesIO
    from statsmodels.formula.api import glm
    import statsmodels.api as sm

    if "mostrar_resultados" not in st.session_state:
        st.session_state["mostrar_resultados"] = False
        st.session_state["zona_anterior"] = None
        st.session_state["meses_anteriores"] = None


    # -----------------------------
    # 1. Cargar y preparar datos
    # -----------------------------

    st.title("🌍 Predicción de Desastres Naturales")
    st.markdown("Este sistema permite estimar la frecuencia, magnitud y posibilidad de tsunamis en zonas sísmicas específicas.")

    with open('Seismos\\cuadrante_nombres.json', 'r', encoding="utf-8") as archivo:
        zonas_sismicas = json.load(archivo)

    df = pd.read_csv("Seismos\\df_terremotos.csv")
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["lat_grid"] = df["Latitud"].apply(lambda x: int(np.floor(x)))
    df["lon_grid"] = df["Longitud"].apply(lambda x: int(np.floor(x)))
    df["cuadrante"] = df["lat_grid"].astype(str) + "," + df["lon_grid"].astype(str)

    # Selección de cuadrante
    st.sidebar.header("📍 Selección de Zona Sísmica")
    zonas = list(zonas_sismicas.values())
    zona_elegida = st.sidebar.selectbox("Elige una zona sísmica", zonas)
    coordenadas = [k for k, v in zonas_sismicas.items() if v == zona_elegida][0]

    # Parámetros de predicción
    st.sidebar.header("🔮 Parámetros de Predicción")
    n_meses = st.sidebar.slider("¿Cuántos meses deseas predecir?", 1, 24, 6)

    # Parámetros actuales
    zona_actual = coordenadas
    meses_actuales = n_meses

    # Botón para iniciar el modelo
    if st.sidebar.button("🔄 Generar predicciones"):
        st.session_state["mostrar_resultados"] = True
        st.session_state["zona_anterior"] = zona_actual
        st.session_state["meses_anteriores"] = meses_actuales
        if (
            st.session_state["mostrar_resultados"]
            and st.session_state["zona_anterior"] == zona_actual
            and st.session_state["meses_anteriores"] == meses_actuales
        ):
            cuadrante_objetivo = zona_actual
            df_cuad = df[df["cuadrante"] == cuadrante_objetivo]
            
            # Si no hay datos:
            if df_cuad.empty:
                st.warning(f"No hay registros de terremotos en el cuadrante {zona_actual}.")
                st.stop()

            # Preparar serie temporal
            df_cuad.set_index("Fecha", inplace=True)
            serie = df_cuad.resample("M").size().rename("Frecuencia").reset_index()
            serie["t"] = range(len(serie))

            # -----------------------------
            # 2. Simulación de eventos con promedio de múltiples repeticiones
            # -----------------------------

            lat, lon = zona_actual.split(",")
            modelo_path = f"Seismos\\modelos_frecuencia\\modelo_frecuencia_{lat}_{lon}.pkl"

            try:
                modelo_frecuencia = joblib.load(modelo_path)
            except FileNotFoundError:
                st.error(f"❌ No se encontró el modelo para el cuadrante {zona_actual}. Asegúrate de que exista: {modelo_path}")
                st.stop()

            # Preparar predicción mensual
            hoy = datetime.now()
            anio_pred = hoy.year if hoy.year >= 2025 else 2025
            mes_pred = hoy.month + 1 if hoy.month < 12 else 1
            anio_pred = anio_pred if hoy.month < 12 else anio_pred + 1
            inicio_pred = pd.Timestamp(year=anio_pred, month=mes_pred, day=1) + MonthEnd(0)

            fechas_pred = pd.date_range(start=inicio_pred + pd.DateOffset(months=1), periods=meses_actuales, freq="M")
            t_inicio = serie["t"].iloc[-1] + 1
            t_futuro = list(range(t_inicio, t_inicio + meses_actuales))
            meses_futuros = [d.month for d in fechas_pred]

            df_future = pd.DataFrame({"t": t_futuro, "Mes": meses_futuros})
            pred_df = pd.DataFrame({
                "Fecha": fechas_pred,
                "Frecuencia estimada": modelo_frecuencia.predict(df_future)
            })

            future_t = list(range(serie["t"].iloc[-1] + 1, serie["t"].iloc[-1] + 1 + n_meses))
            future_months = [(serie["Fecha"].iloc[-1] + pd.DateOffset(months=i)).month for i in range(1, n_meses + 1)]

            df_future = pd.DataFrame({"t": future_t, "Mes": future_months})
            frecuencia_pred = modelo_frecuencia.predict(df_future)

            np.random.seed(42)
            n_simulaciones = 10

            lam = pred_df["Frecuencia estimada"].values  # aseguramos array numérico

            # Realizar múltiples simulaciones y calcular el promedio
            simulaciones = np.array([np.random.poisson(lam) for _ in range(n_simulaciones)])
            eventos_promedio = np.round(simulaciones.mean(axis=0)).astype(int)

            # Asignar al DataFrame
            pred_df["Eventos_simulados"] = eventos_promedio

            # Asegurar al menos un evento si hay historial en el cuadrante
            if pred_df["Eventos_simulados"].sum() == 0 and not df_cuad.empty:
                pred_df.loc[pred_df.index[0], "Eventos_simulados"] = 1



            # -----------------------------
            # 4. Predecir magnitudes
            # -----------------------------
            modelo_magnitud = joblib.load("Seismos\\modelo_magnitud.pkl")
            np.random.seed(42)
            entradas_mag = []
            for idx, row in pred_df.iterrows():
                for _ in range(int(row["Eventos_simulados"])):
                    entradas_mag.append({
                        "Latitud": float(zona_actual.split(",")[0]) + np.random.uniform(0, 1),
                        "Longitud": float(zona_actual.split(",")[1]) + np.random.uniform(0, 1),
                        "Profundidad_km": np.random.uniform(1, 40),
                        "Anio": row["Fecha"].year,
                        "Mes": row["Fecha"].month,
                        "Dia": row["Fecha"].day,
                        "Hora": np.random.randint(0, 24),
                        "Alerta": 1,
                        "Fecha": row["Fecha"]
                    })

            X_mag = pd.DataFrame(entradas_mag)
            if X_mag.empty:
                st.warning("⚠️ No se han simulado eventos sísmicos para este cuadrante. No hay predicciones disponibles.")
                st.stop()

            X_mag_features = X_mag[['Latitud', 'Longitud', 'Profundidad_km', 'Anio', 'Mes', 'Dia', 'Hora', 'Alerta']]
            X_mag["Magnitud"] = modelo_magnitud.predict(X_mag_features)

            magnitudes_por_mes = X_mag.groupby("Fecha")["Magnitud"].mean()
            pred_df["Magnitud_promedio_estimada"] = pred_df["Fecha"].map(magnitudes_por_mes)
            pred_df["Magnitud_promedio_estimada"] = pred_df["Magnitud_promedio_estimada"].round(1)

            # -----------------------------
            # 5. Predecir tsunamis
            # -----------------------------
            modelo_tsunami = joblib.load("Seismos\\modelo_tsunami.pkl")
            X_tsunami = X_mag[["Latitud", "Longitud", "Profundidad_km", "Anio", "Mes", "Dia", "Hora", "Alerta"]].fillna(0)
            X_mag["Tsunami_Previsto"] = modelo_tsunami.predict(X_tsunami)
            tsunamis_por_mes = X_mag.groupby("Fecha")["Tsunami_Previsto"].sum()
            pred_df["Tsunamis_estimados"] = pred_df["Fecha"].map(tsunamis_por_mes).fillna(0).astype(int)

            # -----------------------------
            # 6. Mostrar resultados
            # -----------------------------
            if st.session_state["mostrar_resultados"]:
                st.subheader(f"📈 Predicción para: {zona_elegida}")
                st.dataframe(pred_df[["Fecha", "Eventos_simulados", "Magnitud_promedio_estimada", "Tsunamis_estimados"]], use_container_width=True)

                # -----------------------------
                # 7. Mapa de ubicación 
                # -----------------------------
                st.subheader("🗺️ Mapa de eventos simulados")

                # Preparar datos
                coords_simuladas = X_mag[["Latitud", "Longitud", "Magnitud", "Fecha"]].copy()
                coords_simuladas["Fecha"] = coords_simuladas["Fecha"].astype(str)  # Convertir Fecha a string para tooltip
                coords_simuladas = coords_simuladas.rename(columns={
                    "Latitud": "lat",
                    "Longitud": "lon",
                    "Magnitud": "magnitud",
                    "Fecha": "fecha"
                })

                # Redondear datos
                coords_simuladas["magnitud"] = coords_simuladas["magnitud"].round(1)
                coords_simuladas["lat"] = coords_simuladas["lat"].round(2)
                coords_simuladas["lon"] = coords_simuladas["lon"].round(2)
                coords_simuladas["fecha"] = coords_simuladas["fecha"].astype(str)

                # Función para asignar color por magnitud
                def color_por_magnitud(mag):
                    if mag < 3.5:
                        return [0, 255, 0, 160]     # Verde
                    elif mag < 5.0:
                        return [255, 165, 0, 160]   # Naranja
                    else:
                        return [255, 0, 0, 160]     # Rojo

                coords_simuladas["color"] = coords_simuladas["magnitud"].apply(color_por_magnitud)
                coords_simuladas["radio"] = coords_simuladas["magnitud"] * 3000  # Escala ajustable

                # Centro del mapa
                midpoint = (np.average(coords_simuladas["lat"]), np.average(coords_simuladas["lon"]))

                # Crear capa visual
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=coords_simuladas,
                    get_position='[lon, lat]',
                    get_fill_color="color",
                    get_radius="radio",
                    pickable=True
                )

                # Tooltip interactivo
                tooltip = {
                    "html": "<b>Fecha:</b> {fecha}<br/>"
                            "<b>Magnitud:</b> {magnitud}<br/>"
                            "<b>Latitud:</b> {lat}<br/>"
                            "<b>Longitud:</b> {lon}",
                    "style": {
                        "backgroundColor": "black",
                        "color": "white"
                    }
                }

                # Mostrar mapa
                st.pydeck_chart(pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state=pdk.ViewState(
                        latitude=midpoint[0],
                        longitude=midpoint[1],
                        zoom=6,
                        pitch=0,
                    ),
                    layers=[layer],
                    tooltip=tooltip
                ))


                # -----------------------------
                # 8. Graficar
                # -----------------------------
                def fig_to_bytes(fig):
                    buf = BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight')
                    buf.seek(0)
                    return buf

                st.subheader("📊 Visualización de Resultados")

                fig, ax1 = plt.subplots(figsize=(10, 5))
                ax1.plot(pred_df["Fecha"], pred_df["Eventos_simulados"], marker='o', color='blue', label="Frecuencia estimada")
                ax1.set_ylabel("Cantidad de terremotos")
                ax1.set_title(f"Frecuencia de Terremotos - {zona_elegida}")
                ax2 = ax1.twinx()
                ax2.plot(pred_df["Fecha"], pred_df["Magnitud_promedio_estimada"], marker='s', color='red', linestyle='--', label="Magnitud estimada")
                ax2.set_ylabel("Magnitud promedio")
                fig.legend(loc="upper left")
                st.pyplot(fig)

                st.download_button(
                    label="📥 Descargar gráfico de frecuencia",
                    data=fig_to_bytes(fig),
                    file_name="frecuencia_terremotos.png",
                    mime="image/png"
                )

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(pred_df["Fecha"], pred_df["Tsunamis_estimados"], marker='o', color='purple')
                ax.set_title("Tsunamis estimados por mes")
                ax.set_ylabel("Cantidad")
                st.pyplot(fig)

                st.download_button(
                    label="📥 Descargar gráfico de tsunamis",
                    data=fig_to_bytes(fig),
                    file_name="tsunamis_estimados.png",
                    mime="image/png"
                )


