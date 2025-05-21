# Predicción de Desastres Naturales en España

Este proyecto permite simular y visualizar predicciones de **incendios forestales** y **terremotos** en la Península Ibérica mediante modelos de machine learning y mapas interactivos.

---

## Estructura del Proyecto

```

PrediccionDesastres/
│
├── main\_app.py                     # Menú principal para elegir entre seísmos e incendios
├── requirements.txt
├── README.md
│
├── Incendios/
│   ├── app.py
│   ├── modelo\_ocurrencia\_incendios.pkl
│   ├── modelo\_prediccion\_extension\_incendio.pkl
│   └── MUNICIPIOS.csv
│
├── Seismos/
│   ├── app.py
│   ├── modelo\_magnitud.pkl
│   ├── modelo\_tsunami.pkl
│   ├── cuadrante\_nombres.json
│   ├── df\_terremotos.csv
│   └── modelos\_frecuencia/       # Contiene un .pkl por cada cuadrante (ej: modelo\_frecuencia\_36\_-5.pkl)

````

---

## Cómo ejecutar la aplicación

1. Asegúrate de tener Python 3.8 o superior.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
````

3. Ejecuta la app principal:

   ```bash
   streamlit run main_app.py
   ```

---

## Funcionalidades

### Incendios

* Predicción de ocurrencia de incendios por municipio.
* Estimación de superficie afectada.
* Clasificación de incendios catastróficos (≥ 500 ha).
* Mapa interactivo y gráficos resumen.

### Terremotos

* Predicción mensual de número de terremotos por zona sísmica.
* Estimación de magnitud promedio y detección de tsunamis.
* Visualización en mapa y gráficos descargables.
* Los modelos de predicción por cuadrante están en `Seismos/modelos_frecuencia/`.

---

## Requisitos

* Python ≥ 3.8
* Streamlit
* Pandas
* NumPy
* scikit-learn
* Folium
* Plotly
* Statsmodels
* joblib

---

## Notas importantes

* Los modelos `.pkl` han sido previamente entrenados.
* Las predicciones son simulaciones aproximadas basadas en datos históricos.
* Todos los datos geográficos se limitan a la Península Ibérica.

---

## Autor

Proyecto realizado por Adrián Casto Ruiz (2025).
