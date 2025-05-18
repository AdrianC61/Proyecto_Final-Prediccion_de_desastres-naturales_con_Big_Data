
# Sistema Predictivo de Desastres Naturales

Este repositorio contiene el trabajo final del proyecto que desarrolla un sistema de predicción de eventos naturales como **incendios forestales**, **terremotos** y **tsunamis** usando modelos de Machine Learning.

---

## Estructura del repositorio

```
Repositorio
 ┣ Incendios
 ┃ ┣ incendios_completo.csv
 ┃ ┣ modelo_prediccion_magn_incendio.pkl
 ┃ ┗ Modelo_Incendios.ipynb
 ┣ Seísmos
 ┃ ┣ Modelo_Terremotos.ipynb
 ┃ ┣ cuadrante_nombres.json
 ┃ ┣ df_terremotos.csv
 ┃ ┣ modelo_magnitud.pkl
 ┃ ┗ modelo_tsunami.pkl
```

---

## Formato de los archivos CSV y JSON

### `incendios_completo.csv`

Contiene los datos históricos de incendios forestales, combinando características meteorológicas y operativas:

| Columna            | Descripción                        |
| ------------------ | ---------------------------------- |
| superficie         | Hectáreas afectadas                |
| fecha              | Fecha del incendio (YYYY-MM-DD)    |
| lat / lng          | Coordenadas del incendio           |
| idcomunidad        | ID de comunidad autónoma           |
| idprovincia        | ID de la provincia                 |
| idmunicipio        | ID del municipio                   |
| municipio          | Nombre del municipio               |
| causa              | Origen del incendio                |
| temp / tmin / tmax | Temperatura media, mínima y máxima |
| prcp               | Precipitaciones (mm)               |
| wspd               | Velocidad del viento (km/h)        |
| pres               | Presión atmosférica (hPa)          |
| hum                | Humedad relativa (%)               |
| gastos / perdidas  | Costes estimados del incendio      |
| codigo\_ine        | Código INE del municipio           |
---

### `df_terremotos.csv`

Base de datos de eventos sísmicos desde 1900, con columnas para análisis espacial y temporal:

| Columna               | Descripción                                   |
| --------------------- | --------------------------------------------- |
| Lugar                 | Localización del epicentro                    |
| Magnitud              | Magnitud del terremoto (escala de momento)    |
| Profundidad\_km       | Profundidad en kilómetros                     |
| Latitud / Longitud    | Coordenadas geográficas                       |
| Alerta                | Nivel de alerta (0 a 3)                       |
| Tsunami               | Indicador binario de tsunami (0 = no, 1 = sí) |
| Anio / Mes / Dia      | Fecha del evento desglosada                   |
| Hora                  | Hora del evento (0–23)                        |
| lat\_grid / lon\_grid | Cuadrante 1x1 grado de agrupación geográfica  |
| cuadrante             | ID del cuadrante en formato 'lat,lon'         |
| Fecha                 | Fecha completa del evento (YYYY-MM-DD)        |

---

### `cuadrante_nombres.json`

Diccionario que asocia coordenadas (cuadrantes geográficos) con nombres de zonas sismológicas aproximadas:

```json
{
  "37,-7": "Andalucía",
  "36,-3": "Mar de Alborán",
  "41,2":  "Barcelona",
  ...
}
```

Cada clave representa un cuadrante 1x1 en formato `"latitud,longitud"` y su valor es el nombre interpretativo de la zona.

---

## Requisitos

- Python 3.11+
- Bibliotecas:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `imbalanced-learn`
  - `joblib`, `statsmodels`

---

## Contacto

Proyecto realizado por Adrián Casto Ruiz (2025).
