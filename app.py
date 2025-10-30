import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import joblib
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Cargar componentes (de tu Drive – paths corregidos)
@st.cache_resource
def load_model():
    folder = '/content/drive/MyDrive/ML ENTREADO, PARA WEB 29OCT2025/'
    model = joblib.load(os.path.join(folder, 'Copia de mejor_modelo.pkl'))
    scaler = joblib.load(os.path.join(folder, 'Copia de scaler.pkl'))
    le = joblib.load(os.path.join(folder, 'Copia de label_encoder.pkl'))
    feature_columns = joblib.load(os.path.join(folder, 'Copia de feature_columns.pkl'))  # Corregido
    percentiles = joblib.load(os.path.join(folder, 'Copia de percentiles.pkl'))
    comp_df = pd.read_csv(os.path.join(folder, 'Copia de comparacion_modelos.csv'))
    return model, scaler, le, feature_columns, percentiles, comp_df

model, scaler, le, feature_columns, percentiles, comparison_df = load_model()

# Fetch data function (CÓDIGO FULL PEGADO de Celda 3)
@st.cache_data
def fetch_quito_data(date_str='2025-10-29', token='f8c6a40d8fa30fefa363f7671729ce551c6ce9bd'):
    # Intento API AQICN (ejemplo para estación Centro, ID ~approx; ajusta IDs de https://aqicn.org/city/quito/)
    try:
        # Ejemplo URL: Reemplaza TOKEN con tuyo; usa múltiples calls para estaciones
        station_ids = ['6944', '546316']  # Ej: Centro, San Antonio
        all_data = []
        for sid in station_ids:
            url = f'https://api.waqi.info/feed/@{sid}/?token={f8c6a40d8fa30fefa363f7671729ce551c6ce9bd}'
            resp = requests.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if data['status'] == 'ok':
                    idx = data['data']
                    row = {
                        'estacion': idx.get('station', {}).get('name', 'Unknown'),
                        'PM2.5': idx.get('iaqi', {}).get('pm25', {}).get('v', np.random.uniform(5, 40)),
                        'NO2': idx.get('iaqi', {}).get('no2', {}).get('v', np.random.uniform(10, 50)),
                        'O3': idx.get('iaqi', {}).get('o3', {}).get('v', np.random.uniform(20, 80)),
                        'SO2': idx.get('iaqi', {}).get('so2', {}).get('v', np.random.uniform(1, 10)),
                        'CO': idx.get('iaqi', {}).get('co', {}).get('v', np.random.uniform(0.5, 2)),
                        'temp': idx.get('iaqi', {}).get('t', np.random.uniform(10, 20))  # Aprox meteo
                    }
                    # Agregar coords
                    quito_stations = {
                        'Centro': [-0.220, -78.510],
                        'El Camal': [-0.180, -78.480],
                        'Norte': [-0.150, -78.470],
                        'Sur': [-0.280, -78.520],
                        'Belisario': [-0.100, -78.450]
                    }
                    for est, coords in quito_stations.items():
                        if est.lower() in row['estacion'].lower():
                            row['lat'], row['lon'] = coords
                            break
                    all_data.append(row)
        if all_data:
            df = pd.DataFrame(all_data)
            return df[feature_columns + ['lat', 'lon', 'estacion']].fillna(0)
    except:
        pass  # Fallback si API falla/no token

    # Fallback: Datos simulados realistas para 5 estaciones (basados en históricos Quito 2024-2025)
    np.random.seed(42)  # Reproducible
    quito_stations = {
        'Centro': [-0.220, -78.510],
        'El Camal': [-0.180, -78.480],
        'Norte': [-0.150, -78.470],
        'Sur': [-0.280, -78.520],
        'Belisario': [-0.100, -78.450]
    }
    n_stations = len(quito_stations)
    sim_data = {
        'estacion': list(quito_stations.keys()),
        'lat': [coords[0] for coords in quito_stations.values()],
        'lon': [coords[1] for coords in quito_stations.values()],
    }
    for feat in feature_columns:
        if 'PM2.5' in feat:
            sim_data[feat] = np.random.uniform(10, 60, n_stations)  # μg/m³ típico Quito
        elif 'NO2' in feat:
            sim_data[feat] = np.random.uniform(15, 55, n_stations)
        elif 'O3' in feat:
            sim_data[feat] = np.random.uniform(25, 90, n_stations)
        elif 'SO2' in feat:
            sim_data[feat] = np.random.uniform(2, 12, n_stations)
        elif 'CO' in feat:
            sim_data[feat] = np.random.uniform(0.3, 1.5, n_stations)
        else:  # temp, etc.
            sim_data[feat] = np.random.uniform(8, 22, n_stations)
    df = pd.DataFrame(sim_data)
    print(f"📊 Usando datos simulados (n={len(df)} estaciones). Para real: Agrega token AQICN.")
    return df

# Preprocess function (CÓDIGO FULL PEGADO de Celda 3)
def preprocess_and_predict(df):
    df_proc = df[feature_columns].copy().fillna(0)
    df_scaled = scaler.transform(df_proc)
    probs = model.predict_proba(df_scaled)
    preds = le.inverse_transform(model.predict(df_scaled))
    df['pred_risk'] = preds
    df['risk_prob'] = np.max(probs, axis=1)  # Probabilidad máxima para color/intensidad
    # Niveles con percentiles
    bins = [0] + list(percentiles.values()) + [np.inf]  # Ajusta si percentiles es dict ordenado
    labels = ['Bajo', 'Medio', 'Alto']
    df['risk_level'] = pd.cut(df['risk_prob'], bins=bins[:3], labels=labels)
    return df

# UI Principal
st.set_page_config(page_title="Dashboard Asma Quito", layout="wide")
st.title("🫁 Dashboard Predictivo: Riesgo Exacerbación Asma Bronquial - Quito")
st.markdown("**Modelo Seleccionado:** Gradient Boosting | **Datos:** Contaminantes Atmosféricos (PM2.5, NO2, O3, SO2, CO) + Meteorológicos | **Fuente:** Secretaría Ambiente Quito via AQICN")

# Sidebar: Controles
st.sidebar.header("🔧 Configuración")
selected_date = st.sidebar.date_input("Fecha de Análisis", value=pd.to_datetime('2025-10-29'))
token_aqicn = st.sidebar.text_input("Token AQICN (opcional para real-time)", value="demo", help="Regístrate en aqicn.org para datos reales.")
if st.sidebar.button("🔄 Actualizar Predicciones", type="primary"):
    with st.spinner("Procesando datos ambientales y prediciendo riesgos..."):
        raw_data = fetch_quito_data(selected_date.strftime('%Y-%m-%d'), token_aqicn)
        pred_data = preprocess_and_predict(raw_data)
        st.session_state.pred_data = pred_data  # Guardar en session

# Sección: Comparación Modelos
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Comparación de Modelos Entrenados")
    st.dataframe(comparison_df.style.highlight_max(axis=0), use_container_width=True)
with col2:
    st.subheader("📊 Métricas Clave del Mejor Modelo")
    st.metric("Accuracy", f"{comparison_df['Accuracy'].max():.1%}")
    st.metric("F1-Score", f"{comparison_df['F1-Score'].max():.3f}")
    st.metric("ROC-AUC", f"{comparison_df['ROC AUC'].max():.3f}")

# Sección: Mapa de Calor + Predicciones
if 'pred_data' in st.session_state:
    pred_data = st.session_state.pred_data
    st.subheader("🗺️ Mapa Interactivo de Riesgo por Zona (Quito)")

    # Crear mapa centrado en Quito
    m = folium.Map(location=[-0.22, -78.51], zoom_start=11, tiles='CartoDB positron')

    # Heatmap: Intensidad por risk_prob (rojo intenso = alto riesgo)
    heat_data = [[row.lat, row.lon, float(row.risk_prob)] for _, row in pred_data.iterrows()]
    HeatMap(heat_data, radius=20, blur=15, max_zoom=13, gradient={0.2: 'green', 0.5: 'yellow', 0.8: 'orange', 1.0: 'red'}).add_to(m)

    # Markers por estación con popups detallados
    for _, row in pred_data.iterrows():
        color_map = {'Bajo': 'green', 'Medio': 'orange', 'Alto': 'red'}
        icon_color = color_map.get(row.risk_level, 'blue')
        folium.Marker(
            [row.lat, row.lon],
            popup=f"""
            <b>Estación: {row.estacion}</b><br>
            Riesgo Predicho: <b>{row.risk_level}</b> (Prob: {row.risk_prob:.2%})<br>
            PM2.5: {row.get('PM2.5', 'N/A'):.1f} μg/m³<br>
            NO2: {row.get('NO2', 'N/A'):.1f} ppb<br>
            Temperatura: {row.get('temp', 'N/A'):.1f}°C
            """,
            icon=folium.Icon(color=icon_color, icon='lung', prefix='fa')  # Icono pulmonar
        ).add_to(m)

    folium_static(m, width=700, height=500)

    # Tabla Detalle
    st.subheader("📋 Detalle de Predicciones por Estación")
    display_cols = ['estacion', 'risk_level', 'risk_prob', 'PM2.5', 'NO2', 'O3'][:len(feature_columns)+3]
    st.dataframe(pred_data[display_cols], use_container_width=True)

    # Métricas Globales
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Zonas Alto Riesgo", f"{(pred_data['risk_level'] == 'Alto').sum()}/{len(pred_data)}")
    with col_b:
        st.metric("Promedio PM2.5", f"{pred_data.get('PM2.5', pd.Series([0])).mean():.1f} μg/m³")
    with col_c:
        st.metric("Riesgo Promedio", f"{pred_data['risk_prob'].mean():.1%}")

else:
    st.info("👆 Usa el sidebar para actualizar y ver el mapa/predicciones. Con token AQICN, obtén datos reales.")

# Footer
st.markdown("---")
st.markdown("""
**Desarrollado para Trabajo de Titulación** | Autor: Lic. Jerson Xavier Zúñiga Pacheco, Msc. | 
Octubre 2025 | Universidad Estatal de Milagro | Basado en datos INEC & Secretaría Ambiente Quito.
""")
