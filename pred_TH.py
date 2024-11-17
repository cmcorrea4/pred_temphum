import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import altair as alt
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(page_title="Predicción de Temperatura y Humedad - LSTM", layout="wide")

# Definición del modelo LSTM para múltiples variables
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=2, output_size=2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Función para normalizar datos
def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1  # Evitar división por cero
    return (data - mean) / std, mean, std

# Función para desnormalizar datos
def denormalize_data(data, mean, std):
    return data * std + mean

# Función para crear secuencias
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:(i + seq_length)]
        sequences.append(seq)
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

# Función para cargar datos
def load_data(file):
    try:
        # Cargar el CSV con el formato específico
        df = pd.read_csv(file, skiprows=1)  # Saltar la primera fila que contiene "sep=,"
        
        # Renombrar columnas para simplicidad
        df.columns = ['Time', 'temperatura', 'humedad']
        
        # Convertir columna de tiempo
        df['Time'] = pd.to_datetime(df['Time'])
        
        # Eliminar filas con valores faltantes
        df = df.dropna()
        
        return df.sort_values('Time')
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None

# Función para realizar predicciones
def predict_future(model, last_sequence, n_steps, mean, std):
    model.eval()
    predictions = []
    current_sequence = last_sequence.clone()
    
    with torch.no_grad():
        for _ in range(n_steps):
            # Preparar input
            x = current_sequence.view(1, -1, 2)
            # Predecir siguiente valor
            output = model(x)
            predictions.append(output.numpy()[0])
            # Actualizar secuencia
            current_sequence = torch.cat((current_sequence[1:], output.view(1, -1)), 0)
    
    # Desnormalizar predicciones
    predictions = denormalize_data(np.array(predictions), mean, std)
    return predictions

# Título de la aplicación
st.title("🌡️ Predicción de Temperatura y Humedad con LSTM")

# Configuración en la barra lateral
st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=['csv'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.sidebar.header("Parámetros del Modelo")
        seq_length = st.sidebar.slider("Longitud de secuencia (minutos)", 
                                     min_value=5, max_value=60, value=30)
        prediction_steps = st.sidebar.slider("Minutos a predecir", 
                                           min_value=5, max_value=120, value=30)
        hidden_size = st.sidebar.slider("Tamaño capa oculta", 
                                      min_value=10, max_value=100, value=50)
        epochs = st.sidebar.slider("Épocas de entrenamiento", 
                                 min_value=10, max_value=200, value=50)
        
        # Botón de entrenamiento
        train_button = st.sidebar.button("Entrenar Modelo")
        
        # Mostrar datos originales
        st.subheader("Vista previa de los datos")
        st.write(df.head())
        
        if train_button:
            with st.spinner('Entrenando el modelo...'):
                try:
                    # Preparar datos
                    data = df[['temperatura', 'humedad']].values
                    data_normalized, mean, std = normalize_data(data)
                    
                    # Crear secuencias
                    X, y = create_sequences(data_normalized, seq_length)
                    X = torch.FloatTensor(X)
                    y = torch.FloatTensor(y)
                    
                    # Crear y entrenar modelo
                    model = LSTMPredictor()
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters())
                    
                    # Entrenamiento
                    progress_bar = st.progress(0)
                    train_losses = []
                    
                    for epoch in range(epochs):
                        model.train()
                        optimizer.zero_grad()
                        
                        outputs = model(X)
                        loss = criterion(outputs, y)
                        
                        loss.backward()
                        optimizer.step()
                        
                        train_losses.append(loss.item())
                        progress_bar.progress((epoch + 1) / epochs)
                    
                    # Realizar predicciones
                    last_sequence = torch.FloatTensor(data_normalized[-seq_length:])
                    predictions = predict_future(model, last_sequence, prediction_steps, mean, std)
                    
                    # Crear fechas futuras
                    last_date = df['Time'].iloc[-1]
                    future_dates = [last_date + timedelta(minutes=i+1) for i in range(len(predictions))]
                    
                    # Crear DataFrames de predicciones
                    predictions_df = pd.DataFrame({
                        'Time': future_dates,
                        'temperatura': predictions[:, 0],
                        'humedad': predictions[:, 1],
                        'Tipo': 'Predicción'
                    })
                    
                    # Preparar datos históricos
                    historical_df = df.copy()
                    historical_df['Tipo'] = 'Histórico'
                    
                    # Visualización de temperatura
                    st.header("📈 Predicción de Temperatura")
                    temp_data = pd.concat([
                        historical_df[['Time', 'temperatura', 'Tipo']],
                        predictions_df[['Time', 'temperatura', 'Tipo']]
                    ])
                    
                    temp_chart = alt.Chart(temp_data).mark_line().encode(
                        x=alt.X('Time:T', title='Tiempo'),
                        y=alt.Y('temperatura:Q', title='Temperatura (°C)'),
                        color=alt.Color('Tipo:N', scale=alt.Scale(domain=['Histórico', 'Predicción'],
                                                                range=['#1f77b4', '#ff7f0e']))
                    ).properties(width=800, height=300).interactive()
                    
                    st.altair_chart(temp_chart, use_container_width=True)
                    
                    # Visualización de humedad
                    st.header("📈 Predicción de Humedad")
                    hum_data = pd.concat([
                        historical_df[['Time', 'humedad', 'Tipo']],
                        predictions_df[['Time', 'humedad', 'Tipo']]
                    ])
                    
                    hum_chart = alt.Chart(hum_data).mark_line().encode(
                        x=alt.X('Time:T', title='Tiempo'),
                        y=alt.Y('humedad:Q', title='Humedad (%)'),
                        color=alt.Color('Tipo:N', scale=alt.Scale(domain=['Histórico', 'Predicción'],
                                                                range=['#1f77b4', '#ff7f0e']))
                    ).properties(width=800, height=300).interactive()
                    
                    st.altair_chart(hum_chart, use_container_width=True)
                    
                    # Mostrar tabla de predicciones
                    st.header("📋 Tabla de Predicciones")
                    prediction_table = pd.DataFrame({
                        'Fecha y Hora': [d.strftime('%Y-%m-%d %H:%M:%S') for d in future_dates],
                        'Temperatura (°C)': predictions[:, 0].round(2),
                        'Humedad (%)': predictions[:, 1].round(2)
                    })
                    st.dataframe(prediction_table, height=400)
                    
                    # Botón de descarga
                    st.download_button(
                        label="Descargar Predicciones CSV",
                        data=prediction_table.to_csv(index=False),
                        file_name="predicciones_temp_hum.csv",
                        mime="text/csv",
                    )
                    
                except Exception as e:
                    st.error(f"Error durante el entrenamiento: {str(e)}")
                    st.info("Intenta ajustar los parámetros del modelo o verificar los datos.")

else:
    st.info("👆 Por favor, carga un archivo CSV para comenzar el análisis.")
    st.markdown("""
    El archivo CSV debe contener las siguientes columnas:
    - Time: Fecha y hora de la medición
    - temperatura: Temperatura en °C
    - humedad: Humedad relativa en %
    """)

# Información de uso
st.sidebar.markdown("""
---
### Información de Uso
- Ajusta la longitud de secuencia según el patrón temporal
- Define cuántos minutos hacia el futuro predecir
- Modifica el tamaño de la capa oculta para ajustar la complejidad del modelo
- Aumenta las épocas de entrenamiento para mejor precisión
""")
