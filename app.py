import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

st.title("¿Qué rating tendrá?")

# Crear el formulario
with st.form(key="form_pelicula"):
    nombre = st.text_input("Nombre")
    anio = st.number_input("Año", min_value=1900)
    duracion = st.number_input("Duración (en minutos)", min_value=0)
    generos = st.text_input("Géneros (separado con comas)")
    descripcion = st.text_input("Descripción")
    actores = st.text_input("Quiénes actúan (separado con comas)")
    
    # Botón para enviar
    submit_button = st.form_submit_button(label="Enviar")

# Procesar los datos si se envía el formulario
if submit_button:
    st.success("Gracias por tu pregunta!")

    stars = [a.strip() for a in actores.split(',')]
    stars_str = ", ".join(stars)

    new_data = pd.DataFrame({
    'title': [nombre],
    'year': [anio],
    'duration': [duracion],
    'genre': [generos],
    'description': [descripcion],
    'stars': [stars_str]
    })

    model = load_model('knn_tuned_model')
    predictions = predict_model(model, data=new_data)

    st.write(f"La película {nombre} tiene el siguiente rating {predictions.iat[0,6]}")
