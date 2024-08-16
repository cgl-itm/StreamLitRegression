import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="Advanced Regression Visualizer", layout="wide")

# Título de la aplicación
st.title("Function-based Advanced Regression Visualizer")

# Funciones para generar datos
def generate_data(func_type, num_points=100, noise_level=0.1):
    x = 10.0*np.random.rand(num_points) - 5.0
    if func_type == 'quadratic':
        y = x**2
    elif func_type == 'sine':
        y = np.sin(x)
    elif func_type == 'exp':
        y = np.exp(x)
    else:
        raise ValueError("Unsupported function type")
    
    y += np.random.normal(0, noise_level, num_points)
    return x, y

# Modelos de regresión
models = {
    'SVM': make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2)),
    'Polynomial': make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
}

# Sidebar para controles
st.sidebar.header("Controls")
func_type = st.sidebar.selectbox("Select function type", ['quadratic', 'sine', 'exp'])
model_type = st.sidebar.selectbox("Select regression model", list(models.keys()))
noise_level = st.sidebar.slider("Noise level", 0.0, 1.0, 0.1, 0.05)
num_points = st.sidebar.slider("Number of points", 1, 50, 25, 1)

# Botón para generar datos
if st.sidebar.button("Generate Data"):
    if num_points > 0:
        x, y = generate_data(func_type, num_points=num_points, noise_level=noise_level)
        
        # Guardamos los datos en la sesión de Streamlit
        st.session_state['data'] = {'x': x, 'y': y}
        
        # Mostramos el scatter plot de los datos
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_title(f"{func_type.capitalize()} Function with Noise (n={num_points})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        st.pyplot(fig)
    else:
        st.error("Please select at least 1 point to generate data.")

# Botón para ejecutar la regresión
if st.sidebar.button("Run Regression"):
    if 'data' not in st.session_state:
        st.error("Please generate data first!")
    elif num_points == 0:
        st.error("Please generate data with at least 1 point.")
    else:
        x = st.session_state['data']['x']
        y = st.session_state['data']['y']
        
        # Preparamos los datos para el modelo
        X = x.reshape(-1, 1)
        
        # Entrenamos el modelo
        model = models[model_type]
        model.fit(X, y)
        
        # Generamos predicciones
        X_pred = np.linspace(min(x), max(x), 100).reshape(-1, 1)
        y_pred = model.predict(X_pred)
        
        # Mostramos el gráfico con los datos originales y la línea de regresión
        fig, ax = plt.subplots()
        ax.scatter(x, y, label='Original Data')
        ax.plot(X_pred, y_pred, color='red', label='Regression Line')
        ax.set_title(f"{model_type} on {func_type.capitalize()} Function")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        st.pyplot(fig)
        
        # Mostramos el R-squared score
        r2_score = model.score(X, y)
        st.write(f"R-squared score: {r2_score:.4f}")

        # Mostrar parámetros del modelo
        st.subheader("Model Parameters")
        if model_type == 'SVM':
            st.write(f"C: {model.named_steps['svr'].C}")
            st.write(f"Epsilon: {model.named_steps['svr'].epsilon}")
            st.write(f"Kernel: {model.named_steps['svr'].kernel}")
        elif model_type == 'Polynomial':
            st.write(f"Degree: {model.named_steps['polynomialfeatures'].degree}")
            st.write(f"Coefficients: {model.named_steps['linearregression'].coef_}")
            st.write(f"Intercept: {model.named_steps['linearregression'].intercept_}")

# Información adicional
st.sidebar.markdown("---")
st.sidebar.write("Created with Streamlit")