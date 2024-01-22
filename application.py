
import streamlit as st
import numpy as np

# Générer des données aléatoires
data = np.random.randn(20, 2)

# Créer une application web avec Streamlit
st.title('Application Streamlit Simple')
st.line_chart(data)

