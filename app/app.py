from __future__ import annotations

from components.layout import layout
from model.model import carregar_modelo
import streamlit as st

layout("Home")

try:
    model = carregar_modelo()
except Warning as e:
    st.warning(e)
except IOError as e:
    st.error(e)

st.write("Bem-vindo!")