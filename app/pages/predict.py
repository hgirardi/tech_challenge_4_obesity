from __future__ import annotations

import streamlit as st
import pandas as pd

from components.layout import layout, first_subheader
from model.model import carregar_modelo
from pages.utils import *

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

layout("Modelo de Previsão de Obesidade")

# dicionário que irá conter os valores coletados para que seja usado contra o modelo
input_data = {}

# ------------------------
# DADOS PESSOAIS
# ------------------------
first_subheader("Dados Pessoais")
#c1, c2, c3 = st.columns([1, 1, 1], vertical_alignment="bottom")
c1, c2 = st.columns([1, 1], vertical_alignment="bottom")

with c1:
    input_data["genero"] = st.radio("Sexo Biológico", options=["Female","Male"], format_func=lambda x: "Feminino" if x == "Female" else "Masculino", horizontal=True)

with c2:
    input_data["idade"] = st.number_input("Idade", min_value=0, max_value=120, value=25, step=1, width=100)

# with c3:
#     input_data["family_history"] = st.radio("Possui histórico familiar de sobrepeso?", options=[0,1], format_func=lambda x: "Não" if x == 0 else "Sim", index=0, horizontal=True)


# ------------------------
# HÁBITOS COMPORTAMENTAIS
# ------------------------
st.subheader("Hábitos Comportamentais")
c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
with c1:
    input_data["fumante"] = st.radio("Fumante?", options=["no","yes"], format_func=lambda x: "Não" if x == "no" else "Sim", index=0, horizontal=True)
    input_data["frequencia_semanal_atividade_fisica"] = st.radio("Frequência que pratica atividade física por semana:", options=[0,1,2,3],index=0, horizontal=True, 
                                format_func=lambda x: "Nenhuma" if x == 0 
                                                else "Uma a duas vezes" if x == 1
                                                else "Três a quatro vezes" if x == 2
                                                else "Cinco vezes ou mais")

with c2:
    input_data["consumo_agua_diario"] = st.radio("Consumo de água diário:", options=[1,2,3],index=0, 
                                 horizontal=True, format_func=lambda x: "Menos de 1L" if x == 1 else "Entre 1 e 2L" if x == 2 else "Mais de 2L")
    input_data["meio_transporte_habitual"] = st.radio("Meio de Transporte Habitual:",
                                    options=["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"], index=0, horizontal=True ,
                                    format_func=lambda x: "Automóvel" if x == "Automobile" 
                                                    else "Motocicleta" if x == "Motorbike"
                                                    else "Bicicleta" if x == "Bike"
                                                    else "Transporte Público" if x == "Public_Transportation"
                                                    else "A pé")

input_data["tempo_uso_eletronicos"] = st.radio("Tempo diário usando dispositivos tecnológicos como celular, videogame, televisão, computador e outros:",
                            options=[0,1,2],index=0, horizontal=True , format_func=lambda x: "Até 2 horas" if x == 0 
                                                                                        else "Entre 3 e 5 horas" if x == 1
                                                                                        else "Acima de 5 horas")


# ------------------------
# HÁBITOS ALIMENTARES
# ------------------------
st.subheader("Hábitos Alimentares")

c1, c2 = st.columns([1, 1], vertical_alignment="bottom")

with c1:
    input_data["frequencia_alimentos_caloricos"] = st.radio("Possui consumo frequentes de alimentos muito calóricos?", index=0, horizontal=True,
                            options=["no","yes"], format_func=lambda x: "Não" if x == "no" else "Sim")
    
    input_data["numero_refeicoes_por_dia"] = st.radio("Número de refeições principais por dia:", index=0, horizontal=True, 
                            options=[1,2,3,4], format_func=lambda x: "4 ou mais" if x == 4 else str(x))
    
    input_data["monitora_ingestao_calorica"] = st.radio("Monitora a ingestão calórica diária:", index=0, horizontal=True,
                            options=["no","yes"], format_func=lambda x: "Não" if x == "no" else "Sim")

with c2:
    input_data["frequencia_consumo_vegetais"] = st.radio("Frequência de consumo de vegetais nas refeições:", index=0, horizontal=True, 
                            options=[1,2,3], format_func=lambda x: "Raramente" if x == 1 else "Às vezes" if x == 2 else "Sempre")

    input_data["consumo_lanches_entre_refeicoes"] = st.radio("Consumo de lanches/comes entre as refeições:", index=0, horizontal=True,
                            options=["no","Sometimes","Frequently"], format_func=lambda x: "Não consome" if x == "no" 
                                                                                        else "Às vezes" if x == "Sometimes"
                                                                                        else "Frequentemente")
    
    input_data["consumo_bebidas_alcoolicas"] = st.radio("Consumo de bebidas alcoólicas:", index=0, horizontal=True,
                            options=["no","Sometimes","Frequently", "Always"], format_func=lambda x: "Não consome" if x == "no" 
                                                                                                else "Às vezes" if x == "Sometimes"
                                                                                                else "Frequentemente" if x == "Frequently"
                                                                                                else "Sempre")

# ------------------------
# BOTÃO DE PROCESSAMENTO
# ------------------------
if st.button("Prever"):
    modelo = carregar_modelo()

    try:
        proba, pred = modelo.validar_prob(pd.DataFrame(input_data, index=[0]))

        dict_risco = modelo.classificar_risco(proba)
    
        mostrar_resultado_obesidade(proba, dict_risco)
    except ValueError as e:
        logging.error(e)
