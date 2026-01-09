from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from components.layout import layout, first_subheader
from model.model import carregar_modelo

from pages.charts.fator_risco import *
from pages.charts.dados_brutos import *
from pages.charts.geral import *
from pages.charts.risco import *

layout("Data")

first_subheader("📊 Análise Exploratória - Obesidade")

with st.spinner("Carregando dados..."):
    try:
        model = carregar_modelo()
        df = model.dados_processados
        df['obesidade_bin'] = model.y
    except Warning as e:
        st.warning(e)
    
    except IOError as e:
        st.error(e)


col1, col2, col3, col4 = st.columns(4)

total_registros = len(df)
total_obesos = df['obesidade_bin'].sum()
taxa_obesidade = (total_obesos / total_registros) * 100
idade_media = df['idade'].mean()

metrics = {
    'total_registros':total_registros,
    'total_obesos':total_obesos,
    'taxa_obesidade':taxa_obesidade,
    'idade_media':idade_media
}

with col1:
    st.metric(
        "Total de Registros",
        f"{total_registros}",
        help="Total de pessoas na base de dados"
    )

with col2:
    st.metric(
        "Casos de Obesidade",
        f"{total_obesos:,}",
        help="Pessoas classificadas como obesas"
    )

with col3:
    st.metric(
        "Taxa de Obesidade",
        f"{taxa_obesidade:.1f}%",
        help="Percentual de obesidade na população"
    )

with col4:
    st.metric(
        "Idade Média",
        f"{idade_media:.1f} anos",
        help="Média de idade da população"
    )

# ========== TABS ==========
tab1, tab2, tab3 = st.tabs([
    "🎯 Visão Geral",
    "⚠️ Fatores de Risco",
    "📋 Dados Brutos"
])

# ========== TAB 1: VISÃO GERAL ==========
with tab1:
    st.subheader("Distribuição de Obesidade")
    distribuicao_obesidade(df,{
        'total_registros':total_registros,
        'total_obesos':total_obesos,
        'taxa_obesidade':taxa_obesidade,
        'idade_media':idade_media
    })
    
    st.divider()
    
    st.subheader("Análise por Gênero")
    analise_genero(df)

    st.divider()

    st.subheader("Análise por Faixa Etária")
    analise_faixa_etaria(df)


# ========== TAB 2: FATORES DE RISCO ==========
with tab2:
    
    dados = df[model.retornar_features_escolhidas()].copy()
    dados['obesidade_bin'] = df['obesidade_bin']
    principais_fatores_risco(dados, model)

    st.divider()

    st.subheader("📊 Idade e Obesidade por Nível de Atividade Física")
    idade_obesidade_atividade_fisica(df)

    st.subheader("📊 Idade e Obesidade por Nível de Inatividade")
    idade_obesidade_inatividade(df)

    st.subheader("🔥 Mapa de Calor: Obesidade por Idade e Inatividade")
    mapa_calor_obesidade_idade_inatividade(df)
    

# ========== TAB 3: DADOS BRUTOS ==========
with tab3:
    st.subheader("📋 Dados Brutos")
    dados_brutos(df)
    

st.divider()
st.caption(f"📊 Dashboard gerado com {len(df):,} registros | Modelo: {model.data_status}")