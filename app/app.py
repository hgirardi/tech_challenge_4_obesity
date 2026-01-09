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


# Header Principal
st.markdown("""
<div class="main-header">
    <h1>🏥 Sistema Preditivo de Obesidade</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">
        Utilizando Machine Learning para auxiliar na detecção precoce de obesidade
    </p>
</div>
""", unsafe_allow_html=True)

# Introdução
st.markdown("### 👋 Bem-vindo!")
st.write("""
Este sistema foi desenvolvido como parte do **Tech Challenge Fase 4** da FIAP, 
com o objetivo de criar uma ferramenta inteligente para auxiliar profissionais de 
saúde na identificação de fatores de risco relacionados à obesidade.
""")

st.divider()

# Seção de Funcionalidades
st.markdown("### Funcionalidades do Sistema")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 Sistema Preditivo</h3>
        <p>Questionário inteligente para avaliação de risco de obesidade baseado em:</p>
        <ul>
            <li>Dados demográficos</li>
            <li>Hábitos alimentares</li>
            <li>Nível de atividade física</li>
            <li>Comportamento sedentário</li>
        </ul>
        <p><strong>→ Resultado em tempo real com classificação de risco</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Acessar Sistema Preditivo", key="btn_predict", use_container_width=True):
        st.switch_page("pages/predict.py")

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Painel Analítico</h3>
        <p>Dashboard executivo com análises detalhadas dos dados:</p>
        <ul>
            <li>Visualizações interativas</li>
            <li>Correlações entre variáveis</li>
            <li>Análise demográfica</li>
            <li>Fatores de risco identificados</li>
        </ul>
        <p><strong>→ Insights baseados em dados reais</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📈 Acessar Painel Analítico", key="btn_data", use_container_width=True):
        st.switch_page("pages/data.py")

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>⚙️ Treinar Modelo</h3>
        <p>Gerenciamento e retreinamento do modelo de ML:</p>
        <ul>
            <li>Métricas de performance</li>
            <li>Upload de novos dados</li>
            <li>Retreinamento automatizado</li>
            <li>Histórico de versões</li>
        </ul>
        <p><strong>→ Melhoria contínua do sistema</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔧 Acessar Treinamento", key="btn_train", use_container_width=True):
        st.switch_page("pages/train.py")

st.divider()

# Métricas do Sistema
st.markdown("### 📈 Sobre o Modelo")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2>2,111</h2>
        <p>Registros<br>de Treino</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h2>20+</h2>
        <p>Variáveis<br>Analisadas</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h2>HistGradient</h2>
        <p>Algoritmo<br>Boosting</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h2>~90%</h2>
        <p>Acurácia<br>do Modelo</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Seção de Tecnologias
st.markdown("### 🛠️ Tecnologias Utilizadas")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Machine Learning & Data Science:**
    - 🐍 Python 3.11
    - 📊 Pandas & NumPy
    - 🤖 Scikit-learn (HistGradientBoosting)
    - 📈 Plotly para visualizações
    """)

with col2:
    st.markdown("""
    **Interface & Deploy:**
    - 🎨 Streamlit
    - 🔄 Git & GitHub
    - ☁️ Streamlit Cloud
    - 📦 Joblib para persistência
    """)

st.divider()

# Informações do Autor
st.markdown("### 👨‍💻 Sobre o Projeto")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="author-box">
        <h4>📚 FIAP - Tech Challenge Fase 4</h4>
        <p><strong>Aluno:</strong> Henrique Girardi dos Santos</p>
        <p><strong>RM:</strong> 362082</p>
        <p><strong>Curso:</strong> Pós-graduação em Data Analytics</p>
        <p><strong>Objetivo:</strong> Desenvolver um sistema de predição de obesidade utilizando 
        técnicas de Machine Learning para auxiliar profissionais de saúde na identificação 
        precoce de fatores de risco.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("#### 🔗 Links Úteis")
    st.link_button(
        "📂 GitHub Repository",
        "https://github.com/hgirardi/tech_challenge_4_obesity",
        use_container_width=True
    )

# Rodapé
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>FIAP | © 2026 Henrique Girardi</p>
    <p style="font-size: 0.9rem;">
        Este sistema é uma ferramenta de apoio à decisão e não substitui a avaliação médica profissional.
    </p>
</div>
""", unsafe_allow_html=True)