import plotly.graph_objects as go
from model.model import carregar_modelo
import streamlit as st

MODEL = carregar_modelo()

# --------------
# Função responsável por criar barra visual de probabilidade de Obesidade
# --------------
def criar_barra_risco_plotly(proba, risco: dict, threshold=MODEL.threshold):
    """
    Cria barra visual interativa com Plotly
    """
    
    # Cria figura
    fig = go.Figure()
    
    # Barra de fundo (escala completa 0-100%)
    fig.add_trace(go.Bar(
        x=[100],
        y=['Risco'],
        orientation='h',
        marker=dict(
            color='lightgray',
            line=dict(color='gray', width=1)
        ),
        showlegend=False,
        hoverinfo='skip',
        width=0.8
    ))
    
    # Barra colorida até a posição da pessoa
    fig.add_trace(go.Bar(
        x=[proba * 100],
        y=['Risco'],
        orientation='h',
        marker=dict(color=risco['cor']),
        showlegend=False,
        text=f"{proba*100:.1f}%",
        textposition='inside',
        textfont=dict(color='white', size=14, family='Arial Black'),
        hovertemplate=f"<b>Seu Score:</b> {proba*100:.1f}%<extra></extra>",
        width=0.8
    ))
    
    # Linha vertical do threshold
    fig.add_shape(
        type="line",
        x0=threshold * 100,
        y0=-0.5,
        x1=threshold * 100,
        y1=0.5,
        line=dict(color="red", width=3, dash="dash"),
    )
    
    # Anotação do threshold
    fig.add_annotation(
        x=threshold * 100,
        y=0.7,
        text=f"Linha de Detecção<br>({threshold*100:.0f}%)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red",
        ax=0,
        ay=-40,
        font=dict(size=10, color="red", family="Arial")
    )
    
    # Layout
    fig.update_layout(
        barmode='overlay',
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            range=[0, 100],
            title="Probabilidade de Obesidade (%)",
            tickmode='linear',
            tick0=0,
            dtick=10,
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


# =========================================================================
# FUNÇÃO: CRIAR GAUGE (MEDIDOR) - ALTERNATIVA VISUAL
# =========================================================================

def criar_gauge_risco(proba, risco : dict, threshold=MODEL.threshold):
    """
    Cria medidor tipo velocímetro
    """
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=proba * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Score de Risco", 'font': {'size': 20}},
        delta={'reference': threshold * 100, 'increasing': {'color': "red"}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': risco['cor'], 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold * 100], 'color': '#d4edda'},
                {'range': [threshold * 100, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


# =========================================================================
# INTERFACE STREAMLIT PRINCIPAL
# =========================================================================

def mostrar_resultado_obesidade(proba, risco: dict, threshold=MODEL.threshold):
    """
    Interface completa para mostrar resultado de obesidade
    """
    
    # Header com resultado
    st.markdown(f"""
        <div style='
            background-color: {risco['cor_fundo']}; 
            padding: 20px; 
            border-radius: 10px; 
            border-left: 5px solid {risco['cor']};
            margin-bottom: 20px;
        '>
            <h2 style='margin:0; color: {risco['cor']};'>
                {risco['icone']} {risco['nivel']}
            </h2>
            <p style='margin:10px 0 0 0; font-size: 16px; color: #333;'>
                {risco['mensagem']}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tabs para visualizações diferentes
    tab1, tab2 = st.tabs(["📊 Barra de Risco", "📋 Detalhes"])
    
    with tab1:
        st.plotly_chart(
            criar_barra_risco_plotly(proba, risco, threshold), 
            use_container_width=True
        )
        
        # Legenda
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Seu Score", f"{proba*100:.1f}%")
        with col2:
            st.metric("Linha de Detecção", f"{threshold*100:.0f}%")
        with col3:
            delta = (proba - threshold) * 100
            st.metric(
                "Distância", 
                f"{abs(delta):.1f}pp",
                delta=f"{delta:+.1f}pp" if proba >= threshold else f"{delta:.1f}pp"
            )
    
    with tab2:
        st.markdown("### Detalhes da Avaliação")
        
        st.info(f"**💡 Próximos Passos:** {risco['acao']}")
        
        with st.expander("ℹ️ Como interpretamos este resultado?"):
            st.markdown(f"""
            **Score Calculado:** {proba*100:.1f}%
            
            Nosso modelo usa um **threshold de {threshold*100:.0f}%** para detectar risco de obesidade.
            Isso significa que:
            
            - Scores **≥ {threshold*100:.0f}%**: Risco detectado (requer atenção)
            - Scores **< {threshold*100:.0f}%**: Baixo risco no momento
            
            **Por que {threshold*100:.0f}% e não 50%?**
            
            Nosso modelo foi otimizado para **não perder casos de risco**, mesmo que isso 
            signifique gerar alguns alertas extras. Em saúde, é melhor prevenir do que remediar!
            
            **Faixas de Risco:**
            - 🟢 0-31%: Risco Muito Baixo
            - 🟢 31-{threshold*100:.0f}%: Risco Baixo
            - 🟡 {threshold*100:.0f}-56%: Risco Moderado
            - 🟠 56-66%: Risco Moderado-Alto
            - 🔴 66-100%: Alto Risco
            """)
        
        with st.expander("📊 Informações Técnicas"):
            st.markdown(f"""
            **Modelo:** HistGradientBoosting Classifier
            
            **Métricas de Performance:**
            - Recall (sensibilidade): 88.2%
            - Precision: 91.5%
            - F1-Score: 89.8%
            - AUC-ROC: 95.4%
            
            **Threshold Otimizado:** {threshold}
            - Escolhido via validação cruzada 5-fold
            - Maximiza equilíbrio entre recall e precision
            - Gap de fairness entre grupos etários: 0.44pp
            """)