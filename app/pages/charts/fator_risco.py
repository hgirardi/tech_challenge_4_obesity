import pandas as pd
import streamlit as st
import plotly.express as px

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import model.model as Model


def principais_fatores_risco(dados:pd.DataFrame, model:Model):
    """
    Funcao responsável por criar o gráfico de correlação da tab 'Fatores de Risco'
    """
    
    st.subheader("⚠️ Principais Fatores de Risco")
    
    st.markdown(f"""
        <div class="insight-box">
            Nessa análise, obeserva-se que os fatores de risco para obesidade não se dá a um ou outro fator isolado, mas sim quando multiplos fatores são combinados.
            <br>
            Novos fatores foram criados a partir do cruzamento de informações coletadas:
            <br><br>
            <strong>Score de Controle: </strong>Captura o quão saudável/metabólico é o perfil do indivíduo com base em diferentes fatores comportamentais, como:
            <ul>
                <li><i>Positivos: Monitoramento de ingestão calórica, frequencia de consumo de vegetais, consumo diário de água, frequencia semanal de atividade fisica e numero de refeições por dia.</i>
                <li><i>Negativos: Consumo de bebidas alcoólicas, fumante e tempo de uso de eletronicos.</i>
            </ul>
            <strong>Index de Ingestão Calórica: </strong>Busca compreender/aprender a quantidade de calorias que o indivíduo consome com base no seu comportamento.
            <ul>
                <li><i>Aumenta o index: Consumo de alimentos calóricos, consumo de lanches entre as refeições, número de refeições ao dia.</i>
                <li><i>Diminui o index: Frequencia de consumo de vegetais.</i>
            </ul>
            <strong>Nível de Sedentarismo: </strong>A fórmula baseia-se no cruzamento de falta de atividade física com o meio de transporte usado.
            <ul>
                <li><i>Nível mais alto: Quando o indivíduo não realiza atividade física e usa de meio de transportes como automóvel, moto e público.</i>
                <li><i>Nível mais baixo: Quando o indivíduo se exercita e/ou usa meios de transportes como bicicleta ou a pé.</i>
            </ul>
            <strong>Nível de Inatividade: </strong>A fórmula baseia-se na inversão dos valores de atividade física, dando mais peso para os indivíduos com menos exercícios físicos na semana.
            <br><br>
            <strong>💡 Conclusão</strong><br>
            Abaixo podemos observar os 10 maiores fatores causadores de obesidade encontradas nos 972 casos de obesidade. Podemos observar a nível comportamental que a probabilidade de obesidade aumenta muito em indíduos que consomem muitas calorias e possuem um estilo de vida mais sedentário/inativo.
        </div>
        """, unsafe_allow_html=True)

    
    
    # dados = df.copy()
    dados['genero'] = dados['genero'].map({
        "Male": 0,
        "Female": 1
    }).fillna(1).astype(int)
        
    correlacoes = []
    for feature in dados.columns:
        if feature != 'obesidade_bin':
            corr = dados[[feature, 'obesidade_bin']].corr().iloc[0, 1]
            correlacoes.append({
                'fator': feature,
                'correlacao': corr
            })
    
    df_correlacoes = pd.DataFrame(correlacoes)
    
    df_top10 = (df_correlacoes[df_correlacoes['correlacao'] > 0]
                .sort_values('correlacao', ascending=False)
                .head(10)
                )
    
    nomes_amigaveis = {
        'consumo_lanches_entre_refeicoes_n': 'Consumo de Lanches',
        'tempo_uso_eletronicos_n': 'Tempo em Eletrônicos',
        'tempo_eletronico_x_inatividade': 'Sedentarismo Digital',
        'fumante_bin': 'Fumante',
        'score_controle': 'Score de Controle',
        'index_ingestao_calorica': 'Índice Calórico',
        'transporte_passivo': 'Transporte Passivo',
        'ingestao_x_sedentarismo': 'Index de Ingestão × Sedentarismo',
        'balanco_caloria_atividade': 'Balanço Calórico',
        'idade': 'Idade',
        'obesidade_bin': 'Obesidade',
        'historico_familiar_bin': 'Histórico Familiar',
        'monitora_ingestao_calorica_bin': 'Monitora Calorias',
        'frequencia_alimentos_caloricos_bin': 'Alimentos Calóricos',
        'frequencia_consumo_vegetais_n': 'Consumo de Vegetais',
        'consumo_agua_diario_n': 'Consumo de Água',
        'numero_refeicoes_por_dia_n': 'Refeições por Dia',
        'frequencia_semanal_atividade_fisica_n': 'Atividade Física',
        'consumo_bebidas_alcoolicas_n': 'Consumo de Álcool',
        'score_controle_index': 'Índice de Controle',
        'risco_sedentarismo': 'Risco de Sedentarismo',
        'inatividade': 'Inatividade',
        'score_controle_x_sedentarismo': 'Score de Controle × Sedentarismo',
        'genetica_x_inatividade': 'Genética × Inatividade',
        'ingestao_x_inatividade': 'Index de Ingestão × Inatividade',
        'genero': 'Gênero'
    }
    df_top10['fator_legivel'] = df_top10['fator'].map(nomes_amigaveis)

    # col1, col2 = st.columns([2, 1])
    
    st.markdown("#### Top 10 correlações de fatores com obesidade:")

    # with col1:
    fig_corr = px.bar(
        df_top10,
        x='correlacao',
        y='fator_legivel',
        orientation='h',
        title='',
        labels={'correlacao': 'Correlação', 'fator_legivel': 'Fator de Risco'},
        color='correlacao',
        color_continuous_scale='OrRd',
        text='correlacao'
    )
    fig_corr.update_traces(
        texttemplate='%{text:.2f}',  
        textposition='outside'  
    )

    fig_corr.update_layout(showlegend=False, height=600, xaxis_title='Correlação com Obesidade', yaxis_title='')
    st.plotly_chart(fig_corr, use_container_width=True, key="bar_correlacao_fatores")
    
    # with col2:
    #     st.markdown("### 📊 Taxas de Obesidade")
    #     for _, row in df_correlacoes.iterrows():
    #         st.metric(
    #             row['fator'],
    #             f"{row['taxa_obesidade']:.1f}%",
    #             help=f"Taxa de obesidade entre quem tem este fator"
    #         )
    
    st.warning("⚠️ Cuidado! No gráfico acima, a idade isoladamente aparece como segundo fator que mais tem correlação com obesidade. Entretanto, não é incorreto afirmar que ficaremos mais obesos ao ficarmos mais velhos.\nA idade, nesse caso, entra como um modulador, pois como podemos observar nos gráficos abaixo, ela está diretamente associada a uma baixa nas atividades físicas e em um aumento de ingesta calórica.")

def idade_obesidade_atividade_fisica(df:pd.DataFrame): 
    """
    Funcao responsável por criar o gráfico de idade x obesidade x atividade física da tab 'Fatores de Risco'
    """
    # Assumindo que df é o seu DataFrame com as colunas corretas
    grouped = df.groupby("frequencia_semanal_atividade_fisica").agg(
        mean_age=("idade", "mean"),
        obesity_rate=("obesidade_bin", "mean")
    ).reset_index()

    # Labels textuais
    faf_labels = {
        0: "Nunca",
        1: "1-2x/sem",
        2: "3-4x/sem",
        3: "5x ou mais"
    }

    # Ordenar e criar labels
    grouped = grouped.sort_values("frequencia_semanal_atividade_fisica").copy()
    grouped['faf_label'] = grouped['frequencia_semanal_atividade_fisica'].map(faf_labels)

    # Criar figura com dois eixos Y
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Adicionar barras (idade média)
    fig.add_trace(
        go.Bar(
            x=grouped['faf_label'],
            y=grouped['mean_age'],
            name='Idade Média',
            marker_color='#4A90E2',
            text=grouped['mean_age'].round(1),
            texttemplate='%{text:.1f}',
            textposition='outside',
            textfont=dict(size=11),
            hovertemplate='<b>%{x}</b><br>Idade Média: %{y:.1f} anos<extra></extra>'
        ),
        secondary_y=False
    )

    # Adicionar linha (prevalência de obesidade)
    fig.add_trace(
        go.Scatter(
            x=grouped['faf_label'],
            y=grouped['obesity_rate'] * 100,  # Converter para porcentagem
            name='Prevalência de Obesidade',
            mode='lines+markers+text',
            line=dict(color='#E74C3C', width=3),
            marker=dict(size=10, color='#E74C3C'),
            text=[f"{val:.1f}%" for val in grouped['obesity_rate'] * 100],
            textposition='bottom center',
            textfont=dict(size=11, color='#E74C3C'),
            hovertemplate='<b>%{x}</b><br>Obesidade: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=True
    )

    # Configurar layout
    fig.update_layout(
        title='Idade Média e Prevalência de Obesidade por Nível de Atividade Física',
        xaxis_title='Frequência de Atividade Física',
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Configurar eixos Y
    fig.update_yaxes(
        title_text="Idade Média (anos)",
        secondary_y=False,
        showgrid=True
    )

    fig.update_yaxes(
        title_text="Prevalência de Obesidade (%)",
        secondary_y=True,
        range=[0, 75],
        ticksuffix="%"
    )

    st.plotly_chart(fig, use_container_width=True, key="grafico_idade_obesidade_faf")

def idade_obesidade_inatividade(df:pd.DataFrame):
    
    # Criar faixas de inatividade
    df['faixa_inatividade'] = pd.cut(
        df['inatividade'],
        bins=4,
        labels=['Baixa', 'Moderada', 'Alta', 'Muito Alta']
    )

    # Agrupar dados
    grouped = df.groupby('faixa_inatividade').agg(
        mean_age=('idade', 'mean'),
        obesity_rate=('obesidade_bin', 'mean'),
        mean_sedentarismo=('risco_sedentarismo', 'mean'),
        count=('obesidade_bin', 'count')
    ).reset_index()

    # Criar gráfico com dois eixos Y
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Barras: idade média
    fig.add_trace(
        go.Bar(
            x=grouped['faixa_inatividade'],
            y=grouped['mean_age'],
            name='Idade Média',
            marker_color='#4A90E2',
            text=grouped['mean_age'].round(1),
            texttemplate='%{text:.1f}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Idade: %{y:.1f} anos<extra></extra>'
        ),
        secondary_y=False
    )

    # Linha: prevalência de obesidade
    fig.add_trace(
        go.Scatter(
            x=grouped['faixa_inatividade'],
            y=grouped['obesity_rate'] * 100,
            name='Prevalência de Obesidade',
            mode='lines+markers+text',
            line=dict(color='#E74C3C', width=3),
            marker=dict(size=12, color='#E74C3C'),
            text=[f"{val:.1f}%" for val in grouped['obesity_rate'] * 100],
            textposition='bottom center',
            textfont=dict(size=11, color='#E74C3C'),
            hovertemplate='<b>%{x}</b><br>Obesidade: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=True
    )

    # Linha: risco de sedentarismo (mesmo eixo da obesidade)
    fig.add_trace(
        go.Scatter(
            x=grouped['faixa_inatividade'],
            y=grouped['mean_sedentarismo'] * 100,
            name='Risco Sedentarismo',
            mode='lines+markers',
            line=dict(color='#FFA500', width=2, dash='dash'),
            marker=dict(size=10, color='#FFA500'),
            hovertemplate='<b>%{x}</b><br>Sedentarismo: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=True
    )

    fig.update_layout(
        title='Idade Média, Obesidade e Sedentarismo por Nível de Inatividade',
        xaxis_title='Nível de Inatividade',
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_yaxes(title_text="Idade Média (anos)", secondary_y=False, showgrid=True)
    fig.update_yaxes(title_text="Taxa (%)", secondary_y=True, range=[0, 100], ticksuffix="%")

    st.plotly_chart(fig, use_container_width=True, key="grafico_inatividade")

def mapa_calor_obesidade_idade_inatividade(df):
    
    # Criar faixas de idade e inatividade
    df['faixa_idade'] = pd.cut(
        df['idade'],
        bins=[0, 20, 30, 40, 50, 100],
        labels=['< 20', '20-29', '30-39', '40-49', '50+']
    )

    df['faixa_inatividade'] = pd.cut(
        df['inatividade'],
        bins=4,
        labels=['Baixa', 'Moderada', 'Alta', 'Muito Alta']
    )

    # Criar matriz de obesidade
    heatmap_data = df.groupby(['faixa_idade', 'faixa_inatividade'])['obesidade_bin'].agg(['mean', 'count']).reset_index()
    heatmap_pivot = heatmap_data.pivot(
        index='faixa_inatividade',
        columns='faixa_idade',
        values='mean'
    ) * 100

    # Criar heatmap
    fig = px.imshow(
        heatmap_pivot,
        text_auto='.1f',
        aspect='auto',
        color_continuous_scale='YlOrRd',
        labels={'x': 'Faixa Etária', 'y': 'Nível de Inatividade', 'color': 'Taxa de Obesidade (%)'},
        title='Taxa de Obesidade por Idade e Nível de Inatividade'
    )

    fig.update_xaxes(side="bottom")
    fig.update_layout(
        height=500,
        coloraxis_colorbar=dict(
            title="Obesidade (%)",
            ticksuffix="%"
        )
    )

    st.plotly_chart(fig, use_container_width=True, key="heatmap_idade_inatividade")

    # Insights
    st.info("""
    💡 **Interpretação:** Cores mais quentes (vermelho) indicam maior taxa de obesidade. 
    Quanto mais alta a inatividade e maior a idade, maior tende a ser a prevalência.
    """)