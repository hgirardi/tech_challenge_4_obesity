import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from typing import Dict

def distribuicao_obesidade(df:pd.DataFrame, metrics:Dict):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig_pie = px.pie(
            values=[metrics['total_obesos'], metrics['total_registros'] - metrics['total_obesos']],
            names=['Obeso', 'Não Obeso'],
            #title='Distribuição da População',
            color_discrete_sequence=['#28a745','#dc3545'],
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=450)
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_visao_geral")
    
    with col2:
        st.markdown("####  Insights")
        if metrics['taxa_obesidade'] > 50:
            st.warning("⚠️ Taxa de obesidade acima de 50% indica alto risco populacional")
        elif metrics['taxa_obesidade'] > 30:
            st.warning("💡 Taxa moderada de obesidade na população - atenção aos fatores de risco")
        else:
            st.success("✅ Taxa relativamente controlada")
        st.markdown(f"""
        <div class="insight-box">
            <strong>🔴 {metrics['taxa_obesidade']:.1f}%</strong> da população apresenta obesidade
            <br><br>
            Isso representa <strong>{metrics['total_obesos']:,}</strong> pessoas de um total de <strong>{metrics['total_registros']:,}</strong>
            <br><br>
            Nesse levantamento descobriu-se que não existe apenas um ou dois fatores isolados que causam obesidade na população, mas sim um conjunto de fatores que, juntos, podem levar a obesidade
        </div>
        """, unsafe_allow_html=True)

def analise_genero(df):
    col1, col2 = st.columns([2,1])
    
    mapeamento_genero = {
        'Male': 'Homens',
        'Female': 'Mulheres'
    }

    df_genero = df.groupby(['genero', 'obesidade_bin']).size().reset_index(name='count')
    df_genero['obesidade'] = df_genero['obesidade_bin'].map({0: 'Não Obeso', 1: 'Obeso'})
    df_genero['genero_pt'] = df_genero['genero'].map(mapeamento_genero)
    
    with col1:
        fig_gender = px.bar(
            df_genero,
            x='genero_pt',
            y='count',
            color='obesidade',
            #title='Distribuição de Obesidade por Gênero',
            color_discrete_map={'Obeso': '#dc3545', 'Não Obeso': '#28a745'},
            barmode='group'
        )
        fig_gender.update_layout(xaxis_title="Gênero", yaxis_title="Quantidade", height=400)
        st.plotly_chart(fig_gender, use_container_width=True, key="bar_genero")
    
    with col2:
        
        taxa_genero = df.groupby('genero')['obesidade_bin'].agg(['sum', 'count'])
        taxa_genero['taxa'] = (taxa_genero['sum'] / taxa_genero['count']) * 100
        taxa_genero.index = taxa_genero.index.map(mapeamento_genero)
        
        # Calcular diferença entre gêneros
        if len(taxa_genero) == 2:
            generos = taxa_genero.index.tolist()
            taxa_g1 = taxa_genero.loc[generos[0], 'taxa']
            taxa_g2 = taxa_genero.loc[generos[1], 'taxa']
            diferenca_absoluta = abs(taxa_g1 - taxa_g2)
        
        # Métrica para cada gênero
        for idx, genero in enumerate(taxa_genero.index):
            taxa = taxa_genero.loc[genero, 'taxa']
            total = taxa_genero.loc[genero, 'count']
            
            # Adicionar delta apenas no segundo gênero
            if idx == 1 and len(taxa_genero) == 2:
                delta_val = taxa - taxa_genero.iloc[0]['taxa']
                st.metric(
                    label=f"{genero}",
                    value=f"{taxa:.1f}%",
                    delta=f"{delta_val:+.1f}% vs {generos[0]}",
                    help=f"Total: {total:,} pessoas"
                )
            else:
                st.metric(
                    label=f"{genero}",
                    value=f"{taxa:.1f}%",
                    help=f"Total: {total:,} pessoas"
                )
        
        # Mostrar diferença total
        if len(taxa_genero) == 2:
            st.divider()
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                <strong>Diferença entre gêneros:</strong> {diferenca_absoluta:.1f} pontos percentuais
            </div>
            """, unsafe_allow_html=True)

def analise_faixa_etaria(df:pd.DataFrame):
    col1, col2 = st.columns(2)
    
    df['obesidade_cat'] = df['obesidade_bin'].map({0:"Não Obeso", 1:"Obeso"})

    with col1:
        fig_idade = px.histogram(
            df,
            x='idade',
            color='obesidade_cat',
            title='Distribuição de Idade por Grupo',
            labels={'obesidade_cat': 'Obesidade', 'idade': 'Idade'},
            color_discrete_map={'Não Obeso': '#28a745', 'Obeso': '#dc3545'},
            nbins=30,
            barmode='overlay',
            opacity=0.7
        )
        fig_idade.update_layout(xaxis_title="Idade", yaxis_title="Frequência", height=400)
        st.plotly_chart(fig_idade, use_container_width=True, key="hist_idade")
    
    with col2:
        fig_box = px.box(
            df,
            x='obesidade_cat',
            y='idade',
            title='Distribuição de Idade por Grupo',
            labels={'obesidade_cat': 'Grupo', 'idade': 'Idade'},
            color='obesidade_cat',
            color_discrete_map={'Não Obeso': '#28a745', 'Obeso': '#dc3545'}
        )
        # fig_box.update_xaxes(ticktext=['Não Obeso', 'Obeso'], tickvals=[0, 1])
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True, key="box_idade")
    
    col1, col2, col3 = st.columns(3)
    idade_obesos = df[df['obesidade_bin'] == 1]['idade'].mean()
    idade_nao_obesos = df[df['obesidade_bin'] == 0]['idade'].mean()
    
    with col1:
        st.metric("Idade Média (Obesos)", f"{idade_obesos:.1f} anos")
    with col2:
        st.metric("Idade Média (Não Obesos)", f"{idade_nao_obesos:.1f} anos")
    with col3:
        diferenca = idade_obesos - idade_nao_obesos
        st.metric("Diferença", f"{diferenca:+.1f} anos")