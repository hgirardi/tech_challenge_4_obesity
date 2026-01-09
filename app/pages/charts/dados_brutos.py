import pandas as pd
import streamlit as st

def dados_brutos(df:pd.DataFrame):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filtro_genero = st.multiselect(
            "Filtrar por Gênero",
            options=df['genero'].unique(),
            default=df['genero'].unique(),
            key="filtro_genero_dados"
        )
    
    with col2:
        filtro_obesidade = st.multiselect(
            "Filtrar por Obesidade",
            options=[0, 1],
            default=[0, 1],
            format_func=lambda x: 'Obeso' if x == 1 else 'Não Obeso',
            key="filtro_obesidade_dados"
        )
    
    with col3:
        num_linhas = st.selectbox(
            "Número de linhas",
            options=[50, 100, 500, 1000, len(df)],
            index=1,
            key="num_linhas_dados"
        )
    
    df_filtrado = df[
        (df['genero'].isin(filtro_genero)) &
        (df['obesidade_bin'].isin(filtro_obesidade))
    ]
    
    st.info(f"📊 Mostrando {min(num_linhas, len(df_filtrado))} de {len(df_filtrado)} registros filtrados")
    
    st.dataframe(
        df_filtrado.head(num_linhas),
        use_container_width=True,
        height=500
    )
    
    with st.expander("📊 Estatísticas Descritivas"):
        st.dataframe(df_filtrado.describe(), use_container_width=True)
    
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar dados filtrados (CSV)",
            data=csv,
            file_name='dados_obesidade_filtrados.csv',
            mime='text/csv',
            use_container_width=True,
            key="download_filtrados"
        )
    
    with col2:
        csv_completo = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar dados completos (CSV)",
            data=csv_completo,
            file_name='dados_obesidade_completos.csv',
            mime='text/csv',
            use_container_width=True,
            key="download_completos"
        )