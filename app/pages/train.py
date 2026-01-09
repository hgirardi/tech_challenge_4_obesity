# from __future__ import annotations

# import streamlit as st
# from components.layout import layout, first_subheader
# from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, recall_score, precision_score
# import pandas as pd
# import joblib



# up = st.file_uploader("Arquivo CSV", type=["csv"])
# if up:
#     df = pd.read_csv(up)
#     st.dataframe(df, use_container_width=True)

#     if st.button("Treinar Modelo"):
#         st.success("sucesso")

from components.layout import layout
import streamlit as st
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

from model.model import carregar_modelo
from model.config import DATA_PATH, MODEL_PATH

layout("(Re-)Treinar Modelo")

st.subheader("Dados atuais usados para treinar o modelo")

# Informações
col1, col2, col3, col4 = st.columns([2,1,1,1])
model = None

with col1:
    if DATA_PATH.exists():
        st.info(f"📁 Arquivo atual: `{DATA_PATH.name}`")
        model = carregar_modelo()
    else:
        st.warning("⚠️ Nenhum arquivo de dados encontrado")

with col2:
    cache = "Não"
    if MODEL_PATH.exists():
        cache = "Sim"
    st.metric("Dados em cache?", cache)

with col3:
    if DATA_PATH.exists():
        tamanho = DATA_PATH.stat().st_size / 1024  # KB
        st.metric("Tamanho Atual", f"{tamanho:.1f} KB")

with col4:
    if DATA_PATH.exists():
        data_mod = datetime.fromtimestamp(DATA_PATH.stat().st_mtime)
        st.metric("Última Modificação", data_mod.strftime("%d/%m/%Y"))

if not model == None:
    df = model.dados_processados
    st.dataframe(df.head(50), use_container_width=True)

st.divider()

st.subheader("Atualizar dados do modelo")

# Upload
up = st.file_uploader(
    "Escolha um arquivo CSV com os dados de treinamento",
    type=["csv"],
    help="O arquivo deve conter as mesmas colunas do dataset original"
)

if up:
    try:
        # Ler e validar CSV
        df = pd.read_csv(up)
        
        st.success(f"✅ Arquivo carregado: {len(df)} registros, {len(df.columns)} colunas")
        
        # Preview dos dados
        with st.expander("Visualizar Dados"):
            st.dataframe(df.head(50), use_container_width=True)
            
            # Estatísticas
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Colunas:**")
                st.write(df.columns.tolist())
            with col2:
                st.markdown("**Tipos:**")
                st.write(df.dtypes.to_dict())
        
        # Validação básica
        st.markdown("#### Validação dos Dados")
        
        validacoes = []
        
        # Verificar valores nulos
        total_nulos = df.isnull().sum().sum()
        if total_nulos == 0:
            validacoes.append(("✅", "Sem valores nulos"))
        else:
            validacoes.append(("⚠️", f"{total_nulos} valores nulos encontrados"))
        
        # Verificar tamanho mínimo
        if len(df) >= 100:
            validacoes.append(("✅", f"Tamanho adequado ({len(df)} registros)"))
        else:
            validacoes.append(("❌", f"Dataset muito pequeno ({len(df)} registros)"))
        
        # Mostrar validações
        for emoji, msg in validacoes:
            st.markdown(f"{emoji} {msg}")
        
        st.divider()
        
        # Opções de salvamento
        col1, col2 = st.columns(2)
        
        with col1:
            fazer_backup = st.checkbox(
                "Fazer backup do arquivo atual",
                value=True,
                help="Salva o arquivo atual antes de substituí-lo"
            )
        
        with col2:
            limpar_cache = st.checkbox(
                "Limpar cache após salvar",
                value=True,
                help="Força o recarregamento dos dados"
            )
        
        # Botão de salvar
        if st.button("💾 Salvar e Treinar Modelo", type="primary"):
            try:
                # Criar backup se necessário
                if fazer_backup and DATA_PATH.exists():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = DATA_PATH.parent / f"Obesity_backup_{timestamp}.csv"
                    shutil.copy2(DATA_PATH, backup_path)
                    st.info(f"Backup criado: `{backup_path.name}`")
                
                # Salvar novo arquivo
                df.to_csv(DATA_PATH, index=False)
                st.success(f"✅ Arquivo salvo em: `{DATA_PATH}`")
                
                # Limpar cache
                if limpar_cache:
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.info("🔄 Cache limpo")
                
                # Treinar modelo
                st.divider()
                st.markdown("#### 🤖 Treinando Modelo")
                
                with st.spinner("Treinando modelo... Isso pode levar alguns minutos."):
                    # Carregar modelo e treinar
                    model = carregar_modelo(cache=False)
                    
                    
                st.success("✅ Modelo treinado com sucesso!")
                
            except Exception as e:
                st.error(f"❌ Erro ao salvar/treinar: {str(e)}")
                st.exception(e)
    
    except Exception as e:
        st.error(f"❌ Erro ao ler o arquivo CSV: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Faça upload de um arquivo CSV para começar")