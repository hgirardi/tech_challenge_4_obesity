#model.py
import joblib
import streamlit as st
import pandas as pd

from model.utils import *
from pathlib import Path
from typing import Tuple, Dict
from datetime import datetime

from model.config import DATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE, THRESHOLD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model:
    """ Classe responsável por gerenciar o Modelo de ML """

    def __init__(self, cache=True):

        # variáveis de configuração
        self.model_path = MODEL_PATH
        self.data_path = DATA_PATH
        self.random_state = RANDOM_STATE
        self.test_size = TEST_SIZE
        self.threshold = THRESHOLD

        # variáveis do modelo
        self.pipe = None
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.is_trained = False
        self.data_status = None

        # carregar dados automaticamente para deixá-los em cache
        self._iniciar_modelo(cache=cache)

    def _iniciar_modelo(self, cache=True):
        """Inicializa o modelo, carregando seu dados em cache"""

        # Carrega o modelo treinado, caso o arquivo data/model.joblib exista, ao menos que o parametro cache seja False
        # O Cache pode ser False quando estamos carregando um novo arquivo CSV para re-treinar o modelo via Admin > Treinar Modelo
        if self.model_path.exists() and cache:

            try:
                model_data = joblib.load(self.model_path)
                
                logger.info(f"Arquivo {self.model_path} carregado.")
                
                self.pipe = model_data.get('pipeline')
                self.dados_processados = model_data.get('dados_processados')
                self.X = model_data.get('X')
                self.y = model_data.get('y')
                self.X_train = model_data.get('X_train')
                self.X_test = model_data.get('X_test')
                self.y_train = model_data.get('y_train')
                self.y_test = model_data.get('y_test')
                self.threshold = model_data.get('config').get('threshold')

                if self.pipe is not None:
                    self.is_trained = True
                    self.data_status = 'pipe_finalizado'
                    logger.info(f"✅ Pipeline carregado com sucesso.")
            
            except Exception as e:
                logger.error(f"❌ Erro ao carregar pipeline: {e}")
                raise IOError(f"Erro ao carregar o modelo de {self.model_path}: {str(e)}")

            return

        if self.data_path.exists():
            logger.info(f"Pipeline não encontrado. Carregando dados de {self.data_path}")

            try:
                df = self.carregar_dados_csv()
                self.data_status = 'arquivo_csv_carregado'
                logger.info(f"Dados carregados do arquivo CSV.")
                
                self.data = self.processar_dados(df)
                logger.info(f"Features processadas.")
                                
                self.y = self.criar_target_binario(self.data)
                logger.info(f"Target criado.")
                self.data_status = 'dados_completos'
                
                # Criar X apenas com as features escolhidas
                self.X = self.data[self.retornar_features_escolhidas()]
                logger.info(f"Features escolhidas.")

                # Split estratificado
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=self.y
                )
                logger.info(f"Dados de teste e treino criados.")
                # Treino
                self.pipe = self.contruir_pipeline(self.X_train)
                logger.info(f"Pipeline construído.")
                
                self.pipe.fit(self.X_train, self.y_train)
                logger.info(f"Pipeline.fit executado.")
                self.data_status = 'pipe_finalizado'
                self.is_trained = True

                joblib_file = {
                    'pipeline': self.pipe
                    , 'dados_processados': self.data
                    , 'X':self.X
                    , 'y':self.y
                    , 'X_train':self.X_train
                    , 'X_test':self.X_test
                    , 'y_train':self.y_train
                    , 'y_test':self.y_test
                    , 'config': {
                        'random_state':self.random_state
                        , 'test_size':self.test_size
                        , 'threshold':self.threshold
                    }
                }

                # Cria um backup caso o arquivo já exista
                if self.model_path.exists():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = self.model_path.parent / f"{self.model_path.stem}_backup_{timestamp}{self.model_path.suffix}"
                    self.model_path.rename(backup_path)
                    logger.info(f"Backup criado: {backup_path}")

                # Salva o modelo em um arquivo joblib
                joblib.dump(joblib_file, self.model_path)
                logger.info(f"Arquivo {self.model_path} criado.")

            except Exception as e:
                logger.error("❌ Erro ao carregar o arquivo!")
                raise IOError(e)
            return

        mensagem = "⚠️ O modelo não está treinado. Por favor, vá até Admin > Treinar Modelo para corrigir esse problema."
        logger.warning(mensagem)
        raise Warning(mensagem)

    def carregar_dados_csv(self) -> pd.DataFrame:
        """
        Carrega dados do arquivo CSV configurado
        
        Return: DataFrame com os dados brutos
        """
        return pd.read_csv(DATA_PATH)

    # -------------------------
    # Função responsável por criar target binária (y) com base nos valores da coluna "obesidade"
    # Será considerado obeso (valor=1) se os valores dessa coluna forem: "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"
    # -------------------------
    def criar_target_binario(self, df: pd.DataFrame) -> pd.Series:
        """
        Cria um target binário com base na coluna 'obesidade'.
        Considera obeso (1) se a coluna 'obesidade' contém: "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"
        """
        return df["obesidade"].isin({"Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"}).astype(int)

    def retornar_features_escolhidas(self) -> Dict:
        features = [
            'genero',
            'idade',
            'monitora_ingestao_calorica_bin',
            'fumante_bin',
            'frequencia_consumo_vegetais_n',
            'consumo_agua_diario_n',
            'numero_refeicoes_por_dia_n',
            'consumo_bebidas_alcoolicas_n',
            'tempo_uso_eletronicos_n',
            'consumo_lanches_entre_refeicoes_n',
            'index_ingestao_calorica',
            'balanco_caloria_atividade',
            'score_controle_index',
            'score_controle',
            'transporte_passivo',
            'risco_sedentarismo',
            'ingestao_x_sedentarismo',
            'score_controle_x_sedentarismo',
            'ingestao_x_inatividade',
            'tempo_eletronico_x_inatividade'
        ]

        return features

    def contruir_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """
        Constrói um Pipeline do scikit-learn com:
        1) Pré-processamento (numéricas + categóricas)


        Retorno
        -------
        Pipeline
            Pipeline com dois steps:
            - "prep": transformações de dados (numéricas + categóricas)
            - "model": modelo treinável
        """

        # ---------------
        # Identificar colunas categóricas e numéricas
        # ---------------
        # filtra o tipo objeto para achar as categóricas
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

        # qualquer outra coluna diferente das filtradas acima, entra como numérica
        num_cols = [c for c in X.columns if c not in cat_cols]

        # Construir o pré-processamento, onde colunas numéricas são escaladas e colunas categóricas são one-hot encoded, já que ainda
        # existem columnas que não foram normalizadas no dataset original
        preprocess = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                # OneHotEncoder gera matrizes muito densas; sparse economiza memória
                # Porém, alguns modelos não aceitam sparse (ex: HistGradientBoosting do sklearn),
                # por isso o parâmetro dense_for_model para focar o dense
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=not True), cat_cols), 
            ],
            remainder="drop",
        )

        model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)

        return Pipeline(steps=[("prep", preprocess), ("model", model)])

    def processar_dados(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Pré-processamento conforme dicionário
        df = renomear_colunas(df)

        df = arrendondar_limitar_valores_ordinais(df)
        df = normalizar_valores(df)

        # Feature engineering 100% comportamental, sem levar em conta variáveis antropométricas como IMC, peso, altura
        df = criar_score_ingestao_calorias(df)
        df = criar_score_controle(df)
        df = criar_features_comportamento(df)

        return df

    def treinar_modelo(self) -> Pipeline:
        """ Treinar modelo para criação de pipeline e salvar dados """

        df = self.carregar_dados_csv()
        df = self.processar_dados(df)

        y = self.criar_target_binario(df)

        # Criar X apenas com as features escolhidas
        X = df[self.retornar_features_escolhidas()]

        # Split estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # Treino
        pipeline = self.contruir_pipeline(X_train)
        pipeline.fit(X_train, y_train)

        joblib_file = {
              'pipeline': pipeline
            , 'dados_processados': df
            , 'X':X
            , 'y':y
            , 'X_train':X_train
            , 'X_test':X_test
            , 'y_train':y_test
            , 'y_test':y_test
            , 'config': {
                  'random_state':self.random_state
                , 'test_size':self.test_size
                , 'threshold':self.threshold
            }
        }

        # Salva o modelo em um arquivo joblib
        joblib.dump(joblib_file, self.model_path)

        return pipeline

    def validar_prob(self, dados = pd.DataFrame) -> Tuple[float,int]:
        
        if not self.is_trained:
            raise RuntimeError("❌ Modelo não está treinado")

        # validar se os dados passados por parametros já foram processados
        # vale notar que esses dados podem vir do sistema preditivo, sem as features
        # portanto, caso qualquer dessas features existam
        features = self.retornar_features_escolhidas()
        check_colunas = set(features) - set(dados)
        if check_colunas:
            logger.warning(f"⚠️ validar_prob(): dados recebidos com colunas faltando. Criando features.")
            dados = self.processar_dados(dados)
            logger.warning(f":✅ validar_prob(): dados processados -> features criadas.")
            #logger.info(dados)

        proba = float(self.pipe.predict_proba(dados)[:, 1])
        logger.info(f"✅ validar_prob(): proba criada: {proba} ")
        #logger.info(f"proba: {type(proba)} / {type(self.threshold)}")
        pred = int(proba >= float(self.threshold))
        logger.info(f"✅ validar_prob(): pred criado: {pred} ")

        return (proba, pred)

    def classificar_risco(self, proba) -> Dict[str,str]:
        """
        Classifica nível de risco baseado na probabilidade
        """
        dict_return = {}

        if proba >= self.threshold + 0.20: # >= 0.66
            dict_return = {
                'nivel': 'Alto Risco',
                'cor': '#dc3545', #vermelho
                'cor_fundo': '#f8d7da',
                'icone': '🔴',
                'mensagem': 'Alta probabilidade de obesidade detectada.',
                'acao': 'Recomendamos avaliação médica urgente.'
            }
        elif proba >= self.threshold + 0.10:  # 0.56-0.65
            dict_return = {
                'nivel': 'Risco Moderado-Alto',
                'cor': '#fd7e14',  # laranja
                'cor_fundo': '#fff3cd',
                'icone': '🟠',
                'mensagem': 'Probabilidade elevada de obesidade.',
                'acao': 'Considere consulta nutricional em breve.'
            }
        elif proba >= self.threshold:  # 0.46-0.55
            dict_return = {
                'nivel': 'Risco Moderado',
                'cor': '#ffc107',  # amarelo
                'cor_fundo': '#fff3cd',
                'icone': '🟡',
                'mensagem': 'Alguns indicadores de risco detectados.',
                'acao': 'Atenção ao estilo de vida e hábitos alimentares.'
            }
        elif proba >= self.threshold - 0.15:  # 0.31-0.45
            dict_return = {
                'nivel': 'Risco Baixo',
                'cor': '#28a745',  # verde
                'cor_fundo': '#d4edda',
                'icone': '🟢',
                'mensagem': 'Baixo risco no momento.',
                'acao': 'Mantenha seus hábitos saudáveis!'
            }
        else:  # < 0.31
            dict_return = {
                'nivel': 'Risco Muito Baixo',
                'cor': '#20c997',  # verde-água
                'cor_fundo': '#d1ecf1',
                'icone': '🟢',
                'mensagem': 'Risco muito baixo detectado.',
                'acao': 'Excelente! Continue com estilo de vida saudável.'
            }

        return dict_return
    

@st.cache_resource
def carregar_modelo(cache=True) -> Model:
    """Função que retorna a classe Model com cache """
    logger.info("✅ Inicializando Model em cache ")
    return Model(cache)