import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -------------------------
# Função responsável por renomear colunas
# -------------------------
def renomear_colunas(df: pd.DataFrame) -> pd.DataFrame:
    colunas_renomeadas = {
        "Gender": "genero",
        "Age": "idade",
        "Height": "altura",
        "Weight": "peso",
        #"family_history": "historico_familiar",
        "FAVC": "frequencia_alimentos_caloricos",
        "FCVC": "frequencia_consumo_vegetais",
        "NCP": "numero_refeicoes_por_dia",
        "CAEC": "consumo_lanches_entre_refeicoes",
        "SMOKE": "fumante",
        "CH2O": "consumo_agua_diario",
        "SCC": "monitora_ingestao_calorica",
        "FAF": "frequencia_semanal_atividade_fisica",
        "TUE": "tempo_uso_eletronicos",
        "CALC": "consumo_bebidas_alcoolicas",
        "MTRANS": "meio_transporte_habitual",
        "Obesity": "obesidade"
    }
    df = df.rename(columns=colunas_renomeadas)
    return df

# -------------------------
# Função responsável por arredondar e limitar variáveis ordinais com ruído decimal
# Ex: frequencia_consumo_vegetais, numero_refeicoes_por_dia, consumo_agua_diario, frequencia_semanal_atividade_fisica, tempo_uso_eletronicos
# possuem valores decimais que não fazem sentido
# A ideia aqui é arredondar esses valores e limitar aos valores possíveis
# -------------------------
def arrendondar_limitar_valores_ordinais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Arredonda e limita variáveis ordinais com ruído decimal:
        frequencia_consumo_vegetais: 1..3
        numero_refeicoes_por_dia: 1..4
        consumo_agua_diario: 1..3
        frequencia_semanal_atividade_fisica: 0..3 
        tempo_uso_eletronicos: 0..2
    """
    df = df.copy()

    for c in ["frequencia_consumo_vegetais", "numero_refeicoes_por_dia",
              "consumo_agua_diario", "frequencia_semanal_atividade_fisica", 
              "tempo_uso_eletronicos"]:
        df[c] = np.rint(df[c]).astype(int)

    df["frequencia_consumo_vegetais"]          = df["frequencia_consumo_vegetais"].clip(1, 3)
    df["numero_refeicoes_por_dia"]             = df["numero_refeicoes_por_dia"].clip(1, 4)
    df["consumo_agua_diario"]                  = df["consumo_agua_diario"].clip(1, 3)
    df["frequencia_semanal_atividade_fisica"]  = df["frequencia_semanal_atividade_fisica"].clip(0, 3)
    df["tempo_uso_eletronicos"]                = df["tempo_uso_eletronicos"].clip(0, 2)

    # criar variável ordinal para family_history
    # df["historico_familiar_bin"] = df["historico_familiar"].map({"no": 0,"yes": 1})

    return df


# -------------------------
# Função responsável por normalizar valores 
# - Transforma booleanos em integer
# - Transforma categoricos em integer
# - Normaliza numéricos entre 0..1
# -------------------------
def normalizar_valores(df: pd.DataFrame) -> pd.DataFrame:
    
    # Converter para variável numérica (binária)
    df["monitora_ingestao_calorica_bin"]     = (df["monitora_ingestao_calorica"] == "yes").astype(int)
    df["frequencia_alimentos_caloricos_bin"] = (df["frequencia_alimentos_caloricos"] == "yes").astype(int)
    df["fumante_bin"]                        = (df["fumante"] == "yes").astype(int)

    bebidas = df["consumo_bebidas_alcoolicas"].map({
        "no": 0,
        "Sometimes": 1,
        "Frequently": 2,
        "Always": 3
    }).fillna(1).astype(int)
    
    lanches = df["consumo_lanches_entre_refeicoes"].map({
        "no": 0,
        "Sometimes": 1,
        "Frequently": 2,
        "Always": 3
    }).fillna(0).astype(int)

    # Normalizações 0..1
    df["frequencia_consumo_vegetais_n"]         = (df["frequencia_consumo_vegetais"] - 1) / 2
    df["consumo_agua_diario_n"]                 = (df["consumo_agua_diario"] - 1) / 2
    df["frequencia_semanal_atividade_fisica_n"] = df["frequencia_semanal_atividade_fisica"] / 3
    df["numero_refeicoes_por_dia_n"]            = (df["numero_refeicoes_por_dia"] - 1) / 3
    df["consumo_bebidas_alcoolicas_n"]          = bebidas / 3
    df["tempo_uso_eletronicos_n"]               = df["tempo_uso_eletronicos"] / 2
    df["consumo_lanches_entre_refeicoes_n"]     = lanches / 3

    return df

# -------------------------
# Existe uma variável que captura a frequencia de consumo de alimentos calóricos (FAVC)
# Porém, apenas essa variável por si só não determina se um indivíduo será obeso ou não.
# É necessário considerar outros fatores comportamentais que influenciam a ingestão calórica total
# 
# Essa função é responsável por criar um score de ingestão calórica baseado em variáveis comportamentais, combinando:
#   - hábitos alimentares,
#   - padrão de refeições,
#   - consumo de vegetais (fator protetor),
#   - histórico familiar,
#   - nível de atividade física.
# Consumo de calorias está diretamente relacionado ao comportamento alimentar do indivíduo, 
# sendo um dos principais fatores para o desenvolvimento da obesidade.
# -------------------------
def criar_score_ingestao_calorias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy de carga calórica com base em:
    - frequencia_alimentos_caloricos (yes/no)
    - consumo_lanches_entre_refeicoes (no/Sometimes/Frequently/Always)
    - número de refeições por dia (1..4)
    - frequencia_consumo_vegetais (proteção)
    """
    df = df.copy()

    # Índice linear de ingestão calórica
    df["index_ingestao_calorica"] = (
        # Como a ingestão calórica é fortemente impactada pelo consumo frequente de alimentos calóricos,
        # esse é o principal fator, recebendo um peso maior na composição do índice
        1.2 * df["frequencia_alimentos_caloricos_bin"]

        # Comer entre refeições também impacta significativamente a ingestão calórica total
        # Por isso ele é o segundo fator mais importante
        + 0.9 * df["consumo_lanches_entre_refeicoes_n"]

        # Número de refeições pode influenciar a ingestão calórica total, mas tem um peso menor pois de acordo com as explorações
        # anteriores, o número de refeições não tem uma correlação tão forte com obesidade
        # Essa variável é um fator contextualizador para o cálculo do índice, por isso tem um peso menor
        + 0.4 * df["numero_refeicoes_por_dia_n"]

        # Consumo de vegetais tem efeito protetor, reduzindo a ingestão calórica total consumida nas refeições
        - 0.8 * df["frequencia_consumo_vegetais_n"]
    )

    # Sigmoide para score 0..1
    df["score_ingestao_calorica"] = 1 / (1 + np.exp(-df["index_ingestao_calorica"]))

    # A ideia aqui é capturar o efeito potencializado do histórico familiar (geneticamente predisposto) com a ingestão calórica 
    # df["risco_genetico"] = df["historico_familiar_bin"] * df["score_ingestao_calorica"]

    # Balancear ingestão calórica com nível de atividade física
    df["balanco_caloria_atividade"] = df["score_ingestao_calorica"] / (df["frequencia_semanal_atividade_fisica"] + 1)

    return df

# -------------------------
# Função responsável por criar um score de "perfil controlado/metabólico"
# A ideia é capturar o quão saudável/metabólico é o perfil do indivíduo com base em fatores comportamentais
# A combinação desses fatores pode indicar um perfil mais controlado/metabólico, reduzindo o risco de obesidade
# -------------------------
def criar_score_controle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score de "perfil controlado/metabólico" (0..1) com base em:
    + monitora_ingestao_calorica
    + frequencia_consumo_vegetais
    + consumo_agua_diario
    + frequencia_semanal_atividade_fisica
    + numero_refeicoes_por_dia
    - consumo_bebidas_alcoolicas
    - fumante
    - tempo_uso_eletronicos (sedentarismo)
    """
    df = df.copy()

    # Pesos heurísticos (monotônicos)
    df["score_controle_index"] = (
          1.5 * df["monitora_ingestao_calorica_bin"]
        + 1.2 * df["frequencia_consumo_vegetais_n"]
        + 1.0 * df["consumo_agua_diario_n"]
        + 1.3 * df["frequencia_semanal_atividade_fisica_n"]
        + 0.6 * df["numero_refeicoes_por_dia_n"]
        - 1.2 * df["consumo_bebidas_alcoolicas_n"]
        - 0.3 * df["fumante_bin"]
        - 0.3 * df["tempo_uso_eletronicos_n"]
    )

    # Normalizar 0..1
    df["score_controle"] = 1 / (1 + np.exp(-df["score_controle_index"]))

    return df

# -------------------------
# Função responsável por criar engineering features com base no comportamento do indivíduo,
# cruzando features entre si, a fim de determinar comportamentos que possam determinar obesidade
# -------------------------
def criar_features_comportamento(df: pd.DataFrame) -> pd.DataFrame:
    """
    Criação de features para capturar interações entre variáveis comportamentais
    relevantes para obesidade, sem usar antropometria (peso, altura, IMC, etc)

    Ideia central
    -------------
      - Riscos costumam aparecer em combinação de fatores e não em variáveis isoladas
      - Por isso, criar features que capturem essas interações pode ajudar o modelo a identificar padrões mais complexos

    Cuidados
    --------
        - Evitar criar muitas features que possam levar a overfitting
        - Focar em interações que façam sentido do ponto de vista comportamental e de saúde
        - Colunas precisam ser normalizadas previamente (0..1) para quem os cálculos sejam na mesma escala ou não explodam em magnitude
    """
    df = df.copy()

    # Inatividade física 
    #   Pega-se a freq. semanal de atividade fisica normalizada e inverte-se
    #   Esse é uma das principais features, já que freq. de atividade física é a variável com maior correlação com obesidade,
    #   de acordo com a Análise Exploratória de Dados realizada no arquivo exploiration.ipynb
    df["inatividade"] = 1 - df["frequencia_semanal_atividade_fisica_n"]

    # Identificar quem é mais ativo em relação ao tipo de maio de transporte (caminhada ou bicicleta)
    # De acordo com a exploração, esse grupo tem menor prevalência de obesidade 
    #   Walking: 5.4%
    #   Bike: 14.3%
    #   Motorbike: 27.3%
    #   Automobile: 45.1%
    #   Public_Transportation: 48.0%
    df["transporte_passivo"] = df["meio_transporte_habitual"].isin({"Motorbike", "Automobile", "Public_Transportation"}).astype(int)

    # Sedentarismo: média entre inatividade física e uso de transporte passivo
    #   Esse score tenta capturar uma rotina mais passiva/sedentária da pessoa, combinando esses dois fatores
    #   A média foi escolhida para balancear os dois aspectos igualmente, mantendo o score entre 0 e 1
    #   Interpretação: 
    #    0   = mais ativo (atividade física frequente e transporte ativo) 
    #    0.5 = um fator ruim e o outro bom
    #    1   = mais sedentário (atividade física rara e transporte passivo)
    df["risco_sedentarismo"] = (df["inatividade"] + df["transporte_passivo"]) / 2
    
    # Alta ingestão calórica combinada com risco de sedentarismo
    #   Essa interação cresce quando há uma alta ingestão calórica e uma rotina mais sedentária
    #   A combinação de "comer muito e mover-se pouco" é mais informativa do que olhar para uma variável isoladamente
    df["ingestao_x_sedentarismo"] = df["index_ingestao_calorica"] * df["risco_sedentarismo"]
    
    # Score de controle/metabólico combinado com sedentarismo
    #   A ideia aqui é capturar o efeito potencializado de um perfil comportamental menos saudável (baixo score de controle) com uma rotina sedentária
    #   Essa interação pode indicar um risco aumentado de obesidade, já que tanto o perfil comportamental quanto o sedentarismo contribuem negativamente
    df["score_controle_x_sedentarismo"] = df["score_controle"] * df["risco_sedentarismo"]

    # Histórico familiar combinado com inatividade física
    #   Pré-disposição genética pode se manifestar com mais força em um estilo de vida mais sedentário
    #   Se historico_familiar_bin = 0, essa feature zera
    # df["genetica_x_inatividade"] = df["historico_familiar_bin"] * df["inatividade"]

    # Ingestão calórica X Inatividade
    #   Feature que valida diretamente a associação entre uma alta ingestão calórica com a inatividade isoladamente (sem misturar transporte)
    df["ingestao_x_inatividade"] = df["index_ingestao_calorica"] * df["inatividade"]

    # Tempo de uso eletronico x Inatividade
    #   Caso o tempo de tela seja alto e a inatividade também, isso pode potencializar o sedentarismo, por mais que o TUE esteja mais relacionado com
    #   a idade do que de fato a falta de atividade
    df["tempo_eletronico_x_inatividade"] = df["tempo_uso_eletronicos_n"] * df["inatividade"]

    # Tempo de uso eletronico x Risco Sedentarismo
    #   Similar ao anterior, porém validando o TUE com o Risco de sedentarismo
    df["tempo_eletronico_x_inatividade"] = df["tempo_uso_eletronicos_n"] * df["risco_sedentarismo"]

    return df


# -------------------------
# Função responsável por criar target binária (y) com base nos valores da coluna "obesidade"
# Será considerado obeso (valor=1) se os valores dessa coluna forem: "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"
# -------------------------
def criar_target_binario(df: pd.DataFrame) -> pd.Series:
    """
    Cria um target binário com base na coluna 'obesidade'.
    """
    return df["obesidade"].isin({"Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"}).astype(int)

def retornar_features_escolhidas() -> dict:
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