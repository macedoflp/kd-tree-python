import pandas as pd
import numpy as np
import heapq

# Configuração para exibir todas as linhas
pd.set_option('display.max_rows', None)

# Implementação de um Normalizador simples (StandardScaler)
class Normalizador:
    def __init__(self):
        self.media = None
        self.escala = None

    def ajustar(self, X):
        self.media = np.mean(X, axis=0)
        self.escala = np.std(X, axis=0)
        self.escala[self.escala == 0] = 1.0  # Evitar divisão por zero

    def transformar(self, X):
        return (X - self.media) / self.escala

    def ajustar_transformar(self, X):
        self.ajustar(X)
        return self.transformar(X)

# Classe para representar um nó da Árvore KD
class NoKD:
    def __init__(self, ponto, indice, eixo, esquerda, direita):
        self.ponto = ponto
        self.indice = indice
        self.eixo = eixo
        self.esquerda = esquerda
        self.direita = direita

# Função recursiva para construir a Árvore KD
def construir_arvore_kd(dados, indices, profundidade=0):
    if not indices:
        return None

    k = dados.shape[1]  
    eixo = profundidade % k  

    indices_ordenados = sorted(indices, key=lambda i: dados[i, eixo])
    indice_mediano = len(indices_ordenados) // 2
    indice_central = indices_ordenados[indice_mediano]

    return NoKD(
        ponto=dados[indice_central],
        indice=indice_central,
        eixo=eixo,
        esquerda=construir_arvore_kd(dados, indices_ordenados[:indice_mediano], profundidade + 1),
        direita=construir_arvore_kd(dados, indices_ordenados[indice_mediano + 1:], profundidade + 1)
    )

# Função para buscar os k vizinhos mais próximos na Árvore KD
def buscar_vizinhos(no, ponto_consulta, k, heap):
    if no is None:
        return

    distancia = np.sum((ponto_consulta - no.ponto) ** 2)

    if len(heap) < k:
        heapq.heappush(heap, (-distancia, no.indice))
    else:
        if distancia < -heap[0][0]:
            heapq.heappushpop(heap, (-distancia, no.indice))

    eixo = no.eixo
    diferenca = ponto_consulta[eixo] - no.ponto[eixo]

    primeiro, segundo = (no.esquerda, no.direita) if diferenca < 0 else (no.direita, no.esquerda)

    buscar_vizinhos(primeiro, ponto_consulta, k, heap)

    if len(heap) < k or diferenca**2 < -heap[0][0]:
        buscar_vizinhos(segundo, ponto_consulta, k, heap)

# Classe que encapsula a Árvore KD
class ArvoreKD:
    def __init__(self, dados):
        indices = list(range(dados.shape[0]))
        self.raiz = construir_arvore_kd(dados, indices)
    
    def consultar(self, pontos_consulta, k=1):
        todas_distancias = []
        todos_indices = []
        for ponto in pontos_consulta:
            heap = []
            buscar_vizinhos(self.raiz, ponto, k, heap)
            vizinhos = sorted([(-d, idx) for d, idx in heap])
            distancias = [d for d, idx in vizinhos]
            indices = [idx for d, idx in vizinhos]
            todas_distancias.append(distancias)
            todos_indices.append(indices)
        return np.array(todas_distancias), np.array(todos_indices)

# Função para carregar os dados
def carregar_dados(caminho_treino, caminho_validacao, caminho_resultado):
    return (
        pd.read_csv(caminho_treino),
        pd.read_csv(caminho_validacao),
        pd.read_csv(caminho_resultado)
    )

# Pré-processamento dos dados
def preprocessar_dados(df):
    df_proc = df.copy()
    df_proc['Sex'] = df_proc['Sex'].map({'male': 0, 'female': 1})
    df_proc['Embarked'] = df_proc['Embarked'].fillna('S').astype('category').cat.codes
    df_proc['Age'] = df_proc['Age'].fillna(df_proc['Age'].median())
    df_proc['Fare'] = df_proc['Fare'].fillna(df_proc['Fare'].median())
    return df_proc[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Função para obter predições
def obter_predicoes(X_treino_escalado, y_treino, X_validacao_escalado, k=3):
    arvore_kd = ArvoreKD(X_treino_escalado)
    _, indices = arvore_kd.consultar(X_validacao_escalado, k=k)
    
    predicoes = []
    for vizinhos in indices:
        voto = y_treino.iloc[vizinhos].mode().iloc[0]
        predicoes.append(voto)
    return np.array(predicoes)

def main():
    caminho_treino = "train.csv"
    caminho_validacao = "validation.csv"
    caminho_resultado = "result.csv"

    # Carrega os dados
    df_treino, df_validacao, df_resultado = carregar_dados(caminho_treino, caminho_validacao, caminho_resultado)

    # Pré-processa os dados
    X_treino = preprocessar_dados(df_treino)
    y_treino = df_treino['Survived']
    X_validacao = preprocessar_dados(df_validacao)

    # Converte para NumPy arrays
    X_treino_arr = X_treino.values
    X_validacao_arr = X_validacao.values

    # Normaliza os dados usando a classe customizada
    normalizador = Normalizador()
    X_treino_escalado = normalizador.ajustar_transformar(X_treino_arr)
    X_validacao_escalado = normalizador.transformar(X_validacao_arr)

    # Obtém predições
    predicoes = obter_predicoes(X_treino_escalado, y_treino, X_validacao_escalado, k=3)

    # Compara e exibe os resultados
    df_resultado['Previsto'] = predicoes
    df_resultado['Correto'] = df_resultado['Survived'] == df_resultado['Previsto']

    # Exibe todas as previsões
    print(df_resultado[['Survived', 'Previsto', 'Correto']].to_string(index=False))

    # Exibe acurácia
    acuracia = np.mean(df_resultado['Correto'])
    print(f"\nAcurácia: {acuracia * 100:.2f}%")

if __name__ == "__main__":
    main()
