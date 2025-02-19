import pandas as pd
import numpy as np
import heapq

# Implementação de um StandardScaler simples
class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Evitar divisão por zero
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Classe que representa um nó da KD-Tree
class KDNode:
    def __init__(self, point, index, axis, left, right):
        self.point = point      # Ponto (features) armazenado no nó
        self.index = index      # Índice do ponto no conjunto original
        self.axis = axis        # Eixo (dimensão) utilizado para a divisão
        self.left = left        # Subárvore esquerda
        self.right = right      # Subárvore direita

# Função recursiva para construir a KD-Tree
def build_kd_tree(data, indices, depth=0):
    if not indices:
        return None

    k = data.shape[1]            # Número de dimensões
    axis = depth % k             # Seleciona o eixo com base na profundidade

    # Ordena os índices com base no valor da feature no eixo atual
    sorted_indices = sorted(indices, key=lambda i: data[i, axis])
    median_idx = len(sorted_indices) // 2
    median_index = sorted_indices[median_idx]

    # Cria o nó atual e constrói recursivamente as subárvores
    return KDNode(
        point=data[median_index],
        index=median_index,
        axis=axis,
        left=build_kd_tree(data, sorted_indices[:median_idx], depth + 1),
        right=build_kd_tree(data, sorted_indices[median_idx + 1:], depth + 1)
    )

# Função recursiva para buscar os k vizinhos mais próximos na KD-Tree
def kd_tree_search(node, query_point, k, heap):
    if node is None:
        return

    # Calcula a distância Euclidiana ao quadrado
    dist = np.sum((query_point - node.point) ** 2)

    # Se o heap não estiver cheio, adiciona o ponto; se estiver, substitui se o ponto for mais próximo
    if len(heap) < k:
        heapq.heappush(heap, (-dist, node.index))
    else:
        if dist < -heap[0][0]:
            heapq.heappushpop(heap, (-dist, node.index))

    axis = node.axis
    diff = query_point[axis] - node.point[axis]

    # Decide qual subárvore explorar primeiro
    if diff < 0:
        first, second = node.left, node.right
    else:
        first, second = node.right, node.left

    kd_tree_search(first, query_point, k, heap)

    # Se a distância ao plano de divisão for menor que a pior distância no heap, explora a outra subárvore
    if len(heap) < k or diff**2 < -heap[0][0]:
        kd_tree_search(second, query_point, k, heap)

# Classe que encapsula a construção e consulta da KD-Tree
class KDTreeMy:
    def __init__(self, data):
        self.data = data
        indices = list(range(data.shape[0]))
        self.root = build_kd_tree(data, indices)
    
    def query(self, query_points, k=1):
        all_distances = []
        all_indices = []
        for qp in query_points:
            heap = []
            kd_tree_search(self.root, qp, k, heap)
            # Converte o heap (com distâncias negativas) para valores positivos
            neighbors = sorted([(-d, idx) for d, idx in heap])
            distances = [d for d, idx in neighbors]
            indices = [idx for d, idx in neighbors]
            all_distances.append(distances)
            all_indices.append(indices)
        return np.array(all_distances), np.array(all_indices)

# Função para carregar os dados dos arquivos CSV
def carregar_dados(caminho_treino, caminho_validacao, caminho_resultado):
    df_treino = pd.read_csv(caminho_treino)
    df_validacao = pd.read_csv(caminho_validacao)
    df_resultado = pd.read_csv(caminho_resultado)
    return df_treino, df_validacao, df_resultado

# Função para pré-processar os dados:
# - Converte 'Sex' para numérico;
# - Preenche missing em 'Embarked' e codifica;
# - Preenche missing em 'Age' e 'Fare' com a mediana;
# - Seleciona as features relevantes
def preprocessar_dados(df):
    df_proc = df.copy()
    df_proc['Sex'] = df_proc['Sex'].map({'male': 0, 'female': 1})
    df_proc['Embarked'] = df_proc['Embarked'].fillna('S').astype('category').cat.codes
    df_proc['Age'] = df_proc['Age'].fillna(df_proc['Age'].median())
    df_proc['Fare'] = df_proc['Fare'].fillna(df_proc['Fare'].median())
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    return df_proc[features]

# Função para obter as predições utilizando kNN com a KD-Tree implementada
def obter_predicoes(X_treino_scaled, y_treino, X_validacao_scaled, k=3):
    kd_tree = KDTreeMy(X_treino_scaled)
    _, indices = kd_tree.query(X_validacao_scaled, k=k)
    
    predicoes = []
    for vizinhos in indices:
        # Votação majoritária dos vizinhos
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
    
    # Converte os DataFrames para arrays NumPy
    X_treino_arr = X_treino.values
    X_validacao_arr = X_validacao.values
    
    # Normaliza os dados utilizando nosso CustomStandardScaler
    scaler = CustomStandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino_arr)
    X_validacao_scaled = scaler.transform(X_validacao_arr)
    
    # Obtém as predições utilizando kNN (k=3)
    predicoes = obter_predicoes(X_treino_scaled, y_treino, X_validacao_scaled, k=3)
    
    # Calcula a acurácia comparando com os resultados esperados
    acuracia = np.mean(predicoes == df_resultado['Survived'])
    print(f"Acurácia: {acuracia * 100:.2f}%")

if __name__ == '__main__':
    main()
