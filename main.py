import pandas as pd

# Classe que representa um nó da KD tree
class Node:
    def __init__(self, point, axis, left, right):
        self.point = point    # O ponto armazenado neste nó (lista de valores)
        self.axis = axis      # A dimensão utilizada para dividir os pontos neste nó
        self.left = left      # Subárvore esquerda (pontos menores na dimensão 'axis')
        self.right = right    # Subárvore direita (pontos maiores na dimensão 'axis')

# Função para construir a KD tree recursivamente
def build_kdtree(points, depth=0):
    if not points:
        return None

    # Número de dimensões dos pontos (neste caso, 5)
    k = len(points[0])
    # Seleciona a dimensão de corte (rotação entre 0 e k-1)
    axis = depth % k

    # Ordena os pontos de acordo com o valor na dimensão 'axis'
    points.sort(key=lambda point: point[axis])
    median_index = len(points) // 2

    # Cria o nó com o ponto mediano e constrói recursivamente as subárvores
    return Node(
        point=points[median_index],
        axis=axis,
        left=build_kdtree(points[:median_index], depth + 1),
        right=build_kdtree(points[median_index + 1:], depth + 1)
    )

# Função auxiliar para imprimir a árvore (opcional)
def print_tree(node, depth=0):
    if node is not None:
        print("  " * depth + f"Nó (axis {node.axis}): {node.point}")
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)

def main():
    # Lê o arquivo CSV contendo os dados
    df = pd.read_csv('train.csv')
    
    # Seleciona os atributos numéricos a partir do dicionário de dados:
    # pclass: classe do bilhete (1, 2, 3)
    # Age: idade em anos
    # sibsp: número de irmãos/cônjuges a bordo
    # parch: número de pais/filhos a bordo
    # fare: tarifa paga
    df_numeric = df[['pclass', 'Age', 'sibsp', 'parch', 'fare']].dropna()
    
    # Converte os dados para uma lista de pontos (cada ponto é uma lista de 5 números)
    points = df_numeric.values.tolist()
    
    # Constrói a KD tree a partir dos pontos
    tree = build_kdtree(points)
    
    # Exibe a árvore (apenas para visualização; pode ser omitido se preferir)
    print_tree(tree)

if __name__ == "__main__":
    main()
