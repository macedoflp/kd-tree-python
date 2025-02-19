import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

# Carregar os dados
train = pd.read_csv("train.csv")
validation = pd.read_csv("validation.csv")
result = pd.read_csv("result.csv")

# Pré-processamento dos dados
def preprocess(df):
    # Converter 'Sex' para numérico
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Codificar 'Embarked' (preencher missing com 'S' e converter para códigos)
    df['Embarked'] = df['Embarked'].fillna('S').astype('category').cat.codes
    
    # Preencher valores faltantes em 'Age' e 'Fare'
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Selecionar features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    return df[features]

# Pré-processar dados de treino e validação
X_train = preprocess(train)
y_train = train['Survived']
X_val = preprocess(validation)

# Normalizar os dados (importante para KD Tree)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Construir a KD Tree
kd_tree = KDTree(X_train_scaled, leaf_size=40)

# Buscar vizinhos mais próximos (k=3)
distances, indices = kd_tree.query(X_val_scaled, k=3)

# Prever sobrevivência (votação majoritária dos vizinhos)
predictions = np.array([y_train.iloc[indices[i]].mode()[0] for i in range(len(X_val))])

# Comparar com os resultados esperados
accuracy = np.mean(predictions == result['Survived'])
print(f"Acurácia: {accuracy * 100:.2f}%")