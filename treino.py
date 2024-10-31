import json
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Parâmetros de controle
TEST_SIZE = .2
RANDOM_STATE = 0
MIN_DOC_FREQ = 1
N_COMPONENTS = 1000
N_ITER = 30
N_NEIGHBORS = 4
CV = 3

# Carregar dados do arquivo JSON e aplicar a filtragem
def load_and_filter_data(file_path, min_char_count=100):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Função para contar caracteres alfabéticos
    char_count = lambda text: len(re.sub(r'\W|\d', '', text))
    
    # Filtrar posts com pelo menos `min_char_count` caracteres
    filtered_data = [post for post in data if char_count(post['selftext']) >= min_char_count]
    
    texts = [post['selftext'] for post in filtered_data]
    labels = [post['search_query'] for post in filtered_data]  

    return texts, labels

# Dividir os dados em treino e teste
def split_data(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, 
                                                        test_size=TEST_SIZE, 
                                                        random_state=RANDOM_STATE)
    print(f"{len(y_test)} amostras de teste.")
    return X_train, X_test, y_train, y_test


# Pré-processamento e extração de atributos
def preprocessing_pipeline():
    pattern = r'\W|\d|http.*\s+|www.*\s+'
    preprocessor = lambda text: re.sub(pattern, ' ', text)
    vectorizer = TfidfVectorizer(preprocessor=preprocessor, stop_words='english', min_df=MIN_DOC_FREQ)
    decomposition = TruncatedSVD(n_components=N_COMPONENTS, n_iter=N_ITER)
    pipeline = [('tfidf', vectorizer), ('svd', decomposition)]
    return pipeline


# Seleção de modelos
def cria_modelos():
    modelo_1 = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    modelo_2 = RandomForestClassifier(random_state=RANDOM_STATE)
    modelo_3 = LogisticRegressionCV(cv=CV, random_state=RANDOM_STATE)
    modelos = [("KNN", modelo_1), ("RandomForest", modelo_2), ("LogReg", modelo_3)]
    return modelos

# Treinamento e avaliação dos modelos
def treina_avalia(modelos, pipeline, X_train, X_test, y_train, y_test):
    resultados = []
    for name, modelo in modelos:
        pipe = Pipeline(pipeline + [(name, modelo)])
        print(f"Treinando o modelo {name} com dados de treino...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        # Converta y_pred para uma lista para evitar problemas de serialização
        y_pred = y_pred.tolist()

        report = classification_report(y_test, y_pred)
        print(f"Relatório de Classificação para {name}\n", report)
        
        resultados.append({
            'modelo': name, 
            'previsoes': y_pred, 
            'y_test': y_test,  # y_test já é uma lista, então não precisamos converter
            'report': report
        })
    return resultados


# Executando o pipeline completo

if __name__ == "__main__":
    file_path = 'posts.json'  

    # Carregar e filtrar os dados
    data, labels = load_and_filter_data(file_path)

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = split_data(data, labels)

    # Obter o pipeline de pré-processamento e os modelos
    pipeline = preprocessing_pipeline()
    modelos = cria_modelos()

    # Treinar e avaliar os modelos
    resultados = treina_avalia(modelos, pipeline, X_train, X_test, y_train, y_test)


# Salvando resultados em um JSON 
with open('resultados.json', 'w') as f:
    json.dump(resultados, f)