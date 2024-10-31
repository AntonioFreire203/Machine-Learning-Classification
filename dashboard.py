import streamlit as st
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Configurações do Streamlit
st.title("Análise e Classificação de Posts do Reddit")

# Carregar dados para visualização
with open('resultados.json', 'r') as f:
    resultados = json.load(f)

# Função para exibir a distribuição de rótulos
def plot_distribution(labels):
    _, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots()
    sns.barplot(x=np.unique(labels), y=counts, ax=ax)
    ax.set_title("Distribuição dos Assuntos")
    st.pyplot(fig)

# Função para exibir matriz de confusão
def plot_confusion(result):
    st.write(f"### {result['modelo']} - Matriz de Confusão")
    y_pred = result['previsoes']
    y_test = result['y_test']
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predição")
    ax.set_ylabel("Real")
    st.pyplot(fig)

# Interface Streamlit
if 'KNN' in resultados:
    st.header("Resultados de Classificação")
    for result in resultados:
        plot_confusion(result)
