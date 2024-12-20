# Machine-Learning-Classification
## O que é Apredizado de Maquina?
Apredizado de Maquina ou Machine Learning como a utilização de algoritmos com a finalidade de extrair informações de dados brutos e representá-los por meio de algum tipo de modelo matemático. Este modelo é então usado para fazer inferências — ou predições — a partir de novos conjuntos de dados.

Em Machine Learning, o aprendizado é o objetivo principal. Aprendizado é a capacidade de se adaptar, modificar e melhorar seu comportamento e suas respostas, sendo uma das propriedades mais importantes dos seres inteligentes (humanos ou não). Diz-se que se está aprendendo (treinando, construindo, formulando ou induzindo um modelo de conhecimento) a partir de um conjunto de dados quando se procura por padrões nestes dados.

## Aprendizado Supervisionado 

**1-Aprendizado Supervisionado** 

o modelo (ou algoritmo) é construído a partir dos dados de entrada (também chamados de dataset), que são apresentados na forma de pares ordenados (entrada — saída desejada).

São exemplos de problemas de aprendizado supervisionado a **Classificação** e a **Regressão**, que serão detalhados a seguir.

Dizemos que estes dados são rotulados, pois sabemos de antemão a saída esperada para cada entrada de dados. Neste caso, o aprendizado (ou treinamento) consiste em apresentarmos para o algoritmo um número suficiente de exemplos (também chamados de registros ou instâncias) de entradas e saídas desejadas (já rotuladas previamente). Assim, o objetivo do algoritmo é aprender uma regra geral que mapeie as entradas nas saídas corretamente, o que consiste no modelo final.

Os dados de entrada podem ser divididos em dois grupos:

   * X, com os atributos (também chamados de características) a serem utilizados na determinação da classe de saída (também chamados de atributos previsores ou de predição)

   * Y, com o atributo para o qual se deseja fazer a predição do valor de saída categórico ou numérico (também chamado de atributo-alvo ou target).


  ![supervisionado](pic\supervisionado.png)


## Qual o Problema deve ser resolvido ?
É muito importante que entendamos bem o problema a ser resolvido para que possamos traçar os objetivos principais. Em seguida, será necessário coletar e analisar os dados adequados para o problema e prepará-los, pois na maioria das vezes eles virão com informações faltantes, incompletas ou inconsistentes. Após estas etapas é que podemos construir o modelo de Machine Learning, que deve ser avaliado e criticado e, se necessário voltar à etapa de coleta e análise de dados, para a obtenção de mais dados, ou mesmo retornar à etapa de construção do modelo, usando diferentes estratégias.

## Etapas que devem ser Seguindas em Projeto de M.L
   * 1-Entender o problema e definir objetivos — Que problema estou resolvendo?

   * 2-Coletar e analisar os dados — De que informações preciso?

   * 3-Preparar os dados — Como preciso tratar os dados?

   * 4-Construir o modelo — Quais são os padrões nos dados que levam a soluções?

   * 5-Avaliar e criticar o modelo — O modelo resolve meu problema?

   * 6-Apresentar resultados — Como posso resolver o problema?

   * 7-Distribuir o modelo — Como resolver o problema no mundo real?

   ![Levantamento dos Requisitos](/pic/levatamento-requisitos.png)

## Afinal o que é um Modelo de Classificação?
Ao explorar o vasto universo do aprendizado de máquina, identificamos uma técnica fundamental para resolver diversos problemas: a classificação. E por meio do modelo de M.L de classificação que precisamos resolver um problema do dia a dia que categorizar aos padrões de dados . Por exemplo  presciso categorizar os clientes de uma indústria de calçados em : clientes Inadimplentes ou clientes Adimplentes 

### Categorizando os Dados
É o processo de entender, reconhecer padrões e agrupar o conjunto de dados em categorias, com a ajuda de dados de treino pré-categorizados, de maneira que seja possível determinar quais rótulos serão aplicados em dados não observados.

* **Exemplo**:
Imagine que você está montando uma playlist de músicas para uma festa e precisa escolher três estilos musicais diferentes: rock, sertanejo e eletrônica. Nesse caso, você precisa conhecer as características de cada um desses estilos para criar uma playlist que seus convidados gostem.

Você já ouviu esses estilos antes e consegue identificar os padrões únicos de cada um deles. Essa habilidade de identificar os padrões é semelhante à classificação de dados em Machine Learning, como categorizar o gênero musical de uma canção com base em suas características.

 ![Entendendo Padrões](/pic/padroes.png)

 Na classificação, esse mesmo princípio acontece, pois precisamos saber antecipadamente quais são as respostas corretas para aprendermos os atributos que caracterizam cada categoria

## Algoritmos de classificação
Os algoritmos de classificação são essenciais para categorizar dados, pois por meio deles podemos compreender e responder a questões específicas em diversos contextos e desafios. Exemplos:

* Naive Bayes: classifica dados com base em probabilidades condicionais.

* Redes Neurais Artificiais: reconhece padrões e processamento de linguagem natural inspirado no funcionamento do cérebro humano.

* Support Vector Machines (SVM): mapeia dados em um espaço multidimensional e encontra um hiperplano de separação.

* Regressão Logística: usada principalmente para problemas de classificação binária com intuito de estimar a probabilidade de pertencer a uma das duas classes possíveis.

* Árvore de Decisão: classifica novos dados seguindo um conjunto de regras.
Random Forest: utiliza várias árvores de decisão em conjunto para melhorar a precisão da classificação e reduzir o overfitting.

* Gradient Boosting: constrói um modelo forte combinando vários modelos fracos de forma iterativa, minimizando os erros anteriores.

* K-Nearest Neighbors: faz previsões com base na maioria dos k pontos de dados mais próximos ao ponto de consulta

## Categorias da Classificação
Os algoritmos de classificação podem ser divididos em três categorias principais: classificação binária, classificação multiclasse e classificação multirrótulo .

### Classificação binária
Na classificação binária, a ideia é classificar os dados em apenas duas categorias. Os dados são rotulados de forma binária, por exemplo: verdadeiro ou falso, 0 ou 1, spam ou não spam, etc.

**Exemplos de algoritmos de classificação binária são:**

    1-Naive Bayes
    2-Redes Neurais Artificiais
    3-Regressão Logística
    3-Support Vector Machines
    4-Árvore de Decisão

![classificação binária](/pic/classificacao-binaria.png)

### Classificação multiclasse
Na classificação multiclasse, os dados são classificados em pelo menos duas categorias. A ideia é descobrir a qual das duas ou mais categorias o dado pertence. 
**Exemplos de algoritmos de classificação multiclasse são:**

    1-Naive Bayes
    2-Regressão Logística
    3-K-Nearest Neighbors
    3-Support Vector Machines
    4-Gradient Boosting

![classificação binária](/pic/multiclasse.png)

**Como exemplo, podemos adicionar mais uma categoria ao exemplo anterior, ficando com três categorias: cachorro, gato e pássaro.**

### Classificação multirrótulo
Na classificação multirrótulo, os dados são classificados em 0 ou mais categorias. Nesse caso, um mesmo dado pode ser rotulado em várias categorias .
**Exemplos de algoritmos de classificação multirrótulo são:**
      
   * Árvore de Decisão multirrótulo
   * Gradient Boosting multirrótulo
   * Random Forest multirrótulo

![classificação binária](/pic/multirrótulo.png)

## Referencias de Consultas 
1- (Tatiana Escovedo
)https://tatianaesc.medium.com/machine-learning-conceitos-e-modelos-f0373bf4f445

2-(Problemas resolvidos com classificação-Alura)
https://www.alura.com.br/artigos/problemas-resolvidos-algoritmos-classificacao

# Reddit Data Extraction and Machine Learning Classification

## Descrição do Projeto

Este projeto realiza a extração de dados do Reddit utilizando sua API oficial e faz um estudo de modelos de classificação de machine learning com os dados coletados. O objetivo é analisar e classificar os dados com base em diferentes variáveis e construir um modelo preditivo eficiente.

O projeto foi construído utilizando o [Poetry](https://python-poetry.org/) para gerenciar as dependências e o ambiente virtual.

## Estrutura do Projeto

- **Data Extraction**: Extração de dados da API do Reddit.
- **Data Preprocessing**: Pré-processamento dos dados extraídos para adequá-los ao modelo de machine learning.
- **Model Training**: Treinamento de modelos de classificação de machine learning.
- **Model Evaluation**: Avaliação da performance dos modelos utilizando métricas apropriadas.

## Instalação

Certifique-se de que você tenha o [Poetry](https://python-poetry.org/) instalado no seu sistema. Siga as instruções abaixo para configurar o projeto:

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-projeto.git
