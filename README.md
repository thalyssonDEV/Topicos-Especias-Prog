# Atividade Prática: Tuning de Hiperparâmetros e Comparação de Modelos

Projeto desenvolvido para a disciplina de **Tópicos Especiais em Programação** do curso de Análise e Desenvolvimento de Sistemas do IFPI.

## 🎯 Objetivo

O objetivo deste projeto é aplicar e comparar técnicas de ajuste de hiperparâmetros (Tuning) em diferentes modelos de classificação supervisionada. A análise compara o desempenho e o custo computacional de cada técnica.

* **Modelos Utilizados**: `LogisticRegression`, `SVC`, `RandomForestClassifier`
* **Bases de Dados**: `load_wine()`, `load_digits()`, `load_breast_cancer()`
* **Técnicas de Tuning**: `GridSearchCV`, `RandomizedSearchCV`, `BayesSearchCV`

## 🛠️ Requisitos

O script utiliza Python 3. As bibliotecas necessárias podem ser instaladas via `pip`:

```bash
pip install pandas scikit-learn scikit-optimize tabulate
```

## 🚀 Execução

Para executar os testes e imprimir as tabelas de resultados no console, execute o script principal:
```bash
python3 main.py
```

## 📊 Análise Conclusiva

A análise dos resultados indica que não há um "melhor modelo" universal; o desempenho dependeu do dataset. O SVM foi superior no 'Digits' (Acurácia 0.9566), enquanto a Regressão Logística e o SVM se destacaram nos datasets 'Wine' e 'Breast Cancer'. A técnica de tuning mais eficiente foi o RandomizedSearchCV, que entregou métricas de performance quase idênticas ao GridSearchCV (ex: 0.9560 no 'Digits, SVM'), porém com um custo computacional muito menor (11.80s vs 24.86s). Observou-se que os datasets 'Wine' e 'Breast Cancer' foram simples, com os modelos baseline já atingindo acurácias de ~98%, tornando o tuning quase irrelevante. Em contraste, o 'Digits' foi o mais complexo, onde o tuning se mostrou mais impactante, elevando a performance dos modelos.

## Tabelas Comparativas

![Imagem1](caminho/para/imagem.png)
![Imagem2](caminho/para/imagem2.png)
