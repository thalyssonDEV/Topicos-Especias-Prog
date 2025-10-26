import pandas as pd
import numpy as np
import time
import warnings

# Imports dos Datasets
from sklearn.datasets import load_wine, load_digits, load_breast_cancer

# Imports dos Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Imports de Tuning e Avaliação
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

# BayesSearchCV (scikit-optimize)
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    print("Biblioteca scikit-optimize não encontrada. BayesSearchCV será pulado.")
    print("Instale com: pip install scikit-optimize")
    SKOPT_AVAILABLE = False

# Suprimir warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Configurações Globais ---
datasets_to_run = [
    ("Wine", load_wine),
    ("Digits", load_digits),
    ("Breast Cancer", load_breast_cancer)
]

# Modelos e seus Hiperparâmetros de busca
models_config = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=2000),
        # Grid para GridSearchCV e RandomizedSearchCV
        {
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['saga'] # <-- MUDANÇA AQUI
        },
        # Espaço para BayesSearchCV
        {
            'model__C': Real(0.01, 100, 'log-uniform'),
            'model__penalty': Categorical(['l1', 'l2']),
            'model__solver': Categorical(['saga']) # <-- MUDANÇA AQUI
        }
    ),
    (
        "SVM",
        SVC(probability=True),
        # Grid para GridSearchCV e RandomizedSearchCV
        {
            'model__C': [0.1, 1, 10, 100],
            'model__kernel': ['linear', 'rbf', 'poly'],
            'model__gamma': ['scale', 'auto', 0.01, 0.1, 1]
        },
        # Espaço para BayesSearchCV
        {
            'model__C': Real(0.1, 100, 'log-uniform'),
            'model__kernel': Categorical(['linear', 'rbf', 'poly']),
            'model__gamma': Real(1e-3, 1, 'log-uniform')
        }
    ),
    (
        "Random Forest",
        RandomForestClassifier(random_state=42),
        # Grid para GridSearchCV e RandomizedSearchCV
        {
            'model__n_estimators': [50, 100, 200, 300],
            'model__max_depth': [None, 10, 20, 30, 40],
            'model__min_samples_split': [2, 5, 10]
        },
        # Espaço para BayesSearchCV
        {
            'model__n_estimators': Integer(50, 300),
            'model__max_depth': Integer(10, 40),
            'model__min_samples_split': Integer(2, 10)
        }
    )
]

scoring_metrics = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall': make_scorer(recall_score, average='weighted', zero_division=0),
    'f1': make_scorer(f1_score, average='weighted', zero_division=0)
}

CV = 5
N_ITER_RANDOM_BAYES = 30

# --- 2. Função Auxiliar para Armazenar Resultados ---
def store_results(results_list, dataset, model, tuner, scores, duration_s):
    results_list.append({
        "Dataset": dataset,
        "Model": model,
        "Tuner": tuner,
        "Time (s)": duration_s,
        "Accuracy": np.mean(scores['test_accuracy']),
        "Precision": np.mean(scores['test_precision']),
        "Recall": np.mean(scores['test_recall']),
        "F1": np.mean(scores['test_f1'])
    })

# --- 3. Execução dos Experimentos ---
all_results = []
print(f"Iniciando 9 experimentos (3 datasets x 3 modelos)...")
print(f"Técnicas: Baseline, Grid, Randomized, Bayes. CV={CV}.")

for data_name, data_loader in datasets_to_run:
    X, y = data_loader(return_X_y=True)
    
    print(f"\n--- Processando Dataset: {data_name} ---")

    for model_name, model_instance, param_grid, bayes_space in models_config:
        
        print(f"  Modelo: {model_name}")

        if model_name == "Random Forest":
            pipeline = Pipeline([('model', model_instance)])
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model_instance)
            ])

        # --- a. Modelo sem tuning (Baseline) ---
        print("    Avaliando: Baseline (Padrão)")
        start_time = time.time()
        baseline_scores = cross_validate(pipeline, X, y, cv=CV, scoring=scoring_metrics)
        duration = time.time() - start_time
        store_results(all_results, data_name, model_name, "Baseline", baseline_scores, duration)

        # --- b. GridSearchCV ---
        print("    Avaliando: GridSearchCV")
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=CV,
            scoring='accuracy',
            n_jobs=-1
        )
        start_time = time.time()
        grid_search.fit(X, y)
        duration = time.time() - start_time
        
        tuned_scores_grid = cross_validate(grid_search.best_estimator_, X, y, cv=CV, scoring=scoring_metrics)
        store_results(all_results, data_name, model_name, "GridSearchCV", tuned_scores_grid, duration)

        # --- c. RandomizedSearchCV ---
        print("    Avaliando: RandomizedSearchCV")
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=N_ITER_RANDOM_BAYES,
            cv=CV,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        start_time = time.time()
        random_search.fit(X, y)
        duration = time.time() - start_time
        
        tuned_scores_random = cross_validate(random_search.best_estimator_, X, y, cv=CV, scoring=scoring_metrics)
        store_results(all_results, data_name, model_name, "RandomizedSearchCV", tuned_scores_random, duration)

        # --- d. BayesSearchCV ---
        if SKOPT_AVAILABLE:
            print("    Avaliando: BayesSearchCV")
            bayes_search = BayesSearchCV(
                estimator=pipeline,
                search_spaces=bayes_space,
                n_iter=N_ITER_RANDOM_BAYES,
                cv=CV,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42
            )
            start_time = time.time()
            bayes_search.fit(X, y)
            duration = time.time() - start_time

            tuned_scores_bayes = cross_validate(bayes_search.best_estimator_, X, y, cv=CV, scoring=scoring_metrics)
            store_results(all_results, data_name, model_name, "BayesSearchCV", tuned_scores_bayes, duration)

print("\n--- Processamento Concluído ---")

# --- 4. Geração dos Resultados ---
df_results = pd.DataFrame(all_results)

tuner_order = ["Baseline", "GridSearchCV", "RandomizedSearchCV"]
if SKOPT_AVAILABLE:
    tuner_order.append("BayesSearchCV")

def create_pivot(df, metric):
    pivot = df.pivot_table(
        index=["Dataset", "Model"],
        columns="Tuner",
        values=metric
    )
    return pivot.reindex(columns=tuner_order)

df_accuracy = create_pivot(df_results, "Accuracy")
df_f1 = create_pivot(df_results, "F1")
df_precision = create_pivot(df_results, "Precision")
df_recall = create_pivot(df_results, "Recall")
df_time = create_pivot(df_results, "Time (s)")


print("\n--- TABELAS COMPARATIVAS ---")

print("\nResultados: Acurácia Média (Accuracy)")
print(df_accuracy.to_markdown(floatfmt=".4f"))

print("\nResultados: F1-Score Médio (Weighted)")
print(df_f1.to_markdown(floatfmt=".4f"))

print("\nResultados: Tempo de Tuning (Segundos)")
print(df_time.to_markdown(floatfmt=".2f"))

print("\nResultados: Precision Média (Weighted)")
print(df_precision.to_markdown(floatfmt=".4f"))

print("\nResultados: Recall Média (Weighted)")
print(df_recall.to_markdown(floatfmt=".4f"))