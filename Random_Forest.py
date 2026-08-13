# -*- coding: utf-8 -*-

#%% Importação dos pacotes

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, \
    StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

#%% Carregar dados
titanic = pd.read_pickle('titanic1.pkl')
X = titanic.drop(columns='survived')
y=titanic.survived
#%% Dividir os dados em treino e teste (holdout)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

#%% Definir o espaço de hiperparâmetros para Random Forest
param_dist = {
    'n_estimators': [50, 100, 200, 300],  # Número de árvores
    'max_depth': range(2, 30, 1),      # Profundidade máxima das árvores
    'min_samples_split': [2, 5, 10],      # Número mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4],        # Número mínimo de amostras em uma folha
    'max_features': ['sqrt', 'log2', None],  # Número de features consideradas para divisão
    'bootstrap': [True, False],           # Usar bootstrap ou não
    'criterion': ['gini', 'entropy'],     # Critério de divisão
    'ccp_alpha': np.linspace(0, 0.05, 20)  # Parâmetro de poda de complexidade de custo
}
#%% Número de possibilidades:
4*30*3*3*3*2*2*20

#%% Configurar o RandomizedSearchCV
n_iter = 50  # Número de combinações de hiperparâmetros a serem testadas
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold estratificado
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=n_iter,
    cv=kf,
    scoring='roc_auc',
    n_jobs=-1,  # Usar todos os núcleos do processador
    verbose=1,  # Mostrar progresso
    random_state=42
)

#%% Executar o RandomizedSearchCV no conjunto de treino
# Iniciar o cronômetro
start_time = time.time()
random_search.fit(X_train, y_train)
# Parar o cronômetro
end_time = time.time()
elapsed_time = end_time - start_time

#%% Resultados da busca
print(f"\nMelhores hiperparâmetros: {random_search.best_params_}")
print(f"\nAUC média na validação cruzada: {random_search.best_score_:.4f}")
print(f"\nTempo total de execução: {elapsed_time:.2f} segundos")

#%% Treinar o modelo final com os melhores hiperparâmetros
final_clf = random_search.best_estimator_

# Avaliar no conjunto de teste
random_test_score = final_clf.score(X_test, y_test)
random_roc = roc_auc_score(y_test, final_clf.predict_proba(X_test)[:,1])
random_gini = random_roc*2-1
print(f"Gini do random search no teste: {random_gini:.4f}")
#%% Resgatando a árvore do script anterior

# Carregar o modelo salvo
with open('arvore_final.pkl', 'rb') as file:
    arvore_final = pickle.load(file)

#%% Verificar se o modelo suporta predict_proba (caso contrário, usar a decisão bruta)
if hasattr(arvore_final, "predict_proba"):
    y_scores = arvore_final.predict_proba(X_test)[:, 1]  # Probabilidades da classe positiva
else:
    y_scores = arvore_final.decision_function(X_test)  # Alternativa se não houver predict_proba

# Calcular a AUC
auc_score = roc_auc_score(y_test, y_scores)
gini = auc_score*2-1
print(f"gini na base de teste da árvore: {gini:.4f}")



#%% Avaliando os resultados do tunning
resultados = pd.DataFrame(random_search.cv_results_)
resultados['gini'] = resultados.mean_test_score*2-1
resultados.gini.plot.hist(bins=40)
plt.show()

#####################################
#%% Bayesian search
param_space = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(1, 50),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 4),
    'max_features': Categorical(['sqrt', 'log2', None]),
    'bootstrap': Categorical([True, False]),
    'criterion': Categorical(['gini', 'entropy']),
    'ccp_alpha': Real(0, 0.05)
}

n_iter=20
bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=param_space,
    n_iter=n_iter,  # Número de iterações
    cv=5,       # 5-Fold Cross-Validation
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    random_state=42
)


#%% Executar o RandomizedSearchCV no conjunto de treino
# Iniciar o cronômetro
start_time = time.time()
bayes_search.fit(X_train, y_train)
# Parar o cronômetro
end_time = time.time()
elapsed_time = end_time - start_time

#%% Resultados da busca
print(f"Melhores hiperparâmetros: {bayes_search.best_params_}")
print(f"\nAUC média na validação cruzada: {bayes_search.best_score_:.2%}")
print(f"\nTempo total de execução: {elapsed_time:.2f} segundos")

#%% Treinar o modelo final com os melhores hiperparâmetros
final_clf = bayes_search.best_estimator_

# Avaliar no conjunto de teste
bayes_test_score = final_clf.score(X_test, y_test)
bayes_roc = roc_auc_score(y_test, final_clf.predict_proba(X_test)[:,1])
bayes_gini = bayes_roc*2-1
print(f"Gini do bayesian search no teste: {bayes_gini:.4f}")
#%%
resultados_bayes = pd.DataFrame(bayes_search.cv_results_)
resultados_bayes['gini'] = resultados_bayes.mean_test_score*2-1
resultados_bayes.gini.plot.hist(bins=40)
plt.show()