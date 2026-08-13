# -*- coding: utf-8 -*-

#%% Importação dos pacotes

import pandas as pd
# import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, \
    accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import lightgbm as lgb

#%% Carregar os dados
X_train = pd.read_pickle('X_train.pkl')
y_train = pd.read_pickle('y_train.pkl')['label']
X_test  = pd.read_pickle('X_test.pkl')
y_test  = pd.read_pickle('y_test.pkl')['label']

#%% Verificar as categorias das labels
níveis = y_test.cat.categories
print(níveis)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#%% Definir o espaço de busca de hiperparâmetros para o LightGBM
param_space = {
    'n_estimators': Integer(50, 500),  # Número de árvores
    'max_depth': Integer(3, 15),       # Profundidade máxima das árvores
    'learning_rate': Real(0.01, 0.3, 'log-uniform'),  # Taxa de aprendizado
    'num_leaves': Integer(20, 100),    # Número máximo de folhas
    'min_child_samples': Integer(10, 100),  # Número mínimo de amostras por folha
    'subsample': Real(0.5, 1.0),       # Subamostragem de dados
    'colsample_bytree': Real(0.5, 1.0),  # Subamostragem de features
    'reg_alpha': Real(0, 1),           # Regularização L1
    'reg_lambda': Real(0, 1),          # Regularização L2
    'boosting_type': Categorical(['gbdt', 'dart'])  # Tipo de boosting
}

#%% Configurar o modelo LightGBM
lgb_model = lgb.LGBMClassifier(random_state=2244000, verbose=-1)

#%% Configurar o Bayesian Search
bayes_search = BayesSearchCV(
    estimator=lgb_model,
    search_spaces=param_space,
    n_iter=5,  # Número de iterações
    cv=2,       # Número de folds na validação cruzada
    scoring='accuracy',
    n_jobs=-1,  # Usar todos os núcleos do processador
    verbose=1,
    random_state=2244000
)

#%% Executar o Bayesian Search
tempo_ini = pd.Timestamp.now()  # Início do cronômetro
bayes_search.fit(X_train, y_train)
tempo_fim = pd.Timestamp.now()  # Fim do cronômetro
print(f"Tempo de execução: {tempo_fim - tempo_ini}")

#%% Melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros:", bayes_search.best_params_)

#%% Gerar as previsões do modelo
pred = bayes_search.predict(X_test)

#%% Gerar a matriz de confusão e estatísticas
cm = confusion_matrix(y_test, pred)
print("Matriz de Confusão:")
print(cm)
print("\nRelatório de Classificação:")
print(classification_report(y_test, pred))

#%% Gerar um DataFrame temporário para avaliar o modelo
lgb_aval = pd.DataFrame({
    'pred': pred,
    'obs': y_test
})

#%% Função personalizada para summary (multiClassSummary equivalente)
def multiClassSummary(df, levels):
    report = classification_report(df['obs'], df['pred'], output_dict=True)
    summary = {
        'Accuracy': report['accuracy']
    }
    for level in levels:
        summary[f'{level} Precision'] = report[level]['precision']
        summary[f'{level} Recall'] = report[level]['recall']
        summary[f'{level} F1-score'] = report[level]['f1-score']
    return summary

#%% Calcular métricas de avaliação
metrics = pd.Series(multiClassSummary(lgb_aval, níveis))
print("\nMétricas de Avaliação:")
print(metrics)

 #%%
acc_teste = accuracy_score(y_test, lgb_aval.pred)
print(f'A acurácia na base de teste foi de: {acc_teste:.2%}')

