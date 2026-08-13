# -*- coding: utf-8 -*-
#%% Importação dos pacotes

import pickle
import pandas as pd
from funcoes_ajuda import descritiva, relatorio_missing, \
    diagnóstico, avalia_clf
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, \
    StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt


#%% Importar a base (já tratada)
titanic = pd.read_pickle('titanic1.pkl')

#%% Checar rapidamente a base, tipos de dados e missing
titanic.info()
relatorio_missing(titanic)
#%% Definindo a lista de features
variaveis = list(titanic.columns)
vResp = 'survived'

print(variaveis)
print(vResp)

#%% A análise descritiva é sempre um passo muito importante
for var in variaveis:
    descritiva(titanic, var, vResp, 6)
    
#%% Dividir a base em treino e teste
X = titanic[variaveis]
y=titanic[vResp]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1729)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#%% Construindo a primeira árvore
arvore1 = DecisionTreeClassifier()
arvore1.fit(X_train, y_train)

#%% Avaliando a primeira árvore
avalia_clf(arvore1, y_train, X_train, rótulos_y=['Não sobreviveu', 'Sobreviveu'],base='treino')
avalia_clf(arvore1, y_test, X_test, rótulos_y=['Não sobreviveu', 'Sobreviveu'],base='teste')

#%%
pred = arvore1.predict(X_test)
pd.crosstab(pred, y_test)

#%% Pronto! Acurácia perfeita na base de testes... pera...
print(X_train.columns)
# O que há de errado com a árvore?

#%% O erro numero 1 dos modelos perfeitos: a target no meio das features
# Corrigindo
variaveis.remove('survived')

#%% Refazendo as bases de treino e teste
X = titanic[variaveis]
y=titanic[vResp]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1729)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#%% Treinando a primeira árvore (correta agora)
arvore1 = DecisionTreeClassifier()
arvore1.fit(X_train, y_train)

#%% Avaliando a primeira árvore (correta)
avalia_clf(arvore1, y_train, X_train, rótulos_y=['Não sobreviveu', 'Sobreviveu'],base='treino')
avalia_clf(arvore1, y_test, X_test, rótulos_y=['Não sobreviveu', 'Sobreviveu'],base='teste')

#%%
path = arvore1.cost_complexity_pruning_path(X_train, y_train)  # CCP Path na base de treino
ccp_alphas, impurities = path.ccp_alphas, path.impurities

#%% Verificar se há duplicações nos ccp_alphas
print(len(ccp_alphas))
len(pd.Series(ccp_alphas).unique())
#%%
ccp_alphas = pd.Series(ccp_alphas).unique()
#%%  Avaliar diferentes alfas
ginis=[]

for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    gini = auc*2-1
    ginis.append(gini)
    # Avaliar no conjunto de validação ou com validação cruzada

relatorio = pd.DataFrame({'ccp_alpha':ccp_alphas, 'gini':ginis})    
#%%
maximo = relatorio[relatorio.gini == relatorio.gini.max()]
ccp_max = maximo.ccp_alpha.iloc[0]
maximo
#%%
fig, ax = plt.subplots(1)
sns.pointplot(y=ginis, x=ccp_alphas, ax=ax)

# Configuração dos ticks
passo = 8
x_ticks = ax.get_xticks()[::passo]
x_tick_labels = [f'{x:.3%}' for x in ccp_alphas[::passo]]
ax.set_xticks(x_ticks, x_tick_labels)

# Título do gráfico
ax.set_title('Gini de treino por valor de CCP_alpha')

# Adicionar linha vertical no ccp_alpha que resulta no Gini máximo
ax.axvline(x=maximo.index[0], color='red', linestyle='--', label=f'ccp_alpha (Gini máximo): {ccp_max:.3%}')

# Mostrar legenda
ax.legend()

# Exibir o gráfico
plt.show()

#%%
arvore2 = DecisionTreeClassifier(ccp_alpha=ccp_max)
arvore2.fit(X_train, y_train)
#%%
avalia_clf(arvore2, y_train, X_train, rótulos_y=['Não sobreviveu', 'Sobreviveu'],base='treino')
avalia_clf(arvore2, y_test, X_test, rótulos_y=['Não sobreviveu', 'Sobreviveu'],base='teste')
#%%

# Configurar o grid (as opções que vamos testar)
param_grid = {'ccp_alpha': ccp_alphas}
param_grid = {
    'ccp_alpha': ccp_alphas,                # Valores de poda de complexidade de custo
    'max_depth': [None, 5, 10],       # Profundidade máxima da árvore
    'min_samples_split': [2, 10],        # Número mínimo de amostras para dividir um nó
    # 'min_samples_leaf': [1, 2, 4],          # Número mínimo de amostras em uma folha
    # 'max_features': [None, 'sqrt', 'log2'], # Número máximo de features consideradas
    'criterion': ['gini', 'entropy']        # Critério de divisão
}

# Configurar a validação cruzada (CV)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) 
#Instanciar o classificador (árvore)
clf = DecisionTreeClassifier(random_state=42)

# Instanciar o GridSearchCV com o grid, CV e classificador
grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=kf,
    scoring='roc_auc',
    return_train_score=True
)

# executar o grid
grid_search.fit(X_train, y_train)

# Resultados
best_alpha = grid_search.best_params_['ccp_alpha']
best_score = grid_search.best_score_
print(f"Melhor alpha: {best_alpha}")
print(f"Acurácia média na validação cruzada: {best_score:.4f}")

#%%
resultados = pd.DataFrame(grid_search.cv_results_)
resultados.head()
#%%
resultados.columns

############ Análise do Grid ###############
#%% Avaliação do gini na base de cross validation por ccp_alpha
resultados['gini'] = resultados.mean_test_score*2-1
sns.lineplot(data=resultados, x='param_ccp_alpha', y='gini')
plt.show()

#%%  Boxplot do gini na base de CV por ccp_alpha
# Aqui estou interessado em avaliar máximos e mínimos
fig, ax = plt.subplots(1)

resultados['gini'] = resultados.mean_test_score*2-1
sns.boxplot(data=resultados, 
             x=resultados['param_ccp_alpha'].astype('str'), 
             y='gini',
             ax=ax)
passo = 8
x_ticks = ax.get_xticks()[::passo]
x_tick_labels = [f'{x:.3%}' for x in ccp_alphas[::passo]]
ax.set_xticks(x_ticks, x_tick_labels)
plt.show()

#%% Gini por profundidade máxima
resultados['gini'] = resultados.mean_test_score*2-1
sns.boxplot(data=resultados, 
             x=resultados['param_max_depth'].astype('str'), 
             y='gini')
plt.show()

#%% Gini por param_min_samples_split
resultados['gini'] = resultados.mean_test_score*2-1
sns.boxplot(data=resultados, 
             x=resultados['param_min_samples_split'].astype('str'), 
             y='gini')
plt.show()

#%% Treinar o modelo final com o melhor alpha
final_clf = grid_search.best_estimator_
prob = final_clf.predict_proba(X_test)[:,1]

# Avaliar na base de teste
test_score = final_clf.score(X_test, y_test)
auc = roc_auc_score(y_test, prob)
gini = auc*2-1
print(f"Acurácia na base de teste: {test_score:.4f}")
print(f"Gini na base de teste: {gini:.4f}")

#%%
df_test = X_test.copy()
df_test['y'] = y_test
df_test['p'] = prob
df_test.head(3)
#%% Avaliando a resposta do modelo para cada variável
for var in X_test.columns:
    diagnóstico(df_test, var, vresp='y', pred='p', max_classes=8)

#%% Salvar o modelo final em um arquivo usando Pickle
with open('arvore_final.pkl', 'wb') as file:
    pickle.dump(final_clf, file)
    