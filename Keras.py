# KERAS
#%% Importar pacotes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping

#%% Carregar os dados
titanic = pd.read_pickle('titanic1.pkl')
X = titanic.drop(columns='survived')
y = titanic.survived

#%% Dividir os dados em treino e teste (holdout)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

#%% Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% Construir a rede neural com 5 camadas
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Camada oculta 1
    Dense(32, activation='relu'),  # Camada oculta 2
    Dense(16, activation='relu'),  # Camada oculta 3
    Dense(8, activation='relu'),   # Camada oculta 4
    Dense(1, activation='sigmoid') # Camada de saída (classificação binária)
])

#%% Compilar o modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Função de perda para classificação binária
    metrics=[AUC(name='auc')]    # Usar AUC como métrica
)

#%% Resumo do modelo
model.summary()

#%% Treinar o modelo
early_stopping = EarlyStopping(
    monitor='val_auc',  # Monitorar a AUC no conjunto de validação
    patience=10,        # Parar após 10 épocas sem melhoria
    mode='max',         # Maximizar a AUC
    restore_best_weights=True  # Restaurar os melhores pesos encontrados
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,          # Número máximo de épocas
    batch_size=32,       # Tamanho do batch
    callbacks=[early_stopping],  # Usar early stopping
    verbose=1
)

#%% Avaliar o modelo no conjunto de teste
results = model.evaluate(X_test, y_test, verbose=0)
print(f"AUC no teste: {results[1]:.4f}")
