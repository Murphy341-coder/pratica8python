from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def carregar_dados():
    dados = load_breast_cancer()
    return dados.data, dados.target


def dividir_dados(X, y, teste=0.3, seed=42):
    return train_test_split(X, y, test_size=teste, random_state=seed)


def treinar_modelo(X_train, y_train):
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    return modelo


def avaliar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}')
    print(f'AUC-ROC: {auc:.2f}')


# Pipeline completo
if __name__ == "__main__":
    X, y = carregar_dados()
    X_train, X_test, y_train, y_test = dividir_dados(X, y)
    modelo = treinar_modelo(X_train, y_train)
    avaliar_modelo(modelo, X_test, y_test)
