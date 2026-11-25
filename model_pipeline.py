import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data(
    csv_path: str = "Churn_Modelling (1).csv",
    test_size: float = 0.2,
    random_state: int = 1,
):
    """
    Charge le jeu de données CSV, nettoie les colonnes inutiles,
    encode la colonne 'Gender' en chiffres et sépare en train / test.
    """
    df = pd.read_csv(csv_path)

    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    columns_to_drop = ["Surname", "Geography", "RowNumber", "CustomerId"]
    df = df.drop(columns=columns_to_drop)

    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return x_train, x_test, y_train, y_test


def train_model(
    x_train, y_train, n_estimators: int = 100, random_state: int = 42
):
    """
    Crée et entraîne un classifieur Random Forest sur les données d'entraînement.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    rf.fit(x_train, y_train)
    return rf


def evaluate_model(model, x_test, y_test):
    """
    Évalue le modèle entraîné sur les données de test et affiche
    l'accuracy ainsi que la matrice de confusion.
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy score: {accuracy * 100:.2f}%")
    print("Confusion matrix:\n", matrix)

    return accuracy, matrix


def save_model(model, filename: str = "classifier.joblib"):
    """
    Sauvegarde le modèle entraîné sur le disque avec joblib.
    """
    joblib.dump(model, filename)
    print(f"Modèle sauvegardé dans : {filename}")


def load_model(filename: str = "classifier.joblib"):
    """
    Charge un modèle précédemment sauvegardé depuis un fichier .joblib.
    """
    model = joblib.load(filename)
    print(f"Modèle chargé depuis : {filename}")
    return model
