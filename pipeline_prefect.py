"""Pipeline MLOps orchestré avec Prefect (Atelier 3)."""

import argparse
import subprocess
import sys

from prefect import flow, task

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run(cmd):
    """Exécute une commande et arrête si elle échoue."""
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Échec : " + " ".join(cmd))
    return result.stdout


# ===================== TASKS DATA / MODELE =====================


@task(name="1-Installer les dependances")
def install_dependencies_task():
    return run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="2-Preparation des donnees")
def prepare_data_task(path=DATA_PATH):
    return prepare_data(path)


@task(name="3-Entrainement du modele")
def train_model_task(x_train, y_train):
    return train_model(x_train, y_train)


@task(name="4-Sauvegarder le modele")
def save_model_task(model, path=MODEL_PATH):
    save_model(model, path)
    return path


@task(name="5-Charger le modele")
def load_model_task(path=MODEL_PATH):
    return load_model(path)


@task(name="6-Evaluer le modele")
def evaluate_model_task(model, x_test, y_test):
    return evaluate_model(model, x_test, y_test)


# ===================== TASKS CODE =====================


@task(name="Formatage du code")
def format_code_task():
    return run([sys.executable, "-m", "black", *CODE_FILES])


@task(name="Qualite du code")
def quality_code_task():
    return run([sys.executable, "-m", "flake8", "--max-line-length=100", *CODE_FILES])


@task(name="Securite du code")
def security_code_task():
    return run([sys.executable, "-m", "bandit", "-ll", *CODE_FILES])


@task(name="Tests unitaires")
def unit_tests_task():
    return run([sys.executable, "-m", "pytest", "tests/", "-v"])


# ===================== FLOWS =====================


@flow(name="install")
def install_flow():
    install_dependencies_task()


@flow(name="code")
def code_flow():
    format_code_task()
    quality_code_task()
    security_code_task()
    unit_tests_task()


@flow(name="train")
def train_flow():
    x_train, x_test, y_train, y_test = prepare_data_task()
    model = train_model_task(x_train, y_train)
    save_model_task(model)


@flow(name="evaluate")
def evaluate_flow():
    _, x_test, _, y_test = prepare_data_task()
    model = load_model_task()
    return evaluate_model_task(model, x_test, y_test)


@flow(name="all")
def all_flow():
    install_dependencies_task()
    code_flow()
    x_train, x_test, y_train, y_test = prepare_data_task()
    model = train_model_task(x_train, y_train)
    save_model_task(model)
    return evaluate_model_task(model, x_test, y_test)


FLOWS = {
    "all": all_flow,
    "install": install_flow,
    "code": code_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
