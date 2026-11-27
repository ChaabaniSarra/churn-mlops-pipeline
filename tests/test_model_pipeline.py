"""Exemple de tests unitaires."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_pipeline import prepare_data, train_model  # noqa: E402

DATA_PATH = "Churn_Modelling.csv"


def test_prepare_data():
    x_train, x_test, y_train, y_test = prepare_data(DATA_PATH)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) > 0


def test_train_model():
    x_train, x_test, y_train, _ = prepare_data(DATA_PATH)
    model = train_model(x_train, y_train)
    assert len(model.predict(x_test)) == len(x_test)
