import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline ML de prédiction de churn client"
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["prepare", "train", "evaluate", "all"],
        help="Étape à exécuter : prepare, train, evaluate ou all",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="Churn_Modelling.csv",
        help="Chemin vers le fichier CSV",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="classifier.joblib",
        help="Chemin du fichier modèle sauvegardé (.joblib)",
    )
    args = parser.parse_args()

    if args.step == "prepare":
        x_train, x_test, y_train, y_test = prepare_data(args.data)
        print(
            f"Données préparées : x_train={x_train.shape}, x_test={x_test.shape}"
        )

    elif args.step == "train":
        x_train, _, y_train, _ = prepare_data(args.data)
        model = train_model(x_train, y_train)
        save_model(model, args.model_path)

    elif args.step == "evaluate":
        _, x_test, _, y_test = prepare_data(args.data)
        model = load_model(args.model_path)
        evaluate_model(model, x_test, y_test)

    elif args.step == "all":
        x_train, x_test, y_train, y_test = prepare_data(args.data)
        model = train_model(x_train, y_train)
        save_model(model, args.model_path)
        loaded_model = load_model(args.model_path)
        evaluate_model(loaded_model, x_test, y_test)


if __name__ == "__main__":
    main()