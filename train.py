import os
import yaml
import joblib
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils.preprocessing import load_data, split_data
from utils.evaluation import evaluate_model, visualize_distribution, visualize_importance

os.makedirs("models", exist_ok=True)
os.makedirs("visuals", exist_ok=True)

def objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
    }
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    return clf.score(X_train, y_train)

def main():
    with open("config/train_config.yaml") as f:
        cfg = yaml.safe_load(f)

    df = load_data(cfg["data_path"])
    visualize_distribution(df, cfg["target_column"], "visuals/class_distribution.png")
    X_train, X_test, y_train, y_test = split_data(df, cfg)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=cfg["n_trials"])

    best_params = study.best_params
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.joblib")
    evaluate_model(model, X_test, y_test, "visuals/confusion_matrix.png")
    visualize_importance(model, X_train.columns, "visuals/feature_importance.png")

    print("Model trained and saved. Best params:", best_params)

if __name__ == "__main__":
    main()
