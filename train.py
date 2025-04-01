import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

# Chargement des données
df = pd.read_csv("data/DSA-2025_clean_data.tsv", sep="\t")

# Séparation des features et de la cible
X = df.drop(columns=["readmission"])
y = df["readmission"]

# Découpage train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid search sur 2 hyperparamètres
param_grid = {
    "max_depth": [3, 5],
    "n_estimators": [100, 200]
}

# Boucle sur les combinaisons d'hyperparamètres
for max_depth in param_grid["max_depth"]:
    for n_estimators in param_grid["n_estimators"]:
        with mlflow.start_run():
            model = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            # Log des hyperparamètres et métriques
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_metric("accuracy", acc)

            # Exemple d'entrée + signature pour le modèle
            input_example = X_test.iloc[:1]
            signature = infer_signature(X_train, model.predict(X_train))

            # Enregistrement du modèle dans le registry MLflow
            mlflow.xgboost.log_model(
                model,
                "model",
                registered_model_name="xgb-readmission",
                input_example=input_example,
                signature=signature
            )
