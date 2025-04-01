# 🎓 Examen MLOps - Sup de Vinci 2025 


## Objectif
Prédire les réadmissions hospitalières avec un modèle XGBoost, suivi et déployé avec MLflow et FastAPI.

## Structure
- `train.py` : entraînement du modèle + MLflow
- `app/` : API FastAPI
- `src/` : wrapper pour charger le modèle
- `data/` : jeu de données TSV
- `MLproject` + `requirements.txt` : exécution avec MLflow

### Choix des hyperparamètres

Une recherche en grille a été réalisée sur deux hyperparamètres clés de XGBoost :

- **`max_depth`** : définit la profondeur maximale des arbres. Deux valeurs ont été testées (3 et 5) afin de comparer un modèle simple (moins de risque d’overfitting) à un modèle plus complexe pouvant capturer des interactions plus profondes.

- **`n_estimators`** : contrôle le nombre total d’arbres dans le modèle. Les valeurs 100 et 200 ont été testées pour comparer la performance d’un modèle rapide contre un modèle plus lent mais potentiellement plus performant.

Ces choix ont été guidés par un compromis entre performance, généralisation et temps de calcul dans un environnement cloud partagé.
Un **grid search manuel** a été effectué sur deux hyperparamètres clés de XGBoost :

| `max_depth` | `n_estimators` |
|-------------|----------------|
| 3           | 100            |
| 3           | 200            |
| 5           | 100            |
| 5           | 200            |

Un total de 4 modèles ont été testés avec une recherche en grille sur :
- `max_depth` ∈ {3, 5}
- `n_estimators` ∈ {100, 200}
 
## Résultat
✅ Choix final : max_depth = 5, n_estimators = 200

C’est le modèle le plus performant en accuracy parmi tous les runs : 96.38 %
La profondeur 5 permet au modèle de capturer des relations non linéaires complexes dans les données, ce qui améliore la précision.
Le fait d’avoir 200 arbres permet une meilleure stabilité du modèle, et compense d’éventuels sous-apprentissages.
## Analyse visuelle :
Dans MLflow, la visualisation en Parallel Coordinates Plot montre que la combinaison max_depth=5 et n_estimators=200 est la seule à atteindre la zone rouge (haut de l’échelle de précision), ce qui confirme numériquement et visuellement sa supériorité.


Bien que ce modèle soit le plus long à entraîner (environ 8 minutes), le gain significatif en performance justifie largement ce choix, surtout dans un cadre de production où la performance est prioritaire par rapport au temps d’entraînement ponctuel.

## API
Lancer avec :
```bash
uvicorn app.main:app --reload


---

## 🧪 Entraînement & Suivi avec MLflow


### ✔️ Objectif respecté :
- 📉 `mlflow.log_param()` pour les hyperparamètres
- 📈 `mlflow.log_metric()` pour `accuracy`
- 📦 `mlflow.xgboost.log_model()` avec `signature` et `input_example`
- 🔍 Comparaison des runs via MLflow UI

### 📸 Résultat du meilleur run :

| max_depth | n_estimators | accuracy |
|-----------|--------------|----------|
| 5         | 200          | **0.9638 ✅** |

> Le modèle retenu est enregistré dans le **Model Registry MLflow** sous le nom `xgb-readmission`, version **5**.

👉 [Accéder à MLflow UI](https://user-hadji-mlflow.user.lab.sspcloud.fr)

---



---

## 🌐 Déploiement en API REST (FastAPI)

L’API expose une route `/predict` pour interroger le modèle déployé.

 Tester l’API localement (SSPCloud)
 Sur SSPCloud, le paramètre --root-path /proxy/8000 est obligatoire pour que la documentation /docs fonctionne correctement.
### Exemple d'entrée JSON :

```json
{
  "chol": 3.5,
  "crp": 0.8,
  "phos": 1.4
}
--> PREDICTION 1 

{
  "chol": 9.72,
  "crp": 14.87,
  "phos": 8.48
} 
--> PREDICTION 0

