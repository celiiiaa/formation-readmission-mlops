import mlflow

class ReadmissionPredictor:
    def __init__(self, model_name="xgb-readmission", version="5"):
        self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")

    def predict(self, data):
        return self.model.predict(data)