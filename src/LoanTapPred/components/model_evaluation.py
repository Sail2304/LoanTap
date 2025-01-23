import os 
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from src.LoanTapPred.entity.config_entity import ModelEvaluationConfig
from src.LoanTapPred.utils.common import save_json
from pathlib import Path

import os
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/Sail2304/Ola-driver-churn.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="Sail2304"
os.environ["MLFLOW_TRACKING_PASSWORD"]="b9b3f722c5b2cfcfba8e769fcd6c4ffd37e6136b"

class ModelEvaluation():
    def __init__(self, config:ModelEvaluationConfig):
        self.config = config
    
    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual,pred)
        recall = recall_score(actual,pred)
        f1 = f1_score(actual, pred)
        cm = confusion_matrix(actual,pred)

        return accuracy, precision, recall, f1, cm
    
    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        sc = joblib.load(self.config.scaler_path)

        X_test = test_data.drop(columns=[self.config.target_column])
        y_test = test_data[[self.config.target_column]]
        X_test_scaled=sc.transform(X_test)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            pred = model.predict(X_test_scaled)
            (accuracy, precision, recall, f1, cm) = self.eval_metrics(y_test, pred)
            TN, FP, FN, TP = cm.ravel()
            # saving metrics as local
            scores = {"accuracy": accuracy, 
                      "precision":precision, 
                      "recall": recall, 
                      "f1": f1,
                      "TN": int(TN),
                      "FP": int(FP),
                      "FN": int(FN),
                      "TP": int(TP)
            }

            
            save_json(path=Path(self.config.metric_file_name), data=scores)


            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            mlflow.log_metric("TN", TN)
            mlflow.log_metric("FP", FP)
            mlflow.log_metric("FN", FN)
            mlflow.log_metric("TP", TP)
            mlflow.log_artifact(self.config.scaler_path)
            mlflow.log_artifact(self.config.ohencoder_path)
            mlflow.log_artifact(self.config.le_grade_path)
            mlflow.log_artifact(self.config.le_subgrade_path)
            mlflow.log_artifact(self.config.le_emp_length_path)

            #model registery does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="LoanTapLR")
            else:
                mlflow.sklearn.log_model(model, "model")
    



            





            





            


