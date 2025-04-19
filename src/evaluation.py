import mlflow
from src.utils.utils import logger
import pandas as pd
from typing import Union, Optional
from src.model_training import ModelManager
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from src.utils.utils import logger
import os


class Evaluator:
    def __init__(
        self, 
        model: ModelManager,
        X_test: Union[pd.DataFrame, pd.Series],  # DataFrame or Series, depending on the data structure
        y_test: pd.Series,  # True labels
        uri: str,  # URI where the results might be saved or logged
        original_N_values: Optional[pd.Series],
    ):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.uri = uri
        self.original_N_values = original_N_values

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        if self.model.task == "classification":
            self.log_classification(y_pred)
            logger.info("Logged successfully!")
        else:
            self.log_regression(y_pred)
            logger.info("Logged successfully!")

        
       

    #log results into mlflow
    def log_classification(self, y_pred):

        mlflow.set_tracking_uri(uri=self.uri)
        mlflow.set_experiment(f"{self.model.model_name}")

        report = classification_report(self.y_test, y_pred, output_dict=True)
        # Log the metrics to MLflow
        with mlflow.start_run():
            # Log overall accuracy
            accuracy = report["accuracy"]
            mlflow.log_metric("accuracy", accuracy)
            
            # Log precision, recall, F1-score
            for class_label, metrics in report.items():
                if class_label == "accuracy":
                    continue  
                if class_label == "macro avg" or class_label == "weighted avg":
                    continue  
                
                precision = metrics["precision"]
                recall = metrics["recall"]
                f1_score = metrics["f1-score"]
                
                # Log precision, recall, f1-score
                mlflow.log_metric(f"{class_label}_precision", precision)
                mlflow.log_metric(f"{class_label}_recall", recall)
                mlflow.log_metric(f"{class_label}_f1_score", f1_score)

            # log cm
            cm = confusion_matrix(self.y_test, y_pred)
            # Plot confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=self.model.model.classes_)
            disp.plot(cmap='Blues')
            # Rotate x-axis labels 
            plt.xticks(rotation=90)  # Rotate x-axis labels by 90 degrees

            # Save the confusion matrix plot as an image
            artifact_path = "confusion_matrix.png"
            plt.savefig(artifact_path, bbox_inches="tight")   

            mlflow.log_artifact(artifact_path)
            mlflow.sklearn.log_model(self.model, self.model.model_name)
            if self.model.best_params:
                mlflow.log_params(self.model.best_params)

            os.remove(artifact_path)    


    def log_regression(self, y_pred):

        mlflow.set_tracking_uri(uri=self.uri)
        mlflow.set_experiment(f"{self.model.model_name}")

        #reverse log transform back to get temp values
        y_pred = np.expm1(y_pred)
        self.y_test = np.expm1(self.y_test)
        # Log the metrics to MLflow
        with mlflow.start_run():
            if self.model.best_params:
                for param, value in self.model.best_params.items():
                    mlflow.log_param(param, value)
            #regression metrics
            r2 = r2_score(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse) 
            # Log metrics to MLflow
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)            
            # Assuming y_test contains the true values and y_pred contains the predicted values

            plt.figure(figsize=(8, 6))
            plt.scatter(self.y_test, y_pred, alpha=0.7, color='blue')  # Scatter plot of true vs predicted values
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], color='red', linestyle='--')  # Line representing perfect predictions
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title('True vs Predicted Values')

            # Save the plot as an image file (e.g., 'true_vs_predicted.png')
            plot_path = 'true_vs_predicted.png'
            plt.savefig(plot_path, bbox_inches="tight")

            # Log the image as an artifact in MLflow
            mlflow.log_artifact(plot_path)

            # Close the plot to release resources
            plt.close()
            os.remove(plot_path)  