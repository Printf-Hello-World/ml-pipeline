from typing import Literal, Dict, Any, Union, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVC, SVR
import os
import joblib
from datetime import datetime
from src.utils.utils import logger

class ModelManager:
    """
    A flexible class for model experimentation that allows for quick model swapping
    and hyperparameter tuning.
    """
    
    def __init__(
        self, 
        task: Literal["classification", "regression"],
        model_name: str,
        hyperparams: Dict[str, Any] = None,
        model_path: str = None,

    ):
        self.task = task
        self.model_name = model_name
        self.hyperparams = hyperparams
        self.best_params = None  
        self.model_path = model_path
        
        if self.model_path:
            self.model = self.load_model(self.model_path)
        else:
            self.model = self.set_model(self.task, self.model_name, **hyperparams)


    def set_model(self, 
                  task: str, 
                  model_name: str, 
                  **hyperparams: Dict[str, Any]) -> Any:
            """
            model name must be included in the available models and the hyperparams must be specific to the model according to sklearn doc
            """

            if task == "classification":
                models = {'logistic': LogisticRegression,'gradient_boosting': GradientBoostingClassifier,'svm': SVC, 'mlp':MLPClassifier}
            elif task == "regression":
                models = {'elasticnet': ElasticNet, 'gradient_boosting': GradientBoostingRegressor,'svm': SVR, 'mlp':MLPRegressor}
            else:
                raise ValueError("invalid task name")
         
            if model_name not in models:
                raise ValueError("invalid model name")
            
            timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")

            self.task = task

            self.hyperparams = hyperparams
            model = models[model_name](**hyperparams)

            self.model_name = f"{model_name}_{timestamp}"

            return model
            
    def fit(self, X_train, y_train):
        
        self.model.fit(X_train, y_train)
        self.save_model()

        self.best_params = self.hyperparams if self.hyperparams is not None else self.model.get_params()

        return self

    
    def grid_search_cv(self, 
                    X_train: Union[pd.DataFrame, np.ndarray], 
                    y_train: Union[pd.DataFrame, np.ndarray], 
                    param_grid, 
                    cv: int):
        
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        

        search = GridSearchCV(self.model, param_grid=param_grid, cv=cv)

        search.fit(X_train, y_train)
        
        self.model = search.best_estimator_
        self.best_params = search.best_params_

        self.save_model()
            
        return self


    def predict(self,X_test):
        return self.model.predict(X_test)

    def save_model(self):
        # Create a directory for saving models (optional)
        os.makedirs('saved_models', exist_ok=True)
        # Save the model
        model_filename = f"saved_models/{self.model_name}_{self.task}.pkl"
        joblib.dump(self.model, model_filename)
        
        logger.info(f'Model saved as {model_filename}')
    
    def load_model(self, model_path):
        current_dir = os.getcwd()
        model_path = f"saved_models/{model_path}"
        # join current directory with the relative db path
        full_path = os.path.join(current_dir, model_path)
        full_path = full_path.replace("\\", "/")
        with open(full_path, 'rb') as model_file:
            model = joblib.load(model_file)
        return model

    
    