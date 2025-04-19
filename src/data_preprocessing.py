import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Optional, List, Tuple, Literal
import numpy as np
from src.utils.utils import logger

class DataPreprocessor:
    def __init__(
        self, 
        data: pd.DataFrame,
        target_feature: str, 
        scaler_type: Literal["standard", "minmax", "robust"] = "standard",
        features_to_drop: Optional[List[str]] = None,
        cleaned_data: pd.DataFrame = None,
    ):
        self.data = data
        self.imputer = SimpleImputer(strategy="median")
        # Select the appropriate scaler
        scalers = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}
        self.scaler = scalers.get(scaler_type, StandardScaler())

        # Encoding type
        self.encoder = OneHotEncoder(drop='first',handle_unknown="ignore", sparse_output=False)

        # Feature Selection by Inclusion/Exclusion
        self.target = target_feature
        self.features_to_drop = features_to_drop

        #skewed feature column names
        self.log_transform_cols = {}
        
        self.cleaned_data = None
    
    # process data, split data, and process again if required
    def process_data_split(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        
        X, y = self.presplit_process(self.data)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_clean, X_test_clean, y_train_clean, y_test_clean = self.postsplit_process((X_train, X_test, y_train, y_test))

        self.cleaned_data = (X_train_clean, X_test_clean, y_train_clean, y_test_clean)

        return X_train_clean, y_train_clean, X_test_clean, y_test_clean
    
    def presplit_process(self, data) -> Tuple[pd.DataFrame, pd.Series]:
        """
        handles all the data cleaning/processing pre train test split
        Performs operations like duplicate dropping, column cleaning, column standardizing, cleans data, one hot encoding
        """
        #make a copy, dont modify original data
        df = data.copy()

        # INSERT PRE TRAIN TEST SPLIT OPERATONS HERE

        # Select features and labels
        y = df[self.target]
        X = df.drop(columns=self.target)

        #set dummy variables for catagorical data
        cat_cols = X.select_dtypes(include=['object']).columns
        X = pd.get_dummies(X, columns=cat_cols, drop_first = True)

        return X,y
    
    def postsplit_process(self, data):
        """
        performs operations such as imputing, log transformations
        
        """
        X_train, X_test, y_train, y_test = data

        #  INSERT POST TRAIN TEST SPLIT OPERATIONS HERE


        return X_train, X_test, y_train, y_test


