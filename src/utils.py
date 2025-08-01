import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            import pickle
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluate the performance of different regression models.
    Returns a dictionary with model names as keys and their R2 scores as values.
    """
    model_report = {}
    
    try:
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_value = r2_score(y_test, y_pred)
            model_report[model_name] = r2_value
            logging.info(f"{model_name} R2 Score: {r2_value}")
        return model_report
    except Exception as e:
        raise CustomException(e, sys)
