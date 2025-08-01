import os 
import sys
from src.exception import CustomException
from src.logger import logging
from catboost import CatBoostRegressor
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import save_object, evaluate_model
from src.components.data_injestion import DataIngestion
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0)
            }

            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)

            # Get the key of the best-performing model
            best_model_name = max(model_report, key=model_report.get)

            # Get its corresponding score
            best_model_score = model_report[best_model_name]

            # Create a dictionary with the best model and its score
            best_model = {best_model_name: best_model_score}

            # Check if the score is above threshold
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient score.")
            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score}")
            # Save the best model
            save_object(self.model_trainer_config.trained_model_file_path, models[best_model_name])
            logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")
            print(f"best model R2 score: {best_model_score}")

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    data_injestion = DataIngestion()
    data_transformation = DataTransformation()
    model_trainer = ModelTrainer()

    train_data_path, test_data_path = data_injestion.initiate_data_ingestion()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_trainer.initiate_model_trainer(train_arr, test_arr)
    

