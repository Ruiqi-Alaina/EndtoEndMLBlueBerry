import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet 
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
#from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import load_object
#from src.utils import hyperparam_tune
from src.utils import fit_base_models
from src.utils import get_out_of_fold_predictions
from src.utils import fit_meta_model
from src.utils import super_learner_prediction
@dataclass
class ModelTrainerConfig:
    trained_super_learner_file_path = os.path.join("artifact", "model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_config = ModelTrainerConfig()
    def initiate_model_trainer (self, train_array, test_array):
        try: 
            logging.info("hyperparameter tuning job starts")
            models = {
                "ElasticNet": ElasticNet(max_iter=5000),
                "KNN": KNeighborsRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "LGBM": LGBMRegressor(),
                "CatBoost": CatBoostRegressor(),
                            }
            """ params = {
                "ElasticNet": {
                    #"alpha": [1e-2, 1e-1, 1.0, 5.0, 10.0],
                    "l1_ratio": np.arange(0, 1, 0.2)
                },
                "KNN" : {
                    "n_neighbors": [30,60],
                    "metric": ["manhattan","minkowski"]
                },
                "DecisionTree": {
                    "criterion": ["absolute_error"],
                    "max_depth": np.arange(5,100,20),
                    #"splitter": ["best", "random"]
                },
                "RandomForest":{
                    "n_estimators": [64,128,256],
                    "criterion": ["absolute_error"],
                },
                "AdaBoost":{
                    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0],
                    "n_estimators": [64,128,256],
                },
                "XGBoost": {
                    "eta": [0.0001, 0.001, 0.01, 0.1, 1.0],
                    "objective": ["reg:absoluteerror"],
                    "colsample_bynode":np.arange(0.1,1,0.2),
                    "n_estimators": [64,128,256],
                    #"lambda":[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 5.0, 10.0, 100.0],
                    #"alpha":[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 5.0, 10.0, 100.0]
                },
                "LGBM": {
                    "boosting_type":["gbdt","dart","rf"],
                    "n_estimators":[64,128,256],
                    "learning_rate":[0.0001, 0.001, 0.01, 0.1, 1.0],
                    "num_leaves": [16,32,64,128,256],
                    #"reg_lambda": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 5.0, 10.0, 100.0],
                    #"reg_alpha":[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 5.0, 10.0, 100.0]
                },
                "CatBoost": {
                    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0],
                     "iterations":[64,128,256],
                     #"l2_leaf_reg": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 5.0, 10.0, 100.0],
                     "loss_function": ["MAE"]
                }
            } """
            train_x = train_array[:,:-1]
            train_y = train_array[:,-1]
            test_x =test_array[:,:-1]
            test_y = test_array[:,-1]
            # best_params = hyperparam_tune(train_x, train_y,models= models, params=params)
            param_path = os.path.join("artifact","params.pkl")
            best_params = load_object(param_path)
            logging.info("hyperparameter job has finished and return a dictionary of best parameters of each model")
            logging.info("super learner training starts")
            logging.info("Split the data with kfolds")
            meta_x, meta_y = get_out_of_fold_predictions(train_x, train_y,models,best_params)
            logging.info("meta x and meta y have been successfully prepared for building further ensemble model")
            logging.info("base models start to be fit base on the whole training data set")
            fitted_models = fit_base_models(train_x, train_y, models, best_params)
            base_train_mae = []
            base_test_mae = []
            # evaluate the base model
            for model in fitted_models:
                train_y_prediction = model.predict(train_x)
                train_mae = mean_absolute_error(train_y, train_y_prediction)
                test_y_prediction = model.predict(test_x)
                test_mae = mean_absolute_error(test_y, test_y_prediction)
                base_train_mae.append(train_mae)
                base_test_mae.append(test_mae)
            logging.info("base models have been saved successfully")
            logging.info("starts to fit a meta model")
            meta_model  =  fit_meta_model(meta_x, meta_y) 
            logging.info('starts to make predictions within meta model')
            train_y_prediction = super_learner_prediction(train_x, fitted_models,meta_model) 
            test_y_prediction = super_learner_prediction(test_x, fitted_models, meta_model)
            test_mae = mean_absolute_error(test_y, test_y_prediction)
            train_mae = mean_absolute_error(train_x, train_y_prediction)
            final_model = {"base_models": fitted_models, "meta_model": meta_model}
            save_object(self.model_config.trained_super_learner_file_path,final_model)
            logging.info("super model has been saved")
            return train_mae, test_mae, base_train_mae, base_test_mae
        except Exception as e:
            raise CustomException(e,sys)
            
                    
          