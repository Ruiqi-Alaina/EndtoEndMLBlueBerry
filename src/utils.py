import os
import sys
import pickle
import numpy as np
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
#from scipy.optimize import minimize
from sklearn.linear_model import HuberRegressor

def hyperparam_tune(x, y, models, params):
    try: 
        best_param_results= dict.fromkeys(list(models.keys()))
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_param = params[list(models.keys())[i]]
            gs = GridSearchCV(model, model_param, cv=10)
            gs.fit(x, y)
            best_param_results[list(models.keys())[i]] = gs.best_params_
            save_object("artifact/params.pkl", best_param_results)
        return best_param_results
    except Exception as e:
        raise CustomException(e, sys)

def fit_base_models(x,y,models,best_parameters):
    "fit each base model and return a model list"
    try:
        fitted_models = list()
        for i in range(len(list(models))):
            print(f"start fitting model {i}")
            model = list(models.values())[i]
            best_param = best_parameters[list(models.keys())[i]]
            try:
                model.set_params(**best_param)
                model.fit(x,y)
            except Exception:
                model.fit(x,y)
            fitted_models.append(model)
        return fitted_models
    except Exception as e:
        raise CustomException(e, sys)

def get_out_of_fold_predictions(train_x, train_y,models,best_parameters):
    '''
    input model dict
    '''
    try:
        kfold = KFold(n_splits=5, shuffle=True)
        meta_x = list()
        meta_y = list()
        for train_ix, valid_ix in kfold.split(train_x):
            print(train_ix)
            print(valid_ix)
            fold_train_x, fold_valid_x = train_x[train_ix], train_x[valid_ix]
            fold_train_y, fold_valid_y = train_y[train_ix], train_y[valid_ix]
            meta_y.extend(fold_valid_y)
            y_hat = []
            model_list = fit_base_models(fold_train_x,fold_train_y,models,best_parameters)
            print(model_list)
            for model in model_list:
                print(2)
                fold_valid_y_hat = model.predict(fold_valid_x)
                print(fold_valid_y_hat)
                y_hat.append(fold_valid_y_hat.reshape(len(fold_valid_y_hat),1)) 
            meta_x.append(np.hstack(y_hat))
        return meta_x, meta_y
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open (file_path, 'rb') as file_obj:
            object = pickle.load(file_obj)
            return object
    except Exception as e:
        raise CustomException(e,sys)

def fit_meta_model(x_meta, y_meta):
    try:
        lr_model =  HuberRegressor()
        lr_model.fit(x_meta, y_meta)
        return lr_model
    except Exception as e:
        raise CustomException(e,sys) 

def super_learner_prediction(x,fitted_models,meta_model):
    try:
        meta_x = []
        for model in fitted_models:
            y_pred = model.predict(x)
            meta_x.append(y_pred.reshape(len(y_pred),1))
        meta_x = np.hstack(meta_x)
        return meta_model.predict(meta_x)
    except Exception as e:
        raise CustomException(e,sys)

def save_object(file_path, obj):
    try: 
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

