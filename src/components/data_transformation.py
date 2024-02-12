# feature engineering and data cleaning
import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass 
class DataTransformationConfig:
    preprocessing_obj_file_path = os.path.join("artifact","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        '''
        This function is used to define how to transform data
        '''
        try :
            num_attribs = ['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia','MaxOfUpperTRange', 'RainingDays', 'fruitset', 'fruitmass', 'seeds']
            num_pipeline = Pipeline(
                steps=[("standardscale", StandardScaler())])
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_attribs)
            ],remainder='passthrough')
            return preprocessor
        except Exception as e :
            raise CustomException(e, sys)
        
    def initiate_data_transformer(self, train_path, test_path):
        '''
        This function is used to start the transformation process
        '''
        try: 
            logging.info("Import train and test data set")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            y_train = train_data['yield']
            y_test = test_data["yield"]
            x_train = train_data[['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia','MaxOfUpperTRange', 'RainingDays', 'fruitset', 'fruitmass', 'seeds']]
            x_test = test_data[['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia','MaxOfUpperTRange', 'RainingDays', 'fruitset', 'fruitmass', 'seeds']]
            logging.info("start to transform the data")
            processor_object = self.get_data_transformer_object()
            x_train_transformed = processor_object.fit_transform(x_train)
            x_test_transformed = processor_object.fit_transform(x_test)
            logging.info("Data has been transformed")
            train_arr = np.c_[x_train_transformed, np.array(y_train)]
            print(train_arr.shape)
            test_arr = np.c_[x_test_transformed, np.array(y_test)]
            save_object(file_path=self.data_transformation_config.preprocessing_obj_file_path, obj= processor_object)
            logging.info("Data preprocessor has been saved")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessing_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)