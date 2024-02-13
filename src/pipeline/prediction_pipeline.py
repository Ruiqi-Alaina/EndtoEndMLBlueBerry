import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import super_learner_prediction
from src.utils import load_object

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        try:
            model_path = os.join.path('artifact','model.pkl')
            preprocessor_path = os.join.path('artifact', 'preprocessor.pkl')
            model =  load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_transformed = preprocessor.transform(features)
            base_model = model['base_models']
            meta_model = model["meta_models"]
            preds = super_learner_prediction(data_transformed,base_model,meta_model)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 clonesize:float, 
                 honeybee:float, 
                 bumbles:float, 
                 andrena:float, 
                 osmia:float,
                 MaxOfUpperTRange:float,
                 RainingDays:float, fruitset:float,fruitmass:float, seeds:float):
        self.clonesize = clonesize
        self.honeybee = honeybee
        self.bumbles = bumbles
        self.andrena = andrena
        self.osmia = osmia
        self.MaxOfUpperTRange = MaxOfUpperTRange
        self.RainingDays = RainingDays
        self.fruitset = fruitset
        self.fruitmass = fruitmass
        self.seeds = seeds
    
    def _get_data_as_data_frame(self):
        try:
            Custom_data_input_dict = {
                'clonesize': [self.clonesize], 
                'honeybee': [self.honeybee], 
                'bumbles': [self.bumbles], 
                'andrena': [self.andrena], 
                'osmia': [self.osmia],
                'MaxOfUpperTRange': [self.MaxOfUpperTRange],
                'RainingDays': [self.RainingDays], 
                'fruitset': [self.fruitset], 
                'fruitmass': [self.fruitmass], 
                'seeds': [self.seeds]
            }
            return pd.DataFrame(Custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)