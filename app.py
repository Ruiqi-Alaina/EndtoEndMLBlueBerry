from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('page.html')
    else:
        data=CustomData(
            clonesize=request.form.get("Clone Size"),
            honeybee=request.form.get('HoneyBee Density'),
            bubbles=request.form.get("BubblesBee Density"),
            andrena=request.form.get("AndrenaBee Density"),
            osmia=request.form.get("OsmiaBee Density"),
            MaxOfUpperTRange=request.form.get("Upper T"),
            fruitset=float(request.form.get('Fruitset')),
            fruitmass=float(request.form.get('Fruitmass')),
            seeds=float(request.form.get('Seeds')),
            RainingDays = request.form.get('RainDay')
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('page.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        
