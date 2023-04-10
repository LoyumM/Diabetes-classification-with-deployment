from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predictdata', methods = ['GET','POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        data = CustomData(
            Pregnancies = int(request.form.get('Pregnancies')),
            Glucose = int(request.form.get('Glucose')),
            BloodPressure = int(request.form.get('BloodPressure')),
            SkinThickness = int(request.form.get('SkinThickness')),
            Insulin = int(request.form.get('Insulin')),
            BMI = float(request.form.get('BMI')),
            DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction')),
            Age = int(request.form.get('Age'))
        )
        
        pred_df = data.get_data_as_data_frame()
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        if results[0] == 1:
            results = "Diabetic"
        else:
            results = "Non-Diabetic"
        
        # return render_template('home.html',results=results[0])
        return render_template('home.html',results=results)
    
if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8000)
    # app.run(host = "0.0.0.0", debug = True)