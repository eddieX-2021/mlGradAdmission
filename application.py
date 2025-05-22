from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/predict_admission', methods=['GET', 'POST'])
def predict_admission():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Print form data for debugging
            print("Form Data:", request.form)
            
            data = CustomData(
                GRE_Score=int(request.form.get('gre_score')),
                TOEFL_Score=int(request.form.get('toefl_score')),
                University_Rating=int(request.form.get('university_rating')),
                SOP=float(request.form.get('sop')),
                LOR=float(request.form.get('lor')),
                CGPA=float(request.form.get('cgpa')),
                Research=int(request.form.get('research'))
            )
            pred_df = data.get_data_as_data_frame()
            print("DataFrame created:\n", pred_df)
            
            predict_pipeline = PredictPipeline()
            print("Pipeline created")
            
            results = predict_pipeline.predict(pred_df)
            print("Prediction results:", results)
            
            result_percentage = round(results[0] * 100, 2)
            return render_template('home.html', result=result_percentage)
        except Exception as e:
            print("\n\n!!! ERROR DETAILS !!!")
            print("Type:", type(e))
            print("Message:", str(e))
            print("Traceback:")
            import traceback
            traceback.print_exc()
            print("\n\n")
            return render_template('home.html', error=str(e))

    

if __name__ == "__main__":
    app.run(host="0.0.0.0")
    