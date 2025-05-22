import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
class CustomData:
    def __init__(self, GRE_Score: int, TOEFL_Score: int, University_Rating: int, SOP: float, LOR: float, CGPA: float, Research: int):
        self.GRE_Score = GRE_Score
        self.TOEFL_Score = TOEFL_Score
        self.University_Rating = University_Rating
        self.SOP = SOP
        self.LOR = LOR
        self.CGPA = CGPA
        self.Research = Research
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "GRE Score": [self.GRE_Score],
                "TOEFL Score": [self.TOEFL_Score],
                "University Rating": [self.University_Rating],
                "SOP": [self.SOP],
                "LOR": [self.LOR],
                "CGPA": [self.CGPA],
                "Research": [self.Research]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)