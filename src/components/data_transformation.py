import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformation_object(self):
            try:
                logging.info("Data transformation initiated")
                # Define the numerical and categorical columns
                
                numerical_cols = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA']
                categorical_cols = ['Research']
                # Create the preprocessing pipelines for both numeric and categorical data
                numerical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore')),
                ])  

                # Combine both pipelines into a ColumnTransformer
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num_pipline', numerical_transformer, numerical_cols),
                        ('cat_pipline', categorical_transformer, categorical_cols)
                    ]
                )
                logging.info("Preprocessor object created")
                return preprocessor
            except Exception as e:
                raise CustomException(e, sys) from e
        
    def initiate_data_transformation(self, train_path, test_path):
            try:
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                logging.info("Read train and test data")
                logging.info("Obtaining preprocessing object")

                preprocessing_obj = self.get_data_transformation_object()

                target_column_name = 'Chance of Admit '
                drop_columns = [target_column_name, 'Serial No.']
                # numerical_cols = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA']
                # categorical_cols = ['Research']

                input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
                target_feature_train_df = train_df[target_column_name]

                input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
                target_feature_test_df = test_df[target_column_name]

                logging.info("Applying preprocessing object on training and testing dataframes")

                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

                logging.info("Saved preprocessing object")

                save_object(
                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj
                )

                return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path
                )
            except Exception as e:
                raise CustomException(e, sys) from e