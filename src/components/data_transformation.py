import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    
    def get_data_transformer_object(self):
        try:

            num_columns = ["writing_score","reading_score"]
            cat_columns = [ 
                    "gender",
                    "race_ethnicity",
                    "parental_level_of_education",
                    "lunch",
                    "test_preparation_course"]
            
            logging.info(f"Categorical Columns: {cat_columns}")
            logging.info(f"Numerical Columns: {num_columns}")

            num_pipline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipline,num_columns),
                    ("cat_pipeline",cat_pipeline,cat_columns)
                ]
            )

            return preprocessor
    
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data finished")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            logging.info("Started: Seperating input feature and target feature of train_df")
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df['math_score']
            logging.info("Finished: Seperating input feature and target feature of train_df")

            logging.info("Started: Seperating input feature and target feature of test_df")
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df['math_score']
            logging.info("Finished: Seperating input feature and target feature of test_df")


            logging.info("Applying preprocessing object on train and test dataset")
            input_feature_train_arr = preprocessing_obj.fit_transform(train_df)
            input_feature_test_arr = preprocessing_obj.transform(test_df)
            logging.info("Finished Applying preprocessing object on train and test dataset")

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
            )
            logging.info("saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)

