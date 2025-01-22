import os
from src.LoanTapPred import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from src.LoanTapPred.utils.data_transformation_utils import *
from src.LoanTapPred.utils.common import save_bin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from src.LoanTapPred.entity.config_entity import DataTransormationConfig

class DataTransormation:
    def __init__(self, config: DataTransormationConfig):
        self.config = config

    def transform(self):
        try:
            data = pd.read_csv(self.config.data_path)
            data = missing_value_imputation(data)
            data = feature_engineering(data)
            data = nominal_feature_encoding(data, self.config.ohencoder_path)
            data = ordinal_feature_encoding(data, ordinal_features=['grade', 'sub_grade', 'emp_length'], 
                                            paths=[self.config.le_grade_path, self.config.le_subgrade_path,self.config.le_emp_length_path])
            data = drop_collinear_features(data)
            return data
        except Exception as e:
            raise e
        
    def train_test_splitting(self):
        data = self.transform()
        train,test = train_test_split(data, test_size=0.20)

        train.to_csv(os.path.join(self.config.root_dir,"train.csv"),index=False)
        test.to_csv(os.path.join(self.config.root_dir,"test.csv"),index=False)
        logger.info("Train and test files created")
        print(train.shape, test.shape)    


