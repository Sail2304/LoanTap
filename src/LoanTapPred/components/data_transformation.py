import os
from src.OLAChurnPred import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from src.OLAChurnPred.utils.data_transformation_utils import *
from src.OLAChurnPred.entity.config_entity import DataTransormationConfig

class DataTransormation:
    def __init__(self, config: DataTransormationConfig):
        self.config = config

    def transform(self):
        try:
            data = pd.read_csv(self.config.data_path)
            data = change_data_types(data)
            data = missing_value_imputation(data)
            data = group_transform_data(data)
            data = OHEncoding(data, self.config.ohencoder_path)
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
