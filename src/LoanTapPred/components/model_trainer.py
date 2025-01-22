import os
from src.LoanTapPred import logger
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.LoanTapPred.utils.common import save_bin
from src.LoanTapPred.utils.data_transformation_utils import scale_data
import joblib
from imblearn.over_sampling import SMOTE
from src.LoanTapPred.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config=config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        X_train = train_data.drop(columns=[self.config.target_column])
        y_train = train_data[self.config.target_column]
        
        X_test = test_data.drop(columns=[self.config.target_column])
        y_test = test_data[self.config.target_column]
        
        X_train_scaled=scale_data(X_train, os.path.join(self.config.root_dir, self.config.scaler_path))
        
        sm=SMOTE(random_state=42)
        X_train_res,y_train_res=sm.fit_resample(X_train_scaled,y_train)
        lr=LogisticRegression()
        lr.fit(X_train_res,y_train_res)
        print(lr.score(X_train_res,y_train_res))
        with open(os.path.join(self.config.model_score),'w') as f:
            f.write(f'Model score is: {lr.score(X_train_res,y_train_res)}')

        
        
        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))


        





    