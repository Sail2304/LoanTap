import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from src.OLAChurnPred.utils.common import save_bin
import joblib
from src.OLAChurnPred.entity.config_entity import ModelTrainerConfig


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
        random_grid = {
               'n_estimators': self.config.n_estimators,
               'max_features': [None,'sqrt'],
               'max_depth': self.config.max_depth,
               'min_samples_split': self.config.min_samples_split,
               'min_samples_leaf': self.config.min_samples_leaf,
               'learning_rate': self.config.learning_rate
               }
        
        gbc = GradientBoostingClassifier()
        gbc_randomcv = RandomizedSearchCV(
                                        estimator=gbc,
                                        param_distributions=random_grid,
                                        n_iter=100,
                                        cv=4,
                                        random_state=41, 
                                        n_jobs = -1,
                                        verbose=3,error_score='raise'
                                        )
        
        gbc_randomcv.fit(X_train,y_train)
        with open(os.path.join(self.config.model_score),'w') as f:
            f.write(f'Best model score is: {gbc_randomcv.best_score_}\n')
            f.write(f'Best model parameters are: {gbc_randomcv.best_params_}')

        
        
        joblib.dump(gbc_randomcv.best_estimator_, os.path.join(self.config.root_dir, self.config.model_name))


        



