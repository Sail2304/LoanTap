from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass
class DataValidationConfig:
    root_dir:Path
    STATUS_FILE:str
    unzip_data_dir:Path
    all_schema:dict


@dataclass
class DataTransormationConfig:
    root_dir: Path
    data_path: Path
    ohencoder_path: Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    model_score: Path
    learning_rate: list
    max_depth: list
    max_features: list
    min_samples_leaf: list
    min_samples_split: list 
    n_estimators: list
    target_column: str
    



@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    ohencoder_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str