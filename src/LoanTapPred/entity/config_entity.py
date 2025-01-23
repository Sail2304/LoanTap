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
    le_grade_path:  Path
    le_subgrade_path: Path
    le_emp_length_path: Path
    
@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    model_score: Path
    scaler_path: Path
    target_column: str
    

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    scaler_path: Path
    ohencoder_path: Path
    le_grade_path: Path
    le_subgrade_path: Path
    le_emp_length_path: Path
    metric_file_name: Path
    target_column: str
    mlflow_uri: str
