"""
Helper Functions for AI Data Analysis Assistant
"""

import pandas as pd
import numpy as np

def load_csv(file_path):
    """Load CSV file with error handling"""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"❌ Error: File '{file_path}' is empty.")
        return None
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

def convert_to_serializable(obj):
    """Convert numpy/pandas types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def get_dataset_summary(dataframe):
    """Generate comprehensive dataset summary with JSON serializable types"""
    summary = {
        "total_rows": int(len(dataframe)),
        "total_columns": int(len(dataframe.columns)),
        "column_names": dataframe.columns.tolist(),
        "data_types": dataframe.dtypes.astype(str).to_dict(),
        "missing_values": dataframe.isnull().sum().to_dict(),
        "missing_percentage": (dataframe.isnull().sum() / len(dataframe) * 100).round(2).to_dict(),
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }
    
    summary["missing_values"] = {k: int(v) for k, v in summary["missing_values"].items()}
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary["numeric_stats"] = {}
        for col in numeric_cols:
            summary["numeric_stats"][col] = {
                "min": float(dataframe[col].min()),
                "max": float(dataframe[col].max()),
                "mean": float(dataframe[col].mean()),
                "median": float(dataframe[col].median()),
                "std": float(dataframe[col].std()),
            }
    
    categorical_cols = dataframe.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        summary["categorical_stats"] = {}
        for col in categorical_cols:
            summary["categorical_stats"][col] = {
                "unique_values": int(dataframe[col].nunique()),
                "top_values": dataframe[col].value_counts().head(5).to_dict(),
            }
    
    return summary