# Stub file to help VS Code IntelliSense resolve imports
# This file provides type hints for dynamically imported modules

from pathlib import Path
from typing import Tuple, Any, Dict, Optional
import pandas as pd

# Stub for data_loader module functions
def load_covid_data(data_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Any]:
    """
    Load COVID-19 data with comprehensive analysis
    
    Args:
        data_path: Path to data file. If None, uses default from config.
        
    Returns:
        Tuple of (DataFrame, DataLoader instance)
    """
    pass

# Stub for data_cleaner module functions  
def clean_covid_data(df: pd.DataFrame, 
                     missing_strategy: str = 'auto',
                     outlier_method: str = 'cap') -> Tuple[pd.DataFrame, Any]:
    """
    Clean COVID-19 data with specified strategies
    
    Args:
        df: Input DataFrame
        missing_strategy: Strategy for handling missing values
        outlier_method: Method for handling outliers
        
    Returns:
        Tuple of (cleaned DataFrame, DataCleaner instance)
    """
    pass

# Stub for config variables
IMG_CLEANING_DIR: Path