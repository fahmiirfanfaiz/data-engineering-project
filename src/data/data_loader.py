import sys
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np

current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from config import RAW_DATA_PATH, setup_environment, PANDAS_CONFIG

class DataLoader:
    """
    Data Loader class for COVID-19 Jakarta dataset.
    Handles data loading, initial validation, and basic information extraction.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to the raw data file. If None, uses default from config.
        """
        self.data_path = data_path or RAW_DATA_PATH
        self.df = None
        self.data_info = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup environment
        setup_environment()
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the COVID-19 dataset from CSV file
        
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            Exception: If there's an error loading the data
        """
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            self.logger.info("Loading dataset...")
            self.df = pd.read_csv(self.data_path)
            
            self.logger.info(f"Dataset loaded successfully!")
            self.logger.info(f"Shape: {self.df.shape}")
            self.logger.info(f"Rows: {self.df.shape[0]}")
            self.logger.info(f"Columns: {self.df.shape[1]}")
            
            # Store basic information
            self._extract_basic_info()
            
            return self.df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _extract_basic_info(self) -> None:
        """Extract and store basic information about the dataset"""
        if self.df is None:
            return
            
        self.data_info = {
            'shape': self.df.shape,
            'rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'column_names': list(self.df.columns),
            'data_types': self.df.dtypes.to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'total_cells': np.prod(self.df.shape)
        }
    
    def get_basic_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset
        
        Returns:
            Dict containing basic dataset information
        """
        if self.df is None:
            self.logger.warning("Dataset not loaded yet. Call load_data() first.")
            return {}
            
        return self.data_info
    
    def display_info(self) -> None:
        """Display comprehensive dataset information"""
        if self.df is None:
            self.logger.warning("Dataset not loaded yet. Call load_data() first.")
            return
        
        print("=== DATASET INFORMATION ===")
        print(f"\n1. Basic Info:")
        print(f"   Shape: {self.data_info['shape']}")
        print(f"   Rows: {self.data_info['rows']:,}")
        print(f"   Columns: {self.data_info['columns']}")
        print(f"   Total cells: {self.data_info['total_cells']:,}")
        print(f"   Memory usage: {self.data_info['memory_usage'] / 1024**2:.2f} MB")
        
        print(f"\n2. Column Names:")
        for i, col in enumerate(self.data_info['column_names'], 1):
            print(f"   {i:2d}. {col}")
            
        print(f"\n3. Data Types:")
        for col, dtype in self.data_info['data_types'].items():
            print(f"   {col}: {dtype}")
    
    def get_missing_values_summary(self) -> pd.DataFrame:
        """
        Get summary of missing values in the dataset
        
        Returns:
            pd.DataFrame: Summary of missing values per column
        """
        if self.df is None:
            self.logger.warning("Dataset not loaded yet. Call load_data() first.")
            return pd.DataFrame()
        
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        
        missing_summary = pd.DataFrame({
            'Column': missing_values.index,
            'Missing_Count': missing_values.values,
            'Missing_Percentage': missing_percentage.values
        }).sort_values('Missing_Count', ascending=False)
        
        # Add total missing statistics
        total_missing = missing_values.sum()
        total_cells = np.prod(self.df.shape)
        overall_missing_pct = (total_missing / total_cells) * 100
        
        self.logger.info(f"Total missing values: {total_missing:,}")
        self.logger.info(f"Overall missing percentage: {overall_missing_pct:.2f}%")
        
        return missing_summary
    
    def get_duplicate_summary(self) -> Dict[str, int]:
        """
        Get summary of duplicate rows in the dataset
        
        Returns:
            Dict containing duplicate statistics
        """
        if self.df is None:
            self.logger.warning("Dataset not loaded yet. Call load_data() first.")
            return {}
        
        # Check complete duplicates
        complete_duplicates = self.df.duplicated().sum()
        
        # Check key column duplicates (meaningful columns)
        key_columns = ['periode_data', 'id_kel', 'nama_kelurahan', 'nama_kecamatan']
        available_key_columns = [col for col in key_columns if col in self.df.columns]
        
        key_duplicates = 0
        if available_key_columns:
            key_duplicates = self.df.duplicated(subset=available_key_columns).sum()
        
        duplicate_summary = {
            'complete_duplicates': complete_duplicates,
            'key_column_duplicates': key_duplicates,
            'key_columns_used': available_key_columns
        }
        
        self.logger.info(f"Complete duplicate rows: {complete_duplicates}")
        self.logger.info(f"Key column duplicates: {key_duplicates}")
        
        return duplicate_summary
    
    def get_data_preview(self, n_rows: int = 5) -> pd.DataFrame:
        """
        Get preview of the dataset
        
        Args:
            n_rows: Number of rows to preview
            
        Returns:
            pd.DataFrame: Preview of the dataset
        """
        if self.df is None:
            self.logger.warning("Dataset not loaded yet. Call load_data() first.")
            return pd.DataFrame()
        
        return self.df.head(n_rows)
    
    def get_statistical_summary(self) -> pd.DataFrame:
        """
        Get statistical summary of numerical columns
        
        Returns:
            pd.DataFrame: Statistical summary
        """
        if self.df is None:
            self.logger.warning("Dataset not loaded yet. Call load_data() first.")
            return pd.DataFrame()
        
        return self.df.describe()
    
    def get_categorical_summary(self) -> Dict[str, Any]:
        """
        Get summary of categorical columns
        
        Returns:
            Dict containing categorical column information
        """
        if self.df is None:
            self.logger.warning("Dataset not loaded yet. Call load_data() first.")
            return {}
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        categorical_info = {}
        
        for col in categorical_cols:
            categorical_info[col] = {
                'unique_count': self.df[col].nunique(),
                'unique_values': list(self.df[col].unique()[:10]),  # Show first 10 unique values
                'most_frequent': self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else None,
                'missing_count': self.df[col].isnull().sum()
            }
        
        return categorical_info
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """
        Validate basic data integrity
        
        Returns:
            Dict containing validation results
        """
        if self.df is None:
            self.logger.warning("Dataset not loaded yet. Call load_data() first.")
            return {}
        
        validation_results = {
            'has_missing_values': self.df.isnull().any().any(),
            'has_duplicates': self.df.duplicated().any(),
            'has_infinite_values': False,
            'negative_covid_cases': False,
            'data_type_consistency': True
        }
        
        # Check for infinite values in numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            validation_results['has_infinite_values'] = np.isinf(self.df[numerical_cols]).any().any()
        
        # Check for negative COVID-19 case counts (should be non-negative)
        covid_cols = ['positif', 'sembuh', 'meninggal', 'odp_proses', 'odp_selesai', 
                     'pdp_proses', 'pdp_selesai', 'otg_proses', 'otg_selesai']
        available_covid_cols = [col for col in covid_cols if col in self.df.columns]
        
        if available_covid_cols:
            for col in available_covid_cols:
                if self.df[col].dtype in [np.int64, np.float64]:
                    if (self.df[col] < 0).any():
                        validation_results['negative_covid_cases'] = True
                        break
        
        return validation_results

def load_covid_data(data_path: Optional[Path] = None) -> Tuple[pd.DataFrame, DataLoader]:
    """
    Convenience function to load COVID-19 data with comprehensive analysis
    
    Args:
        data_path: Path to data file. If None, uses default from config.
        
    Returns:
        Tuple of (DataFrame, DataLoader instance)
    """
    loader = DataLoader(data_path)
    df = loader.load_data()
    
    # Display comprehensive information
    loader.display_info()
    
    print("\n=== MISSING VALUES SUMMARY ===")
    missing_summary = loader.get_missing_values_summary()
    print(missing_summary)
    
    print("\n=== DUPLICATE ANALYSIS ===")
    duplicate_summary = loader.get_duplicate_summary()
    print(f"Complete duplicates: {duplicate_summary['complete_duplicates']}")
    print(f"Key column duplicates: {duplicate_summary['key_column_duplicates']}")
    
    print("\n=== DATA INTEGRITY VALIDATION ===")
    validation = loader.validate_data_integrity()
    for key, value in validation.items():
        status = "❌" if value else "✅"
        print(f"{status} {key.replace('_', ' ').title()}: {value}")
    
    return df, loader

if __name__ == "__main__":
    # Test the data loader
    print("Testing Data Loader...")
    df, loader = load_covid_data()
    
    print(f"\n=== DATA PREVIEW ===")
    print(loader.get_data_preview())
    
    print(f"\n=== STATISTICAL SUMMARY ===")
    print(loader.get_statistical_summary())