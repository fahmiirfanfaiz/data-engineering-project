# Data Cleaning Module for COVID-19 Jakarta Dataset
import sys
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any

# Handle imports with try-except for better error handling
try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler
except ImportError as e:
    print(f"Error: Required dependencies not found: {e}")
    print("Please install with: pip install pandas numpy scikit-learn")

# Add parent directory to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from config import (
    CLEANED_DIR, CLEANING_CONFIG, COVID_COLUMNS, ESSENTIAL_COLUMNS,
    KEY_COLUMNS, setup_environment
)

class DataCleaner:
    """
    Data Cleaner class for COVID-19 Jakarta dataset.
    Handles missing values, outliers, noisy data, and basic preprocessing.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataCleaner
        
        Args:
            df: Input DataFrame to be cleaned
        """
        self.df_original = df.copy()
        self.df = df.copy()
        self.cleaning_log = []
        self.outliers_info = {}
        self.missing_info = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup environment
        setup_environment()
        
    def analyze_missing_values(self) -> pd.DataFrame:
        """
        Analyze missing values in the dataset
        
        Returns:
            pd.DataFrame: Summary of missing values
        """
        self.logger.info("Analyzing missing values...")
        
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        
        missing_analysis = pd.DataFrame({
            'Column': missing_values.index,
            'Missing_Count': missing_values.values,
            'Missing_Percentage': missing_percentage.values
        }).sort_values('Missing_Count', ascending=False)
        
        self.missing_info = {
            'total_missing': missing_values.sum(),
            'columns_with_missing': missing_analysis[missing_analysis['Missing_Count'] > 0]['Column'].tolist(),
            'analysis': missing_analysis
        }
        
        self.logger.info(f"Found {self.missing_info['total_missing']} missing values")
        self.logger.info(f"Columns with missing values: {self.missing_info['columns_with_missing']}")
        
        return missing_analysis
    
    def detect_outliers_iqr(self, column: str) -> Tuple[pd.DataFrame, float, float]:
        """
        Detect outliers using IQR method
        
        Args:
            column: Column name to analyze
            
        Returns:
            Tuple of (outliers_df, lower_bound, upper_bound)
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - CLEANING_CONFIG['outlier_threshold'] * IQR
        upper_bound = Q3 + CLEANING_CONFIG['outlier_threshold'] * IQR
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    def analyze_outliers(self) -> Dict[str, Any]:
        """
        Analyze outliers in COVID-19 related columns
        
        Returns:
            Dict containing outlier analysis results
        """
        self.logger.info("Analyzing outliers...")
        
        # Get available COVID columns
        available_covid_cols = [col for col in COVID_COLUMNS if col in self.df.columns]
        
        outliers_summary = {}
        
        for col in available_covid_cols:
            if self.df[col].dtype in [np.int64, np.float64]:
                outliers, lower, upper = self.detect_outliers_iqr(col)
                
                outliers_summary[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(self.df)) * 100,
                    'lower_bound': lower,
                    'upper_bound': upper,
                    'outlier_values': sorted(outliers[col].unique()) if len(outliers) > 0 else []
                }
        
        self.outliers_info = outliers_summary
        return outliers_summary
    
    def analyze_duplicates(self) -> Dict[str, Any]:
        """
        Analyze duplicate rows in the dataset
        
        Returns:
            Dict containing duplicate analysis
        """
        self.logger.info("Analyzing duplicates...")
        
        # Complete duplicates
        complete_duplicates = self.df.duplicated().sum()
        
        # Key column duplicates
        available_key_cols = [col for col in KEY_COLUMNS if col in self.df.columns]
        key_duplicates = 0
        duplicate_data = pd.DataFrame()
        
        if available_key_cols:
            key_duplicates = self.df.duplicated(subset=available_key_cols).sum()
            if key_duplicates > 0:
                duplicate_data = self.df[self.df.duplicated(subset=available_key_cols, keep=False)].sort_values(available_key_cols)
        
        return {
            'complete_duplicates': complete_duplicates,
            'key_column_duplicates': key_duplicates,
            'key_columns_used': available_key_cols,
            'duplicate_data': duplicate_data
        }
    
    def drop_unnecessary_columns(self, columns_to_drop: Optional[List[str]] = None) -> None:
        """
        Drop unnecessary columns (e.g., columns with 100% missing values)
        
        Args:
            columns_to_drop: List of column names to drop. If None, auto-detect.
        """
        self.logger.info("Dropping unnecessary columns...")
        
        if columns_to_drop is None:
            # Auto-detect columns to drop
            columns_to_drop = []
            
            # Drop columns with 100% missing values
            for col in self.df.columns:
                if self.df[col].isnull().sum() == len(self.df):
                    columns_to_drop.append(col)
            
            # Check for 'keterangan' column (typically 100% missing in this dataset)
            if 'keterangan' in self.df.columns:
                columns_to_drop.append('keterangan')
        
        if columns_to_drop:
            cols_before = self.df.shape[1]
            self.df = self.df.drop(columns=columns_to_drop)
            cols_after = self.df.shape[1]
            
            self.cleaning_log.append(f"Dropped {len(columns_to_drop)} columns: {columns_to_drop}")
            self.logger.info(f"Dropped columns: {columns_to_drop}")
            self.logger.info(f"Columns: {cols_before} -> {cols_after}")
        else:
            self.logger.info("No unnecessary columns found to drop")
    
    def handle_missing_values(self, strategy: str = 'auto') -> None:
        """
        Handle missing values based on strategy
        
        Args:
            strategy: Strategy for handling missing values ('drop', 'fill_zero', 'fill_median', 'auto')
        """
        self.logger.info(f"Handling missing values with strategy: {strategy}")
        
        missing_before = self.df.isnull().sum().sum()
        
        if strategy == 'auto':
            # Auto strategy: fill COVID columns with 0, drop others if high missing %
            covid_cols = [col for col in COVID_COLUMNS if col in self.df.columns]
            
            # Fill COVID columns with 0 (assuming 0 cases when missing)
            for col in covid_cols:
                if self.df[col].isnull().any():
                    self.df[col] = self.df[col].fillna(0)
                    self.cleaning_log.append(f"Filled missing values in {col} with 0")
            
            # Drop columns with high missing percentage (> threshold)
            threshold = CLEANING_CONFIG['missing_threshold']
            high_missing_cols = []
            for col in self.df.columns:
                missing_pct = self.df[col].isnull().sum() / len(self.df)
                if missing_pct > threshold and col not in ESSENTIAL_COLUMNS:
                    high_missing_cols.append(col)
            
            if high_missing_cols:
                self.df = self.df.drop(columns=high_missing_cols)
                self.cleaning_log.append(f"Dropped columns with >{threshold*100}% missing: {high_missing_cols}")
        
        elif strategy == 'drop':
            # Drop rows with any missing values
            self.df = self.df.dropna()
            self.cleaning_log.append("Dropped rows with missing values")
        
        elif strategy == 'fill_zero':
            # Fill all missing values with 0
            self.df = self.df.fillna(0)
            self.cleaning_log.append("Filled all missing values with 0")
        
        elif strategy == 'fill_median':
            # Fill numerical columns with median, categorical with mode
            for col in self.df.columns:
                if self.df[col].isnull().any():
                    if self.df[col].dtype in [np.int64, np.float64]:
                        self.df[col] = self.df[col].fillna(self.df[col].median())
                    else:
                        mode_value = self.df[col].mode()
                        if len(mode_value) > 0:
                            self.df[col] = self.df[col].fillna(mode_value.iloc[0])
            self.cleaning_log.append("Filled numerical columns with median, categorical with mode")
        
        missing_after = self.df.isnull().sum().sum()
        self.logger.info(f"Missing values: {missing_before} -> {missing_after}")
    
    def handle_outliers(self, method: str = 'cap') -> None:
        """
        Handle outliers in COVID-19 columns
        
        Args:
            method: Method for handling outliers ('cap', 'remove', 'log_transform', 'none')
        """
        if method == 'none':
            self.logger.info("Skipping outlier handling")
            return
        
        self.logger.info(f"Handling outliers with method: {method}")
        
        covid_cols = [col for col in COVID_COLUMNS if col in self.df.columns]
        outliers_handled = 0
        
        for col in covid_cols:
            if self.df[col].dtype in [np.int64, np.float64]:
                outliers, lower, upper = self.detect_outliers_iqr(col)
                
                if len(outliers) > 0:
                    if method == 'cap':
                        # Cap outliers to bounds
                        self.df.loc[self.df[col] < lower, col] = lower
                        self.df.loc[self.df[col] > upper, col] = upper
                        outliers_handled += len(outliers)
                        self.cleaning_log.append(f"Capped {len(outliers)} outliers in {col}")
                    
                    elif method == 'remove':
                        # Remove outlier rows
                        before_rows = len(self.df)
                        self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
                        after_rows = len(self.df)
                        removed = before_rows - after_rows
                        outliers_handled += removed
                        self.cleaning_log.append(f"Removed {removed} outlier rows based on {col}")
                    
                    elif method == 'log_transform':
                        # Log transform to reduce outlier impact
                        if (self.df[col] > 0).all():
                            self.df[col] = np.log1p(self.df[col])
                            self.cleaning_log.append(f"Applied log transform to {col}")
        
        self.logger.info(f"Handled {outliers_handled} outliers using {method} method")
    
    def remove_duplicates(self) -> None:
        """Remove duplicate rows from the dataset"""
        self.logger.info("Removing duplicate rows...")
        
        before_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        after_rows = len(self.df)
        removed = before_rows - after_rows
        
        if removed > 0:
            self.cleaning_log.append(f"Removed {removed} duplicate rows")
            self.logger.info(f"Removed {removed} duplicate rows")
        else:
            self.logger.info("No duplicate rows found")
    
    def validate_data_consistency(self) -> Dict[str, Any]:
        """
        Validate data consistency and logical constraints
        
        Returns:
            Dict containing validation results
        """
        self.logger.info("Validating data consistency...")
        
        validation_results = {
            'negative_covid_cases': [],
            'inconsistent_odp': 0,
            'inconsistent_positif': [],
            'total_check': True
        }
        
        # Check for negative COVID cases
        covid_cols = [col for col in COVID_COLUMNS if col in self.df.columns]
        for col in covid_cols:
            if self.df[col].dtype in [np.int64, np.float64]:
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    validation_results['negative_covid_cases'].append((col, negative_count))
        
        # Check ODP consistency (if columns exist)
        if all(col in self.df.columns for col in ['odp_proses', 'odp_selesai']):
            # odp_total should equal odp_proses + odp_selesai
            if 'odp' in self.df.columns:
                inconsistent = (self.df['odp'] != self.df['odp_proses'] + self.df['odp_selesai']).sum()
                validation_results['inconsistent_odp'] = inconsistent
        
        # Check total consistency (if total columns exist)
        if 'positif' in self.df.columns:
            positif_components = ['dirawat', 'sembuh', 'meninggal']
            available_components = [col for col in positif_components if col in self.df.columns]
            
            if len(available_components) >= 2:
                # Check if positif >= sum of components (allowing for missing components)
                component_sum = self.df[available_components].sum(axis=1)
                inconsistent_rows = self.df[self.df['positif'] < component_sum]
                validation_results['inconsistent_positif'] = len(inconsistent_rows)
        
        # Overall validation status
        validation_results['total_check'] = (
            len(validation_results['negative_covid_cases']) == 0 and
            validation_results['inconsistent_odp'] == 0 and
            validation_results['inconsistent_positif'] == 0
        )
        
        return validation_results
    
    def encode_categorical_variables(self) -> None:
        """Encode categorical variables"""
        self.logger.info("Encoding categorical variables...")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in KEY_COLUMNS:  # Don't encode key identifier columns
                if CLEANING_CONFIG['categorical_encoding'] == 'label':
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
                    self.cleaning_log.append(f"Label encoded {col}")
                # Add other encoding methods as needed
    
    def scale_numerical_features(self) -> None:
        """Scale numerical features"""
        self.logger.info("Scaling numerical features...")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if CLEANING_CONFIG['numerical_scaling'] == 'standard':
            scaler = StandardScaler()
            self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])
            self.cleaning_log.append("Applied StandardScaler to numerical features")
    
    def clean_data(self, 
                   drop_unnecessary: bool = True,
                   missing_strategy: str = 'auto',
                   outlier_method: str = 'cap',
                   remove_duplicates: bool = True,
                   encode_categorical: bool = False,
                   scale_numerical: bool = False) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning
        
        Args:
            drop_unnecessary: Whether to drop unnecessary columns
            missing_strategy: Strategy for handling missing values
            outlier_method: Method for handling outliers
            remove_duplicates: Whether to remove duplicate rows
            encode_categorical: Whether to encode categorical variables
            scale_numerical: Whether to scale numerical features
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        self.logger.info("Starting comprehensive data cleaning...")
        
        # 1. Analyze current state
        self.analyze_missing_values()
        self.analyze_outliers()
        
        # 2. Drop unnecessary columns
        if drop_unnecessary:
            self.drop_unnecessary_columns()
        
        # 3. Handle missing values
        self.handle_missing_values(missing_strategy)
        
        # 4. Remove duplicates
        if remove_duplicates:
            self.remove_duplicates()
        
        # 5. Handle outliers
        self.handle_outliers(outlier_method)
        
        # 6. Validate data consistency
        validation = self.validate_data_consistency()
        
        # 7. Encode categorical variables (if requested)
        if encode_categorical:
            self.encode_categorical_variables()
        
        # 8. Scale numerical features (if requested)
        if scale_numerical:
            self.scale_numerical_features()
        
        self.logger.info("Data cleaning completed!")
        self.logger.info(f"Original shape: {self.df_original.shape}")
        self.logger.info(f"Cleaned shape: {self.df.shape}")
        
        return self.df
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """
        Get comprehensive cleaning report
        
        Returns:
            Dict containing cleaning statistics and log
        """
        return {
            'original_shape': self.df_original.shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': self.df_original.shape[0] - self.df.shape[0],
            'columns_removed': self.df_original.shape[1] - self.df.shape[1],
            'cleaning_log': self.cleaning_log,
            'missing_info': self.missing_info,
            'outliers_info': self.outliers_info
        }
    
    def save_cleaned_data(self, 
                         basic_filename: str = 'covid19_jakarta_mei2020_cleaned_basic.csv',
                         full_filename: str = 'covid19_jakarta_mei2020_cleaned_full.csv') -> Tuple[Path, Path]:
        """
        Save cleaned data to files
        
        Args:
            basic_filename: Filename for basic cleaned data
            full_filename: Filename for full processed data
            
        Returns:
            Tuple of (basic_path, full_path)
        """
        CLEANED_DIR.mkdir(parents=True, exist_ok=True)
        
        # Basic cleaned data (minimal processing)
        basic_path = CLEANED_DIR / basic_filename
        df_basic = self.df.copy()
        df_basic.to_csv(basic_path, index=False)
        
        # Full processed data (with encoding and scaling if applied)
        full_path = CLEANED_DIR / full_filename
        self.df.to_csv(full_path, index=False)
        
        self.logger.info(f"Saved basic cleaned data to: {basic_path}")
        self.logger.info(f"Saved full processed data to: {full_path}")
        
        return basic_path, full_path

def clean_covid_data(df: pd.DataFrame, 
                     missing_strategy: str = 'auto',
                     outlier_method: str = 'cap') -> Tuple[pd.DataFrame, DataCleaner]:
    """
    Convenience function to clean COVID-19 data
    
    Args:
        df: Input DataFrame
        missing_strategy: Strategy for handling missing values
        outlier_method: Method for handling outliers
        
    Returns:
        Tuple of (cleaned_df, cleaner_instance)
    """
    cleaner = DataCleaner(df)
    
    # Perform cleaning
    cleaned_df = cleaner.clean_data(
        missing_strategy=missing_strategy,
        outlier_method=outlier_method
    )
    
    # Display report
    report = cleaner.get_cleaning_report()
    print("=== DATA CLEANING REPORT ===")
    print(f"Original shape: {report['original_shape']}")
    print(f"Cleaned shape: {report['cleaned_shape']}")
    print(f"Rows removed: {report['rows_removed']}")
    print(f"Columns removed: {report['columns_removed']}")
    
    print("\n=== CLEANING LOG ===")
    for i, log_entry in enumerate(report['cleaning_log'], 1):
        print(f"{i}. {log_entry}")
    
    return cleaned_df, cleaner

if __name__ == "__main__":
    # Test the data cleaner
    from data.data_loader import load_covid_data
    
    print("Testing Data Cleaner...")
    df, loader = load_covid_data()
    
    cleaned_df, cleaner = clean_covid_data(df)
    
    # Save cleaned data
    basic_path, full_path = cleaner.save_cleaned_data()
    print(f"\nCleaned data saved to:")
    print(f"Basic: {basic_path}")
    print(f"Full: {full_path}")