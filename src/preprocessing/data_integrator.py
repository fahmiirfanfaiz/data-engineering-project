# Data Integration Module for COVID-19 Jakarta Dataset
import sys
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Handle imports with try-except for better error handling
try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.feature_selection import VarianceThreshold
except ImportError as e:
    print(f"Error: Required dependencies not found: {e}")
    print("Please install with: pip install pandas numpy scikit-learn")

# Add parent directory to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from config import (
    INTEGRATED_DIR, INTEGRATION_CONFIG, COVID_COLUMNS, KEY_COLUMNS,
    TARGET_VARIABLES, setup_environment
)

class DataIntegrator:
    """
    Data Integration class for COVID-19 Jakarta dataset.
    Handles correlation analysis, covariance analysis, feature engineering,
    and redundancy removal.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataIntegrator
        
        Args:
            df: Input cleaned DataFrame
        """
        self.df_original = df.copy()
        self.df = df.copy()
        self.integration_log = []
        self.correlation_analysis = {}
        self.covariance_analysis = {}
        self.feature_engineering_results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup environment
        setup_environment()
        
        # Get available COVID columns
        self.covid_numeric_cols = [col for col in COVID_COLUMNS if col in self.df.columns and self.df[col].dtype in [np.int64, np.float64]]
        self.logger.info(f"Found {len(self.covid_numeric_cols)} COVID-19 numeric columns")
    
    def analyze_correlation(self) -> Dict[str, Any]:
        """
        Analyze correlation between variables to identify redundancy
        
        Returns:
            Dict containing correlation analysis results
        """
        self.logger.info("Analyzing correlation matrix...")
        
        # Calculate correlation matrix
        correlation_matrix = self.df[self.covid_numeric_cols].corr()
        
        # Find high correlation pairs
        high_threshold = INTEGRATION_CONFIG['correlation_threshold']
        high_corr_pairs = []
        moderate_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                var1 = correlation_matrix.columns[i]
                var2 = correlation_matrix.columns[j]
                
                if abs(corr_val) > high_threshold:
                    high_corr_pairs.append((var1, var2, corr_val))
                elif 0.5 <= abs(corr_val) <= high_threshold:
                    moderate_corr_pairs.append((var1, var2, corr_val))
        
        # Categorize variables for analysis
        categories = {
            'Suspek_ODP': ['odp', 'proses_pemantauan', 'selesai_pemantauan'],
            'Probable_PDP': ['pdp', 'masih_dirawat', 'pulang_dan_sehat'],
            'Confirmed': ['positif', 'dirawat', 'sembuh', 'meninggal', 'self_isolation']
        }
        
        category_correlations = {}
        for category, variables in categories.items():
            available_vars = [var for var in variables if var in self.covid_numeric_cols]
            if len(available_vars) > 1:
                cat_corr = correlation_matrix.loc[available_vars, available_vars]
                category_correlations[category] = cat_corr
        
        self.correlation_analysis = {
            'correlation_matrix': correlation_matrix,
            'high_corr_pairs': high_corr_pairs,
            'moderate_corr_pairs': moderate_corr_pairs,
            'category_correlations': category_correlations,
            'high_threshold': high_threshold
        }
        
        self.logger.info(f"Found {len(high_corr_pairs)} high correlation pairs (|r| > {high_threshold})")
        self.logger.info(f"Found {len(moderate_corr_pairs)} moderate correlation pairs")
        
        return self.correlation_analysis
    
    def analyze_covariance(self) -> Dict[str, Any]:
        """
        Analyze covariance to understand linear relationships
        
        Returns:
            Dict containing covariance analysis results
        """
        self.logger.info("Analyzing covariance matrix...")
        
        # Calculate covariance matrix
        covariance_matrix = self.df[self.covid_numeric_cols].cov()
        
        # Standardize data for better covariance interpretation
        scaler = StandardScaler()
        covid_data_scaled = scaler.fit_transform(self.df[self.covid_numeric_cols])
        covid_df_scaled = pd.DataFrame(covid_data_scaled, columns=self.covid_numeric_cols)
        covariance_scaled = covid_df_scaled.cov()
        
        # Eigenvalue decomposition for PCA insights
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_scaled)
        eigenvalues = eigenvalues[::-1]  # Sort descending
        eigenvectors = eigenvectors[:, ::-1]
        explained_variance_ratio = eigenvalues / eigenvalues.sum()
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Find high covariance pairs
        cov_threshold = INTEGRATION_CONFIG['covariance_threshold']
        high_cov_pairs = []
        
        for i in range(len(covariance_scaled.columns)):
            for j in range(i+1, len(covariance_scaled.columns)):
                cov_val = covariance_scaled.iloc[i, j]
                if abs(cov_val) > cov_threshold:
                    var1 = covariance_scaled.columns[i]
                    var2 = covariance_scaled.columns[j]
                    high_cov_pairs.append((var1, var2, cov_val))
        
        # Variance analysis
        variances = self.df[self.covid_numeric_cols].var().sort_values(ascending=False)
        
        self.covariance_analysis = {
            'covariance_matrix': covariance_matrix,
            'covariance_scaled': covariance_scaled,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'high_cov_pairs': high_cov_pairs,
            'variances': variances,
            'scaler': scaler
        }
        
        self.logger.info(f"PC1 explains {explained_variance_ratio[0]:.1%} of variance")
        self.logger.info(f"Found {len(high_cov_pairs)} high covariance pairs")
        
        return self.covariance_analysis
    
    def remove_low_variance_features(self) -> None:
        """Remove features with low variance"""
        self.logger.info("Removing low variance features...")
        
        threshold = INTEGRATION_CONFIG['variance_threshold']
        selector = VarianceThreshold(threshold=threshold)
        
        # Apply to numeric columns only
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        X_filtered = selector.fit_transform(self.df[numeric_cols])
        
        # Get selected feature names
        selected_features = numeric_cols[selector.get_support()]
        removed_features = numeric_cols[~selector.get_support()]
        
        if len(removed_features) > 0:
            # Update dataframe
            self.df = self.df.drop(columns=removed_features)
            self.integration_log.append(f"Removed {len(removed_features)} low variance features: {list(removed_features)}")
            self.logger.info(f"Removed {len(removed_features)} low variance features")
        else:
            self.logger.info("No low variance features found to remove")
    
    def create_risk_components(self) -> None:
        """Create risk-based engineered features"""
        if not INTEGRATION_CONFIG['feature_engineering']['risk_components']:
            return
        
        self.logger.info("Creating risk-based features...")
        
        risk_features = []
        
        # Total cases per area
        if 'positif' in self.df.columns:
            self.df['total_kasus'] = self.df['positif']
            risk_features.append('total_kasus')
        
        # Recovery rate (if both sembuh and positif exist)
        if all(col in self.df.columns for col in ['sembuh', 'positif']):
            self.df['recovery_rate'] = np.where(
                self.df['positif'] > 0,
                self.df['sembuh'] / self.df['positif'],
                0
            )
            risk_features.append('recovery_rate')
        
        # Mortality rate (if both meninggal and positif exist)
        if all(col in self.df.columns for col in ['meninggal', 'positif']):
            self.df['mortality_rate'] = np.where(
                self.df['positif'] > 0,
                self.df['meninggal'] / self.df['positif'],
                0
            )
            risk_features.append('mortality_rate')
        
        # Active cases (if dirawat exists)
        if 'dirawat' in self.df.columns:
            self.df['active_cases'] = self.df['dirawat']
            risk_features.append('active_cases')
        
        # Risk score (composite measure)
        risk_components = {}
        if 'positif' in self.df.columns:
            risk_components['positif'] = self.df['positif']
        if 'mortality_rate' in self.df.columns:
            risk_components['mortality_rate'] = self.df['mortality_rate']
        if 'active_cases' in self.df.columns:
            risk_components['active_cases'] = self.df['active_cases']
        
        if risk_components:
            # Normalize components and create composite risk score
            scaler = StandardScaler()
            risk_data = pd.DataFrame(risk_components)
            risk_scaled = scaler.fit_transform(risk_data.fillna(0))
            self.df['risk_score'] = np.mean(risk_scaled, axis=1)
            risk_features.append('risk_score')
        
        if risk_features:
            self.integration_log.append(f"Created {len(risk_features)} risk-based features: {risk_features}")
            self.logger.info(f"Created {len(risk_features)} risk-based features")
        
        self.feature_engineering_results['risk_features'] = risk_features
    
    def create_statistical_features(self) -> None:
        """Create statistical engineered features"""
        if not INTEGRATION_CONFIG['feature_engineering']['statistical_features']:
            return
        
        self.logger.info("Creating statistical features...")
        
        statistical_features = []
        
        # Per-district (kecamatan) statistics if available
        if 'nama_kecamatan' in self.df.columns and 'positif' in self.df.columns:
            # Calculate kecamatan-level statistics
            kecamatan_stats = self.df.groupby('nama_kecamatan')['positif'].agg([
                'mean', 'std', 'min', 'max', 'sum'
            ]).add_prefix('kecamatan_positif_')
            
            # Merge back to original dataframe
            self.df = self.df.merge(
                kecamatan_stats,
                left_on='nama_kecamatan',
                right_index=True,
                how='left'
            )
            
            statistical_features.extend(kecamatan_stats.columns)
            
            # Relative position within kecamatan
            self.df['positif_vs_kecamatan_mean'] = (
                self.df['positif'] / self.df['kecamatan_positif_mean']
            ).fillna(0)
            statistical_features.append('positif_vs_kecamatan_mean')
        
        # Percentile rankings for key COVID variables
        covid_vars_for_percentile = ['positif', 'sembuh', 'meninggal']
        available_covid_vars = [var for var in covid_vars_for_percentile if var in self.df.columns]
        
        for var in available_covid_vars:
            percentile_col = f'{var}_percentile'
            self.df[percentile_col] = self.df[var].rank(pct=True) * 100
            statistical_features.append(percentile_col)
        
        # COVID burden indicators
        if all(col in self.df.columns for col in ['positif', 'sembuh', 'meninggal']):
            self.df['covid_burden'] = (
                self.df['positif'] * 1.0 +
                self.df['meninggal'] * 3.0 +  # Weight mortality higher
                (self.df['positif'] - self.df['sembuh'] - self.df['meninggal']) * 1.5  # Active cases
            )
            statistical_features.append('covid_burden')
        
        if statistical_features:
            self.integration_log.append(f"Created {len(statistical_features)} statistical features: {statistical_features}")
            self.logger.info(f"Created {len(statistical_features)} statistical features")
        
        self.feature_engineering_results['statistical_features'] = statistical_features
    
    def process_categorical_features(self) -> None:
        """Process categorical features for integration"""
        if not INTEGRATION_CONFIG['feature_engineering']['categorical_features']:
            return
        
        self.logger.info("Processing categorical features...")
        
        categorical_features = []
        
        # Encode categorical variables
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in KEY_COLUMNS]
        
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                # Label encoding for categorical variables
                le = LabelEncoder()
                encoded_col = f'{col}_encoded'
                self.df[encoded_col] = le.fit_transform(self.df[col].astype(str))
                categorical_features.append(encoded_col)
                
                # Create dummy variables for categorical with few unique values
                if self.df[col].nunique() <= 10:
                    dummies = pd.get_dummies(self.df[col], prefix=col)
                    self.df = pd.concat([self.df, dummies], axis=1)
                    categorical_features.extend(dummies.columns.tolist())
        
        if categorical_features:
            self.integration_log.append(f"Processed {len(categorical_features)} categorical features")
            self.logger.info(f"Processed {len(categorical_features)} categorical features")
        
        self.feature_engineering_results['categorical_features'] = categorical_features
    
    def identify_redundant_features(self) -> List[str]:
        """
        Identify redundant features based on correlation and covariance analysis
        
        Returns:
            List of redundant feature names to consider for removal
        """
        self.logger.info("Identifying redundant features...")
        
        redundant_features = []
        
        # Based on high correlation pairs
        high_corr_pairs = self.correlation_analysis.get('high_corr_pairs', [])
        for var1, var2, corr in high_corr_pairs:
            # Keep the variable with higher variance or more interpretable name
            var1_var = self.df[var1].var()
            var2_var = self.df[var2].var()
            
            if var1_var < var2_var:
                redundant_features.append(var1)
            else:
                redundant_features.append(var2)
        
        # Remove duplicates
        redundant_features = list(set(redundant_features))
        
        self.logger.info(f"Identified {len(redundant_features)} potentially redundant features")
        
        return redundant_features
    
    def integrate_data(self, 
                      remove_low_variance: bool = True,
                      create_risk_features: bool = True,
                      create_statistical_features: bool = True,
                      process_categorical: bool = True) -> pd.DataFrame:
        """
        Perform comprehensive data integration
        
        Args:
            remove_low_variance: Whether to remove low variance features
            create_risk_features: Whether to create risk-based features
            create_statistical_features: Whether to create statistical features
            process_categorical: Whether to process categorical features
            
        Returns:
            pd.DataFrame: Integrated dataset
        """
        self.logger.info("Starting comprehensive data integration...")
        
        # 1. Analyze correlation and covariance
        self.analyze_correlation()
        self.analyze_covariance()
        
        # 2. Remove low variance features
        if remove_low_variance:
            self.remove_low_variance_features()
        
        # 3. Feature engineering
        if create_risk_features:
            self.create_risk_components()
        
        if create_statistical_features:
            self.create_statistical_features()
        
        if process_categorical:
            self.process_categorical_features()
        
        # 4. Identify redundant features (for information, not automatic removal)
        redundant_features = self.identify_redundant_features()
        
        self.logger.info("Data integration completed!")
        self.logger.info(f"Original shape: {self.df_original.shape}")
        self.logger.info(f"Integrated shape: {self.df.shape}")
        
        return self.df
    
    def get_integration_report(self) -> Dict[str, Any]:
        """
        Get comprehensive integration report
        
        Returns:
            Dict containing integration statistics and analysis
        """
        # Get new features created
        original_features = set(self.df_original.columns)
        current_features = set(self.df.columns)
        new_features = current_features - original_features
        
        return {
            'original_shape': self.df_original.shape,
            'integrated_shape': self.df.shape,
            'features_added': len(new_features),
            'new_features': list(new_features),
            'integration_log': self.integration_log,
            'correlation_analysis': self.correlation_analysis,
            'covariance_analysis': {
                'n_components_80pct': np.sum(self.covariance_analysis.get('cumulative_variance', []) <= 0.8) + 1,
                'pc1_variance': self.covariance_analysis.get('explained_variance_ratio', [0])[0],
                'high_cov_pairs_count': len(self.covariance_analysis.get('high_cov_pairs', []))
            },
            'feature_engineering': self.feature_engineering_results
        }
    
    def save_integrated_data(self) -> Tuple[Path, Path, Path]:
        """
        Save integrated data in multiple formats
        
        Returns:
            Tuple of (full_path, analysis_path, ml_ready_path)
        """
        INTEGRATED_DIR.mkdir(parents=True, exist_ok=True)
        
        # Full integrated dataset
        full_path = INTEGRATED_DIR / 'covid19_jakarta_mei2020_integrated_full.csv'
        self.df.to_csv(full_path, index=False)
        
        # Analysis-ready dataset (includes original columns + key new features)
        analysis_columns = list(self.df_original.columns)
        
        # Add key engineered features
        key_new_features = []
        for feature_list in self.feature_engineering_results.values():
            key_new_features.extend(feature_list[:5])  # Limit to avoid too many features
        
        analysis_columns.extend([col for col in key_new_features if col in self.df.columns])
        df_analysis = self.df[analysis_columns]
        
        analysis_path = INTEGRATED_DIR / 'covid19_jakarta_mei2020_integrated_analysis.csv'
        df_analysis.to_csv(analysis_path, index=False)
        
        # ML-ready dataset (numeric only, properly scaled)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        df_ml = self.df[numeric_cols].copy()
        
        # Remove ID columns and keys
        id_cols = [col for col in df_ml.columns if 'id' in col.lower() or col in KEY_COLUMNS]
        df_ml = df_ml.drop(columns=[col for col in id_cols if col in df_ml.columns])
        
        ml_path = INTEGRATED_DIR / 'covid19_jakarta_mei2020_integrated_ml_ready.csv'
        df_ml.to_csv(ml_path, index=False)
        
        self.logger.info(f"Saved full integrated data to: {full_path}")
        self.logger.info(f"Saved analysis data to: {analysis_path}")
        self.logger.info(f"Saved ML-ready data to: {ml_path}")
        
        return full_path, analysis_path, ml_path

def integrate_covid_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, DataIntegrator]:
    """
    Convenience function to integrate COVID-19 data
    
    Args:
        df: Input cleaned DataFrame
        
    Returns:
        Tuple of (integrated_df, integrator_instance)
    """
    integrator = DataIntegrator(df)
    
    # Perform integration
    integrated_df = integrator.integrate_data()
    
    # Display report
    report = integrator.get_integration_report()
    print("=== DATA INTEGRATION REPORT ===")
    print(f"Original shape: {report['original_shape']}")
    print(f"Integrated shape: {report['integrated_shape']}")
    print(f"Features added: {report['features_added']}")
    
    print(f"\n=== NEW FEATURES ===")
    for feature in report['new_features'][:10]:  # Show first 10
        print(f"  • {feature}")
    
    print(f"\n=== CORRELATION ANALYSIS ===")
    corr_info = report['correlation_analysis']
    print(f"High correlation pairs: {len(corr_info.get('high_corr_pairs', []))}")
    print(f"Moderate correlation pairs: {len(corr_info.get('moderate_corr_pairs', []))}")
    
    print(f"\n=== INTEGRATION LOG ===")
    for i, log_entry in enumerate(report['integration_log'], 1):
        print(f"{i}. {log_entry}")
    
    return integrated_df, integrator

if __name__ == "__main__":
    # Test the data integrator
    from data.data_loader import load_covid_data
    from preprocessing.data_cleaner import clean_covid_data
    
    print("Testing Data Integrator...")
    df, loader = load_covid_data()
    cleaned_df, cleaner = clean_covid_data(df)
    
    integrated_df, integrator = integrate_covid_data(cleaned_df)
    
    # Save integrated data
    full_path, analysis_path, ml_path = integrator.save_integrated_data()
    print(f"\nIntegrated data saved to:")
    print(f"Full: {full_path}")
    print(f"Analysis: {analysis_path}")
    print(f"ML-ready: {ml_path}")