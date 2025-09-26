# Data Reduction Module for COVID-19 Jakarta Dataset
import sys
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import warnings

# Handle imports with try-except for better error handling
try:
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.manifold import TSNE
    from sklearn.feature_selection import (
        SelectKBest, f_classif, f_regression, chi2,
        RFE, SelectFromModel, VarianceThreshold
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LassoCV, RidgeCV
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
    from sklearn.metrics import explained_variance_score
except ImportError as e:
    print(f"Error: Required dependencies not found: {e}")
    print("Please install with: pip install pandas numpy scikit-learn")

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from config import (
    REDUCTED_DIR, REDUCTION_CONFIG, TARGET_VARIABLES,
    setup_environment, RANDOM_SEED
)

class DataReducer:
    """
    Data Reduction class for COVID-19 Jakarta dataset.
    Handles PCA, t-SNE, feature selection, clustering, sampling, and discretization.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataReducer
        
        Args:
            df: Input integrated DataFrame
        """
        self.df_original = df.copy()
        self.df = df.copy()
        self.reduction_log = []
        
        # Results storage
        self.pca_results = {}
        self.tsne_results = {}
        self.feature_selection_results = {}
        self.clustering_results = {}
        self.sampling_results = {}
        self.discretization_results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup environment
        setup_environment()
        
        # Prepare data for reduction
        self._prepare_data_for_reduction()
        
    def _prepare_data_for_reduction(self):
        """Prepare data for reduction techniques"""
        # Get numeric columns excluding IDs and keys
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['id_kel', 'year', 'month'] + [col for col in numeric_cols if 'id' in col.lower()]
        
        self.feature_names = [col for col in numeric_cols if col not in exclude_cols]
        self.target_variables = [var for var in TARGET_VARIABLES if var in self.df.columns]
        
        # Create feature matrix
        self.X = self.df[self.feature_names].fillna(0)
        
        # Scale features for PCA/t-SNE
        self.scaler_reduction = StandardScaler()
        self.X_scaled = self.scaler_reduction.fit_transform(self.X)
        
        self.logger.info(f"Prepared {len(self.feature_names)} features for reduction")
        self.logger.info(f"Target variables: {self.target_variables}")
    
    def apply_pca(self) -> Dict[str, Any]:
        """Apply Principal Component Analysis"""
        self.logger.info("Applying PCA analysis...")
        
        variance_thresholds = REDUCTION_CONFIG['pca']['variance_thresholds']
        pca_results = {}
        
        # Full PCA for variance analysis
        pca_full = PCA(random_state=RANDOM_SEED)
        X_pca_full = pca_full.fit_transform(self.X_scaled)
        
        explained_var_ratio = pca_full.explained_variance_ratio_
        cumulative_var_ratio = np.cumsum(explained_var_ratio)
        
        # Store full PCA results
        pca_results['full_analysis'] = {
            'explained_variance_ratio': explained_var_ratio,
            'cumulative_variance_ratio': cumulative_var_ratio,
            'eigenvalues': pca_full.explained_variance_
        }
        
        # Apply PCA for different variance thresholds
        for threshold in variance_thresholds:
            n_components = np.argmax(cumulative_var_ratio >= threshold) + 1
            
            pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
            X_reduced = pca.fit_transform(self.X_scaled)
            
            # Calculate reconstruction error
            X_reconstructed = pca.inverse_transform(X_reduced)
            reconstruction_error = np.mean((self.X_scaled - X_reconstructed) ** 2)
            
            pca_key = f'PCA_{int(threshold*100)}'
            pca_results[pca_key] = {
                'n_components': n_components,
                'variance_explained': pca.explained_variance_ratio_.sum(),
                'reduction_ratio': n_components / len(self.feature_names),
                'reconstruction_error': reconstruction_error,
                'data_reduced': X_reduced,
                'pca_object': pca
            }
            
            self.logger.info(f"PCA {threshold:.0%}: {n_components} components explain {pca.explained_variance_ratio_.sum():.1%} variance")
        
        self.pca_results = pca_results
        return pca_results
    
    def apply_tsne(self) -> Dict[str, Any]:
        """Apply t-SNE for nonlinear dimensionality reduction"""
        self.logger.info("Applying t-SNE analysis...")
        
        tsne_configs = [
            {'n_components': 2, 'perplexity': 30, 'key': 'tSNE_2D_p30'},
            {'n_components': 2, 'perplexity': 50, 'key': 'tSNE_2D_p50'},
            {'n_components': 3, 'perplexity': 30, 'key': 'tSNE_3D_p30'}
        ]
        
        tsne_results = {}
        
        for config in tsne_configs:
            self.logger.info(f"Computing {config['key']}...")
            
            tsne = TSNE(
                n_components=config['n_components'],
                perplexity=config['perplexity'],
                random_state=RANDOM_SEED,
                max_iter=1000
            )
            
            X_tsne = tsne.fit_transform(self.X_scaled)
            
            tsne_results[config['key']] = {
                'n_components': config['n_components'],
                'perplexity': config['perplexity'],
                'data_reduced': X_tsne,
                'kl_divergence': tsne.kl_divergence_
            }
        
        self.tsne_results = tsne_results
        return tsne_results
    
    def apply_feature_selection(self) -> Dict[str, Any]:
        """Apply various feature selection techniques"""
        self.logger.info("Applying feature selection techniques...")
        
        k_best = REDUCTION_CONFIG['feature_selection']['k_best']
        rfe_features = REDUCTION_CONFIG['feature_selection']['rfe_features']
        
        feature_results = {}
        
        # For each target variable
        for target in self.target_variables:
            if target not in self.df.columns:
                continue
                
            y = self.df[target].fillna(0)
            target_results = {}
            
            # 1. Univariate Selection
            univariate_results = {}
            
            # SelectKBest with f_regression
            selector_k = SelectKBest(f_regression, k=k_best)
            X_selected = selector_k.fit_transform(self.X, y)
            selected_features = [self.feature_names[i] for i in selector_k.get_support(indices=True)]
            
            univariate_results['Univariate_k10'] = {
                'selected_features': selected_features,
                'scores': selector_k.scores_[selector_k.get_support()],
                'data_reduced': X_selected
            }
            
            target_results['univariate'] = univariate_results
            
            # 2. Recursive Feature Elimination (RFE)
            rfe_results = {}
            
            rf_estimator = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
            rfe = RFE(estimator=rf_estimator, n_features_to_select=rfe_features)
            X_rfe = rfe.fit_transform(self.X, y)
            rfe_features_selected = [self.feature_names[i] for i in rfe.get_support(indices=True)]
            
            rfe_results['RFE_n10'] = {
                'selected_features': rfe_features_selected,
                'ranking': rfe.ranking_,
                'data_reduced': X_rfe
            }
            
            target_results['rfe'] = rfe_results
            
            # 3. Model-based Selection
            model_results = {}
            
            # Lasso-based selection
            lasso_cv = LassoCV(alphas=REDUCTION_CONFIG['feature_selection']['lasso_alpha'], random_state=RANDOM_SEED)
            lasso_selector = SelectFromModel(lasso_cv)
            X_lasso = lasso_selector.fit_transform(self.X, y)
            lasso_features = [self.feature_names[i] for i in lasso_selector.get_support(indices=True)]
            
            model_results['Lasso_Selection'] = {
                'selected_features': lasso_features,
                'data_reduced': X_lasso,
                'alpha_used': getattr(lasso_cv, 'alpha_', 'N/A')
            }
            
            # Random Forest importance
            rf_importance = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
            rf_selector = SelectFromModel(rf_importance)
            X_rf = rf_selector.fit_transform(self.X, y)
            rf_features = [self.feature_names[i] for i in rf_selector.get_support(indices=True)]
            
            model_results['RF_Importance'] = {
                'selected_features': rf_features,
                'data_reduced': X_rf,
                'feature_importance': rf_importance.fit(self.X, y).feature_importances_[rf_selector.get_support()]
            }
            
            target_results['model_based'] = model_results
            feature_results[target] = target_results
        
        # Cross-target consensus analysis
        consensus_results = self._analyze_feature_consensus(feature_results)
        feature_results['consensus'] = consensus_results
        
        self.feature_selection_results = feature_results
        return feature_results
    
    def _analyze_feature_consensus(self, feature_results: Dict) -> Dict[str, Any]:
        """Analyze consensus across different feature selection methods"""
        all_selected_features = []
        
        # Collect all selected features across targets and methods
        for target, methods in feature_results.items():
            if target == 'consensus':
                continue
                
            for method_type, method_results in methods.items():
                for method_name, results in method_results.items():
                    all_selected_features.extend(results['selected_features'])
        
        # Count frequency of each feature
        from collections import Counter
        feature_frequency = Counter(all_selected_features)
        
        # Find universal features (selected by multiple methods)
        universal_features = [feature for feature, count in feature_frequency.items() if count >= 3]
        
        # Cross-target consensus
        cross_target_consensus = {}
        for target in self.target_variables:
            if target in feature_results:
                target_features = []
                for method_type, method_results in feature_results[target].items():
                    for method_name, results in method_results.items():
                        target_features.extend(results['selected_features'])
                
                for feature in set(target_features):
                    if feature not in cross_target_consensus:
                        cross_target_consensus[feature] = 0
                    cross_target_consensus[feature] += 1
        
        return {
            'feature_frequency': feature_frequency,
            'universal_features': universal_features,
            'cross_target_consensus': cross_target_consensus
        }
    
    def apply_clustering_reduction(self) -> Dict[str, Any]:
        """Apply clustering for data reduction"""
        self.logger.info("Applying clustering-based reduction...")
        
        k_values = REDUCTION_CONFIG['clustering']['k_values']
        clustering_results = {}
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X_scaled)
            
            # Find representative samples (closest to centroids)
            cluster_centers = kmeans.cluster_centers_
            representatives = []
            
            for i in range(k):
                cluster_mask = cluster_labels == i
                cluster_data = self.X_scaled[cluster_mask]
                
                if len(cluster_data) > 0:
                    # Find sample closest to centroid
                    distances = np.linalg.norm(cluster_data - cluster_centers[i], axis=1)
                    closest_idx = np.argmin(distances)
                    
                    # Get original index
                    original_indices = np.where(cluster_mask)[0]
                    representative_idx = original_indices[closest_idx]
                    representatives.append(representative_idx)
            
            # Create reduced dataset with representatives
            representative_data = self.df.iloc[representatives].copy()
            
            clustering_results[f'KMeans_{k}'] = {
                'n_clusters': k,
                'cluster_labels': cluster_labels,
                'cluster_centers': cluster_centers,
                'representatives': representatives,
                'representative_data': representative_data,
                'inertia': kmeans.inertia_,
                'reduction_ratio': k / len(self.df)
            }
            
            self.logger.info(f"K-means k={k}: {len(representatives)} representatives, inertia={kmeans.inertia_:.2f}")
        
        self.clustering_results = clustering_results
        return clustering_results
    
    def apply_sampling_techniques(self) -> Dict[str, Any]:
        """Apply various sampling techniques for data reduction"""
        self.logger.info("Applying sampling techniques...")
        
        ratios = REDUCTION_CONFIG['sampling']['ratios']
        strategies = REDUCTION_CONFIG['sampling']['strategies']
        
        sampling_results = {}
        
        for ratio in ratios:
            for strategy in strategies:
                sample_size = int(len(self.df) * ratio)
                
                if strategy == 'random':
                    sampled_indices = np.random.choice(len(self.df), sample_size, replace=False)
                    sampled_data = self.df.iloc[sampled_indices].copy()
                    
                elif strategy == 'stratified':
                    # Stratify by target variable if available
                    if self.target_variables:
                        target_col = self.target_variables[0]  # Use first target
                        
                        # Create strata based on quantiles
                        target_values = self.df[target_col].fillna(0)
                        quantiles = pd.qcut(target_values, q=5, duplicates='drop')
                        
                        sampled_indices = []
                        for stratum in quantiles.cat.categories:
                            stratum_data = self.df[quantiles == stratum]
                            stratum_size = max(1, int(len(stratum_data) * ratio))
                            stratum_sample = stratum_data.sample(n=min(stratum_size, len(stratum_data)), random_state=RANDOM_SEED)
                            sampled_indices.extend(stratum_sample.index.tolist())
                        
                        sampled_data = self.df.loc[sampled_indices].copy()
                    else:
                        # Fallback to random sampling
                        sampled_indices = np.random.choice(len(self.df), sample_size, replace=False)
                        sampled_data = self.df.iloc[sampled_indices].copy()
                
                key = f'{strategy.title()}_{int(ratio*100)}pct'
                sampling_results[key] = {
                    'strategy': strategy,
                    'target_ratio': ratio,
                    'actual_ratio': len(sampled_data) / len(self.df),
                    'sampled_size': len(sampled_data),
                    'sampled_data': sampled_data,
                    'sampled_indices': sampled_indices if strategy == 'random' else list(sampled_data.index)
                }
                
                self.logger.info(f"{key}: {len(sampled_data)} samples ({len(sampled_data)/len(self.df):.1%})")
        
        self.sampling_results = sampling_results
        return sampling_results
    
    def apply_discretization(self) -> Dict[str, Any]:
        """Apply discretization techniques for data compression"""
        self.logger.info("Applying discretization techniques...")
        
        n_bins_list = REDUCTION_CONFIG['discretization']['n_bins']
        strategies = REDUCTION_CONFIG['discretization']['strategies']
        
        discretization_results = {}
        
        # Select continuous variables for discretization
        continuous_vars = []
        for col in self.feature_names:
            if self.df[col].nunique() > 20:  # Consider as continuous if >20 unique values
                continuous_vars.append(col)
        
        if not continuous_vars:
            self.logger.warning("No continuous variables found for discretization")
            return {}
        
        for n_bins in n_bins_list:
            for strategy in strategies:
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
                
                continuous_data = self.df[continuous_vars].fillna(0)
                discretized_data = discretizer.fit_transform(continuous_data)
                
                # Calculate compression ratio
                original_unique_values = sum(self.df[col].nunique() for col in continuous_vars)
                discretized_unique_values = sum(np.unique(discretized_data[:, i]).shape[0] for i in range(len(continuous_vars)))
                compression_ratio = discretized_unique_values / original_unique_values
                
                # Create discretized dataframe
                discretized_df = self.df.copy()
                for i, col in enumerate(continuous_vars):
                    discretized_df[col] = discretized_data[:, i]
                
                key = f'{strategy}_{n_bins}bins'
                discretization_results[key] = {
                    'n_bins': n_bins,
                    'strategy': strategy,
                    'continuous_vars': continuous_vars,
                    'discretized_data': discretized_df,
                    'compression_ratio': compression_ratio,
                    'discretizer': discretizer
                }
                
                self.logger.info(f"{key}: {compression_ratio:.3f} compression ratio")
        
        self.discretization_results = discretization_results
        return discretization_results
    
    def reduce_data(self, 
                   apply_pca: bool = True,
                   apply_tsne: bool = True,
                   apply_feature_selection: bool = True,
                   apply_clustering: bool = True,
                   apply_sampling: bool = True,
                   apply_discretization: bool = True) -> Dict[str, Any]:
        """
        Apply comprehensive data reduction
        
        Args:
            apply_pca: Whether to apply PCA
            apply_tsne: Whether to apply t-SNE
            apply_feature_selection: Whether to apply feature selection
            apply_clustering: Whether to apply clustering
            apply_sampling: Whether to apply sampling
            apply_discretization: Whether to apply discretization
            
        Returns:
            Dict containing all reduction results
        """
        self.logger.info("Starting comprehensive data reduction...")
        
        reduction_results = {}
        
        # 1. PCA
        if apply_pca:
            reduction_results['pca'] = self.apply_pca()
        
        # 2. t-SNE
        if apply_tsne:
            reduction_results['tsne'] = self.apply_tsne()
        
        # 3. Feature Selection
        if apply_feature_selection:
            reduction_results['feature_selection'] = self.apply_feature_selection()
        
        # 4. Clustering
        if apply_clustering:
            reduction_results['clustering'] = self.apply_clustering_reduction()
        
        # 5. Sampling
        if apply_sampling:
            reduction_results['sampling'] = self.apply_sampling_techniques()
        
        # 6. Discretization
        if apply_discretization:
            reduction_results['discretization'] = self.apply_discretization()
        
        self.logger.info("Data reduction completed!")
        return reduction_results
    
    def generate_reduced_datasets(self) -> Dict[str, Path]:
        """
        Generate and save various reduced datasets
        
        Returns:
            Dict mapping dataset names to file paths
        """
        self.logger.info("Generating reduced datasets...")
        
        REDUCTED_DIR.mkdir(parents=True, exist_ok=True)
        saved_files = {}
        
        # 1. PCA-based datasets
        if self.pca_results:
            for pca_key, pca_data in self.pca_results.items():
                if 'data_reduced' in pca_data:
                    # Create dataframe with PCA components
                    pca_df = pd.DataFrame(
                        pca_data['data_reduced'],
                        columns=[f'PC{i+1}' for i in range(pca_data['data_reduced'].shape[1])]
                    )
                    
                    # Add target variables
                    for target in self.target_variables:
                        if target in self.df.columns:
                            pca_df[target] = self.df[target].values
                    
                    filename = f'covid19_jakarta_reduced_{pca_key.lower()}.csv'
                    filepath = REDUCTED_DIR / filename
                    pca_df.to_csv(filepath, index=False)
                    saved_files[pca_key] = filepath
        
        # 2. Feature selection datasets
        if self.feature_selection_results:
            # Universal features dataset
            consensus = self.feature_selection_results.get('consensus', {})
            universal_features = consensus.get('universal_features', [])
            
            if universal_features:
                universal_df = self.df[universal_features + self.target_variables].copy()
                filepath = REDUCTED_DIR / 'covid19_jakarta_reduced_universal_features.csv'
                universal_df.to_csv(filepath, index=False)
                saved_files['universal_features'] = filepath
            
            # RFE-based datasets for each target
            for target in self.target_variables:
                if target in self.feature_selection_results:
                    rfe_results = self.feature_selection_results[target].get('rfe', {})
                    if 'RFE_n10' in rfe_results:
                        selected_features = rfe_results['RFE_n10']['selected_features']
                        rfe_df = self.df[selected_features + [target]].copy()
                        
                        filename = f'covid19_jakarta_reduced_rfe_{target}.csv'
                        filepath = REDUCTED_DIR / filename
                        rfe_df.to_csv(filepath, index=False)
                        saved_files[f'rfe_{target}'] = filepath
        
        # 3. Clustering-based datasets
        if self.clustering_results:
            best_clustering = min(self.clustering_results.keys(), 
                                key=lambda k: self.clustering_results[k]['inertia'])
            
            cluster_data = self.clustering_results[best_clustering]['representative_data']
            filepath = REDUCTED_DIR / 'covid19_jakarta_reduced_clustering.csv'
            cluster_data.to_csv(filepath, index=False)
            saved_files['clustering'] = filepath
        
        # 4. Sampling-based datasets
        if self.sampling_results:
            for sampling_key, sampling_data in self.sampling_results.items():
                sampled_df = sampling_data['sampled_data']
                filename = f'covid19_jakarta_reduced_{sampling_key.lower()}.csv'
                filepath = REDUCTED_DIR / filename
                sampled_df.to_csv(filepath, index=False)
                saved_files[sampling_key] = filepath
        
        # 5. Discretization datasets
        if self.discretization_results:
            best_discretization = min(self.discretization_results.keys(),
                                    key=lambda k: self.discretization_results[k]['compression_ratio'])
            
            discretized_df = self.discretization_results[best_discretization]['discretized_data']
            filepath = REDUCTED_DIR / 'covid19_jakarta_reduced_discretized.csv'
            discretized_df.to_csv(filepath, index=False)
            saved_files['discretized'] = filepath
        
        # 6. Hybrid optimized dataset (combination of best techniques)
        hybrid_df = self._create_hybrid_dataset()
        if hybrid_df is not None:
            filepath = REDUCTED_DIR / 'covid19_jakarta_reduced_hybrid_optimized.csv'
            hybrid_df.to_csv(filepath, index=False)
            saved_files['hybrid_optimized'] = filepath
        
        self.logger.info(f"Generated {len(saved_files)} reduced datasets")
        return saved_files
    
    def _create_hybrid_dataset(self) -> Optional[pd.DataFrame]:
        """Create a hybrid dataset combining best reduction techniques"""
        try:
            # Start with universal features if available
            consensus = self.feature_selection_results.get('consensus', {})
            universal_features = consensus.get('universal_features', [])
            
            if not universal_features:
                return None
            
            # Base dataset with universal features
            hybrid_df = self.df[universal_features + self.target_variables].copy()
            
            # Add best PCA components if available
            if self.pca_results and 'PCA_90' in self.pca_results:
                pca_data = self.pca_results['PCA_90']['data_reduced']
                n_pca = min(5, pca_data.shape[1])  # Add top 5 PCA components
                
                for i in range(n_pca):
                    hybrid_df[f'PC{i+1}'] = pca_data[:, i]
            
            return hybrid_df
            
        except Exception as e:
            self.logger.warning(f"Could not create hybrid dataset: {e}")
            return None
    
    def get_reduction_report(self) -> Dict[str, Any]:
        """Get comprehensive reduction report"""
        return {
            'original_shape': self.df_original.shape,
            'features_analyzed': len(self.feature_names),
            'target_variables': self.target_variables,
            'pca_summary': {
                'variance_95_components': self.pca_results.get('PCA_95', {}).get('n_components', 0),
                'variance_90_components': self.pca_results.get('PCA_90', {}).get('n_components', 0),
            },
            'feature_selection_summary': {
                'universal_features': len(self.feature_selection_results.get('consensus', {}).get('universal_features', [])),
                'targets_analyzed': len([k for k in self.feature_selection_results.keys() if k != 'consensus'])
            },
            'clustering_summary': {
                'best_k': min(self.clustering_results.keys(), key=lambda k: self.clustering_results[k]['inertia']) if self.clustering_results else None,
                'configurations_tested': len(self.clustering_results)
            },
            'sampling_summary': {
                'techniques_applied': len(self.sampling_results),
                'reduction_ratios': [data['actual_ratio'] for data in self.sampling_results.values()]
            },
            'reduction_log': self.reduction_log
        }

def reduce_covid_data(df: pd.DataFrame) -> Tuple[Dict[str, Any], DataReducer]:
    """
    Convenience function to reduce COVID-19 data
    
    Args:
        df: Input integrated DataFrame
        
    Returns:
        Tuple of (reduction_results, reducer_instance)
    """
    reducer = DataReducer(df)
    
    # Perform reduction
    results = reducer.reduce_data()
    
    # Generate datasets
    saved_files = reducer.generate_reduced_datasets()
    
    # Display report
    report = reducer.get_reduction_report()
    print("=== DATA REDUCTION REPORT ===")
    print(f"Original shape: {report['original_shape']}")
    print(f"Features analyzed: {report['features_analyzed']}")
    print(f"Target variables: {report['target_variables']}")
    
    print(f"\n=== PCA SUMMARY ===")
    pca_summary = report['pca_summary']
    print(f"95% variance: {pca_summary['variance_95_components']} components")
    print(f"90% variance: {pca_summary['variance_90_components']} components")
    
    print(f"\n=== FEATURE SELECTION SUMMARY ===")
    fs_summary = report['feature_selection_summary']
    print(f"Universal features: {fs_summary['universal_features']}")
    print(f"Targets analyzed: {fs_summary['targets_analyzed']}")
    
    print(f"\n=== SAVED DATASETS ===")
    for name, path in saved_files.items():
        print(f"  • {name}: {path.name}")
    
    return results, reducer

if __name__ == "__main__":
    # Test the data reducer
    from data.data_loader import load_covid_data
    from preprocessing.data_cleaner import clean_covid_data
    from preprocessing.data_integrator import integrate_covid_data
    
    print("Testing Data Reducer...")
    df, loader = load_covid_data()
    cleaned_df, cleaner = clean_covid_data(df)
    integrated_df, integrator = integrate_covid_data(cleaned_df)
    
    results, reducer = reduce_covid_data(integrated_df)
    
    print(f"\nData reduction completed successfully!")