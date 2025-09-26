# Main Pipeline Orchestrator for COVID-19 Jakarta Dataset Analysis
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import traceback
import warnings
import importlib.util
import os

# Handle imports with try-except for better error handling
try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: Required dependencies not found: {e}")
    print("Please install with: pip install -r requirements.txt")
    sys.exit(1)

warnings.filterwarnings('ignore')

# Add current directory and src directory to path for imports
current_dir = Path(__file__).parent
src_path = current_dir / 'src'
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(src_path))

# Import modules for dynamic loading

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import config module
config_path = src_path / 'config.py'
config = import_module_from_path('config', config_path)
RAW_DATA_PATH = config.RAW_DATA_PATH
CLEANED_DIR = config.CLEANED_DIR
INTEGRATED_DIR = config.INTEGRATED_DIR
REDUCTED_DIR = config.REDUCTED_DIR
COVID_COLUMNS = config.COVID_COLUMNS
setup_logging = config.setup_logging
setup_matplotlib = config.setup_matplotlib
setup_directories = config.setup_directories

# Import data modules
data_loader_path = src_path / 'data' / 'data_loader.py'
data_loader_module = import_module_from_path('data_loader', data_loader_path)
load_covid_data = data_loader_module.load_covid_data

# Import preprocessing modules
data_cleaner_path = src_path / 'preprocessing' / 'data_cleaner.py'
data_cleaner_module = import_module_from_path('data_cleaner', data_cleaner_path)
clean_covid_data = data_cleaner_module.clean_covid_data

data_integrator_path = src_path / 'preprocessing' / 'data_integrator.py'
data_integrator_module = import_module_from_path('data_integrator', data_integrator_path)
integrate_covid_data = data_integrator_module.integrate_covid_data

data_reducer_path = src_path / 'preprocessing' / 'data_reducer.py'
data_reducer_module = import_module_from_path('data_reducer', data_reducer_path)
reduce_covid_data = data_reducer_module.reduce_covid_data

# Import visualization modules
vis_utils_path = src_path / 'visualization' / 'visualization_utils.py'
vis_utils_module = import_module_from_path('visualization_utils', vis_utils_path)
create_cleaning_visualizations = vis_utils_module.create_cleaning_visualizations
create_integration_visualizations = vis_utils_module.create_integration_visualizations
create_reduction_visualizations = vis_utils_module.create_reduction_visualizations

class COVIDDataPipeline:
    """
    Main pipeline class for COVID-19 Jakarta dataset preprocessing and analysis.
    Orchestrates data loading, cleaning, integration, reduction, and visualization.
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize pipeline with logging configuration"""
        self.verbose = verbose
        self.logger = setup_logging()
        setup_matplotlib()
        setup_directories()
        
        self.logger.info("=" * 80)
        self.logger.info("COVID-19 Jakarta Dataset Analysis Pipeline Initialized")
        self.logger.info("=" * 80)
        
        # Pipeline state tracking
        self.pipeline_state = {
            'data_loaded': False,
            'data_cleaned': False,
            'data_integrated': False,
            'data_reduced': False,
            'visualizations_created': False
        }
        
        # Data storage
        self.data = {}
        self.reports = {}
        self.visualizations = {}
    
    def log_step(self, step: str, status: str = "START"):
        """Log pipeline step with clear formatting"""
        if status == "START":
            self.logger.info(f"\\n{'='*60}")
            self.logger.info(f"STEP: {step}")
            self.logger.info(f"{'='*60}")
        elif status == "SUCCESS":
            self.logger.info(f"✓ COMPLETED: {step}")
        elif status == "ERROR":
            self.logger.error(f"✗ FAILED: {step}")
        elif status == "SKIP":
            self.logger.info(f"⊘ SKIPPED: {step}")
    
    def run_data_loading(self) -> bool:
        """Execute data loading phase"""
        self.log_step("Data Loading Phase")
        
        try:
            # Load raw data
            data_result = load_covid_data()
            
            if isinstance(data_result, tuple):
                self.data['raw'], self.data['loader'] = data_result
            else:
                self.data['raw'] = data_result
            
            if self.data['raw'] is None:
                raise ValueError("Failed to load COVID-19 dataset")
            
            # Log basic information
            shape = self.data['raw'].shape
            self.logger.info(f"Dataset loaded successfully: {shape[0]} rows × {shape[1]} columns")
            self.logger.info(f"COVID-19 specific columns available: {len([col for col in COVID_COLUMNS if col in self.data['raw'].columns])}")
            
            # Store basic information
            self.reports['loading'] = {
                'original_shape': shape,
                'covid_columns_available': [col for col in COVID_COLUMNS if col in self.data['raw'].columns],
                'total_columns': list(self.data['raw'].columns),
                'missing_values_total': self.data['raw'].isnull().sum().sum(),
                'data_types': self.data['raw'].dtypes.to_dict(),
                'loading_timestamp': datetime.now().isoformat()
            }
            
            self.pipeline_state['data_loaded'] = True
            self.log_step("Data Loading Phase", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step("Data Loading Phase", "ERROR")
            self.logger.error(f"Error in data loading: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run_data_cleaning(self) -> bool:
        """Execute data cleaning phase"""
        if not self.pipeline_state['data_loaded']:
            self.log_step("Data Cleaning Phase", "SKIP")
            self.logger.warning("Cannot clean data - data not loaded")
            return False
        
        self.log_step("Data Cleaning Phase")
        
        try:
            # Clean data
            cleaning_result = clean_covid_data(self.data['raw'])
            
            if isinstance(cleaning_result, tuple):
                self.data['cleaned'], cleaner_instance = cleaning_result
            else:
                self.data['cleaned'] = cleaning_result
                cleaner_instance = None
            
            # Get missing analysis and report from cleaner instance
            if cleaner_instance:
                missing_analysis = cleaner_instance.analyze_missing_values()
                cleaning_report = cleaner_instance.get_cleaning_report()
            else:
                # Fallback if no cleaner instance
                missing_analysis = pd.DataFrame()
                cleaning_report = {'original_shape': self.data['raw'].shape, 'cleaned_shape': self.data['cleaned'].shape}
            
            # Store results
            self.reports['cleaning'] = cleaning_report
            self.reports['missing_analysis'] = missing_analysis
            
            # Log results
            original_shape = self.reports['loading']['original_shape']
            cleaned_shape = self.data['cleaned'].shape
            
            self.logger.info(f"Data cleaning completed:")
            self.logger.info(f"  Shape change: {original_shape} → {cleaned_shape}")
            self.logger.info(f"  Rows removed: {original_shape[0] - cleaned_shape[0]}")
            self.logger.info(f"  Columns removed: {original_shape[1] - cleaned_shape[1]}")
            self.logger.info(f"  Missing values handled: {cleaning_report.get('missing_info', {}).get('total_missing', 0)}")
            
            self.pipeline_state['data_cleaned'] = True
            self.log_step("Data Cleaning Phase", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step("Data Cleaning Phase", "ERROR")
            self.logger.error(f"Error in data cleaning: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run_data_integration(self) -> bool:
        """Execute data integration phase"""
        if not self.pipeline_state['data_cleaned']:
            self.log_step("Data Integration Phase", "SKIP")
            self.logger.warning("Cannot integrate data - data not cleaned")
            return False
        
        self.log_step("Data Integration Phase")
        
        try:
            # Integrate data
            integration_result = integrate_covid_data(self.data['cleaned'])
            
            if isinstance(integration_result, tuple):
                self.data['integrated'], integrator_instance = integration_result
            else:
                self.data['integrated'] = integration_result
                integrator_instance = None
            
            # Get correlation, covariance analysis and report from integrator instance
            if integrator_instance:
                correlation_matrix = integrator_instance.correlation_analysis.get('correlation_matrix', pd.DataFrame())
                covariance_analysis = integrator_instance.covariance_analysis
                integration_report = integrator_instance.get_integration_report()
            else:
                # Fallback if no integrator instance
                correlation_matrix = pd.DataFrame()
                covariance_analysis = {}
                integration_report = {'original_shape': self.data['cleaned'].shape, 'integrated_shape': self.data['integrated'].shape}
            
            # Store results
            self.reports['integration'] = integration_report
            self.data['correlation_matrix'] = correlation_matrix
            self.data['covariance_analysis'] = covariance_analysis
            
            # Log results
            cleaned_shape = self.data['cleaned'].shape
            integrated_shape = self.data['integrated'].shape
            
            self.logger.info(f"Data integration completed:")
            self.logger.info(f"  Shape change: {cleaned_shape} → {integrated_shape}")
            self.logger.info(f"  Features added: {integrated_shape[1] - cleaned_shape[1]}")
            
            # Log correlation analysis
            correlation_analysis = integration_report.get('correlation_analysis', {})
            high_corr_pairs = len(correlation_analysis.get('high_corr_pairs', []))
            moderate_corr_pairs = len(correlation_analysis.get('moderate_corr_pairs', []))
            
            self.logger.info(f"  High correlation pairs (>0.9): {high_corr_pairs}")
            self.logger.info(f"  Moderate correlation pairs (0.5-0.9): {moderate_corr_pairs}")
            
            # Log feature engineering
            feature_engineering = integration_report.get('feature_engineering', {})
            for category, features in feature_engineering.items():
                if features:
                    self.logger.info(f"  {category.replace('_', ' ').title()}: {len(features)} features")
            
            self.pipeline_state['data_integrated'] = True
            self.log_step("Data Integration Phase", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step("Data Integration Phase", "ERROR")
            self.logger.error(f"Error in data integration: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run_data_reduction(self) -> bool:
        """Execute data reduction phase"""
        if not self.pipeline_state['data_integrated']:
            self.log_step("Data Reduction Phase", "SKIP")
            self.logger.warning("Cannot reduce data - data not integrated")
            return False
        
        self.log_step("Data Reduction Phase")
        
        try:
            # Define target variables for reduction
            target_variables = [col for col in ['positif', 'sembuh', 'meninggal'] 
                              if col in self.data['integrated'].columns]
            
            if not target_variables:
                self.logger.warning("No target variables found for reduction analysis")
                target_variables = ['positif']  # Default fallback
            
            # Reduce data
            reduction_result = reduce_covid_data(self.data['integrated'])
            
            if not isinstance(reduction_result, tuple) or len(reduction_result) != 2:
                raise ValueError("Invalid reduction result format")
            
            reduction_results, reducer = reduction_result
            
            # Generate datasets and get report from reducer
            reduced_datasets = reducer.generate_reduced_datasets()
            reduction_report = reducer.get_reduction_report()
            
            # Store results
            self.data['reduction_results'] = reduction_results
            self.data['reduced_datasets'] = reduced_datasets
            self.reports['reduction'] = reduction_report
            self.data['target_variables'] = target_variables
            
            # Log results
            integrated_shape = self.data['integrated'].shape
            
            self.logger.info(f"Data reduction completed:")
            self.logger.info(f"  Original shape: {integrated_shape}")
            self.logger.info(f"  Target variables analyzed: {len(target_variables)}")
            
            # Log PCA results
            pca_summary = reduction_report.get('pca_summary', {})
            if pca_summary:
                self.logger.info(f"  PCA 95% variance: {pca_summary.get('variance_95_components', 0)} components")
                self.logger.info(f"  PCA 90% variance: {pca_summary.get('variance_90_components', 0)} components")
            
            # Log feature selection results
            fs_summary = reduction_report.get('feature_selection_summary', {})
            if fs_summary:
                self.logger.info(f"  Feature selection targets: {fs_summary.get('targets_analyzed', 0)}")
                self.logger.info(f"  Universal features found: {fs_summary.get('universal_features', 0)}")
            
            # Log clustering results
            cluster_summary = reduction_report.get('clustering_summary', {})
            if cluster_summary:
                self.logger.info(f"  Clustering configurations tested: {cluster_summary.get('configurations_tested', 0)}")
            
            # Log sampling results
            sampling_summary = reduction_report.get('sampling_summary', {})
            if sampling_summary:
                self.logger.info(f"  Sampling techniques applied: {sampling_summary.get('techniques_applied', 0)}")
            
            # Log datasets created
            self.logger.info(f"  Reduced datasets created: {len(reduced_datasets)}")
            for dataset_name, dataset_path in reduced_datasets.items():
                self.logger.info(f"    {dataset_name}: {dataset_path.name}")
            
            self.pipeline_state['data_reduced'] = True
            self.log_step("Data Reduction Phase", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step("Data Reduction Phase", "ERROR")
            self.logger.error(f"Error in data reduction: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run_visualization_creation(self) -> bool:
        """Execute visualization creation phase"""
        # Check dependencies
        phases_required = ['data_cleaned', 'data_integrated', 'data_reduced']
        phases_ready = [self.pipeline_state.get(phase, False) for phase in phases_required]
        
        if not all(phases_ready):
            missing_phases = [phases_required[i] for i, ready in enumerate(phases_ready) if not ready]
            self.log_step("Visualization Creation Phase", "SKIP")
            self.logger.warning(f"Cannot create visualizations - missing phases: {missing_phases}")
            return False
        
        self.log_step("Visualization Creation Phase")
        
        try:
            total_visualizations = 0
            
            # 1. Create cleaning visualizations
            self.logger.info("Creating data cleaning visualizations...")
            cleaning_viz = create_cleaning_visualizations(
                self.data['cleaned'], 
                self.reports['missing_analysis'], 
                self.reports['cleaning']
            )
            self.visualizations['cleaning'] = cleaning_viz
            total_visualizations += len(cleaning_viz)
            self.logger.info(f"  Created {len(cleaning_viz)} cleaning visualizations")
            
            # 2. Create integration visualizations
            self.logger.info("Creating data integration visualizations...")
            integration_viz = create_integration_visualizations(
                self.data['correlation_matrix'],
                self.data['covariance_analysis'],
                self.reports['integration']
            )
            self.visualizations['integration'] = integration_viz
            total_visualizations += len(integration_viz)
            self.logger.info(f"  Created {len(integration_viz)} integration visualizations")
            
            # 3. Create reduction visualizations
            self.logger.info("Creating data reduction visualizations...")
            reduction_viz = create_reduction_visualizations(
                self.data['reduction_results'],
                self.reports['reduction'],
                self.data['integrated'],
                self.data['target_variables']
            )
            self.visualizations['reduction'] = reduction_viz
            total_visualizations += len(reduction_viz)
            self.logger.info(f"  Created {len(reduction_viz)} reduction visualizations")
            
            # Log summary
            self.logger.info(f"Visualization creation completed:")
            self.logger.info(f"  Total independent visualizations: {total_visualizations}")
            self.logger.info(f"  Cleaning phase: {len(self.visualizations['cleaning'])}")
            self.logger.info(f"  Integration phase: {len(self.visualizations['integration'])}")
            self.logger.info(f"  Reduction phase: {len(self.visualizations['reduction'])}")
            
            self.pipeline_state['visualizations_created'] = True
            self.log_step("Visualization Creation Phase", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step("Visualization Creation Phase", "ERROR")
            self.logger.error(f"Error in visualization creation: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def generate_pipeline_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline execution report"""
        report = {
            'pipeline_execution': {
                'timestamp': datetime.now().isoformat(),
                'pipeline_state': self.pipeline_state.copy(),
                'success_rate': sum(self.pipeline_state.values()) / len(self.pipeline_state) * 100
            },
            'data_summary': {},
            'processing_summary': {},
            'visualization_summary': {},
            'recommendations': []
        }
        
        # Data summary
        if self.pipeline_state['data_loaded']:
            report['data_summary'] = {
                'original_shape': self.reports['loading']['original_shape'],
                'covid_columns_found': len(self.reports['loading']['covid_columns_available']),
                'total_missing_values': self.reports['loading']['missing_values_total']
            }
            
            if self.pipeline_state['data_integrated']:
                integrated_shape = self.data['integrated'].shape
                report['data_summary']['final_shape'] = integrated_shape
                report['data_summary']['dimension_change'] = {
                    'rows': integrated_shape[0] - self.reports['loading']['original_shape'][0],
                    'columns': integrated_shape[1] - self.reports['loading']['original_shape'][1]
                }
        
        # Processing summary
        if self.reports:
            report['processing_summary'] = {
                'cleaning_applied': 'cleaning' in self.reports,
                'integration_applied': 'integration' in self.reports,
                'reduction_applied': 'reduction' in self.reports
            }
            
            if 'cleaning' in self.reports:
                cleaning_report = self.reports['cleaning']
                report['processing_summary']['cleaning_results'] = {
                    'rows_removed': cleaning_report.get('rows_removed', 0),
                    'columns_removed': cleaning_report.get('columns_removed', 0),
                    'missing_values_handled': cleaning_report.get('missing_info', {}).get('total_missing', 0)
                }
            
            if 'reduction' in self.reports:
                reduction_report = self.reports['reduction']
                report['processing_summary']['reduction_results'] = {
                    'techniques_applied': [],
                    'datasets_created': len(self.data.get('reduced_datasets', {}))
                }
                
                # Check which techniques were applied
                if reduction_report.get('pca_summary', {}).get('variance_95_components', 0) > 0:
                    report['processing_summary']['reduction_results']['techniques_applied'].append('PCA')
                
                if reduction_report.get('feature_selection_summary', {}).get('targets_analyzed', 0) > 0:
                    report['processing_summary']['reduction_results']['techniques_applied'].append('Feature Selection')
                
                if reduction_report.get('clustering_summary', {}).get('configurations_tested', 0) > 0:
                    report['processing_summary']['reduction_results']['techniques_applied'].append('Clustering')
                
                if reduction_report.get('sampling_summary', {}).get('techniques_applied', 0) > 0:
                    report['processing_summary']['reduction_results']['techniques_applied'].append('Sampling')
        
        # Visualization summary
        if self.pipeline_state['visualizations_created']:
            report['visualization_summary'] = {
                'total_visualizations': sum(len(viz) for viz in self.visualizations.values()),
                'by_phase': {
                    'cleaning': len(self.visualizations.get('cleaning', [])),
                    'integration': len(self.visualizations.get('integration', [])),
                    'reduction': len(self.visualizations.get('reduction', []))
                },
                'independent_plots_created': True
            }
        
        # Generate recommendations
        recommendations = []
        
        # Success rate recommendations
        success_rate = report['pipeline_execution']['success_rate']
        if success_rate == 100:
            recommendations.append("✓ Complete pipeline executed successfully")
        elif success_rate >= 80:
            recommendations.append("⚠ Most pipeline phases completed - review failed phases")
        else:
            recommendations.append("✗ Multiple pipeline failures - check error logs")
        
        # Data quality recommendations
        if 'data_summary' in report:
            missing_ratio = report['data_summary'].get('total_missing_values', 0) / np.prod(report['data_summary'].get('original_shape', [1, 1]))
            if missing_ratio > 0.1:
                recommendations.append(f"⚠ High missing data ratio ({missing_ratio:.1%}) - consider additional imputation")
            elif missing_ratio > 0:
                recommendations.append(f"✓ Manageable missing data ({missing_ratio:.1%}) - handled by pipeline")
            else:
                recommendations.append("✓ Complete dataset - no missing values")
        
        # Processing recommendations
        if report.get('processing_summary', {}).get('reduction_applied', False):
            techniques = report['processing_summary']['reduction_results']['techniques_applied']
            if len(techniques) >= 3:
                recommendations.append(f"✓ Comprehensive reduction applied ({len(techniques)} techniques)")
            elif techniques:
                recommendations.append(f"⚠ Partial reduction applied - consider additional techniques")
        
        # Visualization recommendations
        if report.get('visualization_summary', {}).get('total_visualizations', 0) > 0:
            total_viz = report['visualization_summary']['total_visualizations']
            recommendations.append(f"✓ Independent visualizations created ({total_viz} plots)")
        
        report['recommendations'] = recommendations
        return report
    
    def run_full_pipeline(self, 
                         include_visualization: bool = True,
                         save_report: bool = True) -> bool:
        """Execute complete pipeline from start to finish"""
        start_time = datetime.now()
        
        self.logger.info("\\n" + "="*80)
        self.logger.info("STARTING COMPLETE COVID-19 JAKARTA ANALYSIS PIPELINE")
        self.logger.info("="*80)
        
        # Execute pipeline phases
        phases = [
            ("Loading", self.run_data_loading),
            ("Cleaning", self.run_data_cleaning),
            ("Integration", self.run_data_integration),
            ("Reduction", self.run_data_reduction),
        ]
        
        if include_visualization:
            phases.append(("Visualization", self.run_visualization_creation))
        
        # Track success for each phase
        phase_results = {}
        
        for phase_name, phase_function in phases:
            success = phase_function()
            phase_results[phase_name] = success
            
            if not success:
                self.logger.error(f"Pipeline stopped due to {phase_name} phase failure")
                break
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        # Generate and display final report
        pipeline_report = self.generate_pipeline_report()
        
        self.logger.info("\\n" + "="*80)
        self.logger.info("PIPELINE EXECUTION COMPLETE")
        self.logger.info("="*80)
        
        # Log execution summary
        success_count = sum(phase_results.values())
        total_phases = len(phase_results)
        
        self.logger.info(f"Execution time: {execution_time}")
        self.logger.info(f"Phases completed: {success_count}/{total_phases}")
        self.logger.info(f"Success rate: {success_count/total_phases*100:.1f}%")
        
        # Log phase results
        for phase_name, success in phase_results.items():
            status = "✓" if success else "✗"
            self.logger.info(f"  {status} {phase_name}")
        
        # Log recommendations
        self.logger.info("\\nRecommendations:")
        for rec in pipeline_report['recommendations']:
            self.logger.info(f"  {rec}")
        
        # Log data transformation summary
        if 'data_summary' in pipeline_report:
            data_summary = pipeline_report['data_summary']
            if 'final_shape' in data_summary:
                original_shape = data_summary['original_shape']
                final_shape = data_summary['final_shape']
                self.logger.info(f"\\nData transformation: {original_shape} → {final_shape}")
        
        # Log visualization summary
        if 'visualization_summary' in pipeline_report:
            viz_summary = pipeline_report['visualization_summary']
            total_viz = viz_summary.get('total_visualizations', 0)
            if total_viz > 0:
                self.logger.info(f"Independent visualizations created: {total_viz}")
                
                by_phase = viz_summary.get('by_phase', {})
                for phase, count in by_phase.items():
                    if count > 0:
                        self.logger.info(f"  {phase.title()}: {count} plots")
        
        # Save report if requested
        if save_report:
            try:
                report_path = Path("pipeline_execution_report.json")
                import json
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(pipeline_report, f, indent=2, ensure_ascii=False, default=str)
                self.logger.info(f"\\nPipeline report saved: {report_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save pipeline report: {e}")
        
        self.logger.info("="*80)
        
        # Return overall success
        return success_count == total_phases
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'state': self.pipeline_state.copy(),
            'data_shapes': {
                phase: self.data[phase].shape if phase in self.data and hasattr(self.data[phase], 'shape') else None
                for phase in ['raw', 'cleaned', 'integrated']
            },
            'reports_available': list(self.reports.keys()),
            'visualizations_count': {
                phase: len(visualizations) for phase, visualizations in self.visualizations.items()
            } if self.visualizations else {}
        }

# Convenience functions for direct execution
def run_complete_pipeline(include_visualization: bool = True, 
                         verbose: bool = True, 
                         save_report: bool = True) -> bool:
    """Run the complete COVID-19 data analysis pipeline"""
    pipeline = COVIDDataPipeline(verbose=verbose)
    return pipeline.run_full_pipeline(
        include_visualization=include_visualization,
        save_report=save_report
    )

def run_pipeline_phase(phase: str, data: pd.DataFrame = None) -> Any:
    """Run a specific pipeline phase independently"""
    pipeline = COVIDDataPipeline(verbose=True)
    
    if phase.lower() == 'loading':
        return pipeline.run_data_loading()
    elif phase.lower() == 'cleaning':
        if data is not None:
            pipeline.data['raw'] = data
            pipeline.pipeline_state['data_loaded'] = True
        return pipeline.run_data_cleaning()
    elif phase.lower() == 'integration':
        if data is not None:
            pipeline.data['cleaned'] = data
            pipeline.pipeline_state['data_cleaned'] = True
        return pipeline.run_data_integration()
    elif phase.lower() == 'reduction':
        if data is not None:
            pipeline.data['integrated'] = data
            pipeline.pipeline_state['data_integrated'] = True
        return pipeline.run_data_reduction()
    elif phase.lower() == 'visualization':
        return pipeline.run_visualization_creation()
    else:
        raise ValueError(f"Unknown phase: {phase}. Use 'loading', 'cleaning', 'integration', 'reduction', or 'visualization'")

if __name__ == "__main__":
    print("="*80)
    print("COVID-19 Jakarta Dataset Analysis Pipeline")
    print("="*80)
    print()
    
    # Check if user wants to run specific phase or complete pipeline
    import sys
    
    if len(sys.argv) > 1:
        phase = sys.argv[1].lower()
        if phase in ['loading', 'cleaning', 'integration', 'reduction', 'visualization']:
            print(f"Running {phase} phase only...")
            success = run_pipeline_phase(phase)
            print(f"Phase {phase}: {'SUCCESS' if success else 'FAILED'}")
        else:
            print(f"Unknown phase: {phase}")
            print("Available phases: loading, cleaning, integration, reduction, visualization")
    else:
        # Run complete pipeline
        print("Running complete pipeline...")
        print("This will execute all phases: loading → cleaning → integration → reduction → visualization")
        print()
        
        success = run_complete_pipeline(
            include_visualization=True,
            verbose=True,
            save_report=True
        )
        
        print()
        print("="*80)
        if success:
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("✓ All data processing phases executed")
            print("✓ Independent visualizations created")
            print("✓ Modular structure ready for use")
        else:
            print("❌ PIPELINE COMPLETED WITH ERRORS")
            print("⚠ Check logs for details")
            print("⚠ Some phases may have failed")
        print("="*80)