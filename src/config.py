# Configuration file for COVID-19 Jakarta Data Preprocessing Pipeline
import os
import sys
import warnings
import logging
from pathlib import Path

# Handle imports with try-except for better error handling
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Warning: Some dependencies not found: {e}")
    print("Please install with: pip install -r requirements.txt")

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
NOTEBOOKS_DIR = BASE_DIR / 'notebooks'
IMG_DIR = BASE_DIR / 'img'
SRC_DIR = BASE_DIR / 'src'

# Data paths
RAW_DATA_PATH = DATA_DIR / 'raw' / 'data-rekap-harian-kasus-covid19-per-kelurahan-di-provinsi-dki-jakarta-bulan-mei-2020.csv'
CLEANED_DIR = DATA_DIR / 'cleaned'
INTEGRATED_DIR = DATA_DIR / 'integrated'
REDUCTED_DIR = DATA_DIR / 'reducted'

# Image directories for visualization outputs
IMG_CLEANING_DIR = IMG_DIR / 'cleaning'
IMG_INTEGRATION_DIR = IMG_DIR / 'integration'
IMG_REDUCTION_DIR = IMG_DIR / 'reduction'

# Create directories if they don't exist
for directory in [CLEANED_DIR, INTEGRATED_DIR, REDUCTED_DIR, 
                 IMG_CLEANING_DIR, IMG_INTEGRATION_DIR, IMG_REDUCTION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data cleaning parameters
CLEANING_CONFIG = {
    'missing_threshold': 0.5,  # Drop columns with > 50% missing values
    'outlier_method': 'iqr',   # Method for outlier detection
    'outlier_threshold': 1.5,  # IQR multiplier for outlier detection
    'categorical_encoding': 'label',  # Method for categorical encoding
    'numerical_scaling': 'standard',  # Method for numerical scaling
}

# Data integration parameters
INTEGRATION_CONFIG = {
    'correlation_threshold': 0.9,  # Threshold for high correlation
    'covariance_threshold': 0.95,  # Threshold for high covariance
    'variance_threshold': 0.01,    # Threshold for low variance features
    'feature_engineering': {
        'risk_components': True,   # Create risk-based features
        'statistical_features': True,  # Create statistical features
        'categorical_features': True,  # Process categorical features
    }
}

# Data reduction parameters
REDUCTION_CONFIG = {
    'pca': {
        'variance_thresholds': [0.95, 0.90, 0.85],  # PCA variance retention levels
    },
    'feature_selection': {
        'k_best': 10,              # Number of features for univariate selection
        'rfe_features': 10,        # Number of features for RFE
        'lasso_alpha': [0.001, 0.01, 0.1, 1.0],  # Alpha values for Lasso
    },
    'clustering': {
        'k_values': [5, 10, 15, 20],  # K values for K-means clustering
    },
    'sampling': {
        'ratios': [0.5, 0.7],      # Sampling ratios
        'strategies': ['stratified', 'random'],  # Sampling strategies
    },
    'discretization': {
        'n_bins': [5, 10, 15],     # Number of bins for discretization
        'strategies': ['uniform', 'quantile', 'kmeans']  # Binning strategies
    }
}

# Visualization parameters
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'default',
    'color_palette': 'husl',
    'save_format': 'png',
    'save_dpi': 300,
}

# Target variables for analysis
TARGET_VARIABLES = ['positif', 'sembuh', 'meninggal']

# Key columns for data integrity
KEY_COLUMNS = ['periode_data', 'id_kel', 'nama_kelurahan', 'nama_kecamatan']

# COVID-19 specific columns
COVID_COLUMNS = [
    'positif', 'sembuh', 'meninggal', 'odp_proses', 'odp_selesai',
    'pdp_proses', 'pdp_selesai', 'otg_proses', 'otg_selesai'
]

# Essential columns that should not be dropped
ESSENTIAL_COLUMNS = KEY_COLUMNS + COVID_COLUMNS

# Random seed for reproducibility
RANDOM_SEED = 42

# Pandas display options
PANDAS_CONFIG = {
    'display.max_columns': None,
    'display.width': None,
    'display.max_colwidth': None,
}

def setup_pandas_options():
    """Setup pandas display options"""
    import pandas as pd
    for option, value in PANDAS_CONFIG.items():
        pd.set_option(option, value)

def setup_matplotlib():
    """Setup matplotlib configuration"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use(VISUALIZATION_CONFIG['style'])
    sns.set_palette(VISUALIZATION_CONFIG['color_palette'])
    
    # Set default figure size
    plt.rcParams['figure.figsize'] = VISUALIZATION_CONFIG['figure_size']
    plt.rcParams['figure.dpi'] = VISUALIZATION_CONFIG['dpi']

def setup_warnings():
    """Setup warning filters"""
    import warnings
    warnings.filterwarnings('ignore')

def setup_logging():
    """Setup logging configuration"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def setup_directories():
    """Setup and create all necessary directories"""
    for directory in [CLEANED_DIR, INTEGRATED_DIR, REDUCTED_DIR, 
                     IMG_CLEANING_DIR, IMG_INTEGRATION_DIR, IMG_REDUCTION_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def setup_environment():
    """Setup complete environment configuration"""
    setup_pandas_options()
    setup_matplotlib()
    setup_warnings()
    setup_directories()
    
    # Set random seed
    import numpy as np
    np.random.seed(RANDOM_SEED)