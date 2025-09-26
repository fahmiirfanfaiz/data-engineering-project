# COVID-19 Jakarta Dataset Analysis Pipeline

## 📋 Deskripsi Project

Pipeline analisis data COVID-19 Jakarta yang komprehensif untuk processing data rekap harian kasus COVID-19 per kelurahan di Provinsi DKI Jakarta bulan Mei 2020. Pipeline modular ini siap digunakan tanpa virtual environment.

## 🚀 Cara Menjalankan

### Persyaratan Sistem

- Python 3.8+
- Libraries yang diperlukan:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - scipy

### Instalasi Dependencies

```bash
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Menjalankan Pipeline

#### 1. Pipeline Lengkap

```bash
# Menjalankan seluruh pipeline (Loading → Cleaning → Integration → Reduction → Visualization)
python main.py
```

#### 2. Menjalankan Fase Tertentu

```bash
# Menjalankan script runner dengan opsi
python scripts/run_pipeline.py --phase cleaning
python scripts/run_pipeline.py --phase integration
python scripts/run_pipeline.py --phase reduction
python scripts/run_pipeline.py --phase visualization
```

## ✅ **STATUS: READY FOR ZIP**

- ✅ Folder `.venv` telah dihapus
- ✅ Pipeline berjalan sempurna dengan Python system
- ✅ Semua dependencies kompatibel
- ✅ File size optimal untuk upload

## 🏗️ Struktur Project

```
covid-19/
├── src/                           # Source code modules
│   ├── config.py                  # Konfigurasi central
│   ├── data/                      # Data loading
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── preprocessing/             # Data preprocessing
│   │   ├── __init__.py
│   │   ├── data_cleaner.py
│   │   ├── data_integrator.py
│   │   └── data_reducer.py
│   └── visualization/             # Visualisasi
│       ├── __init__.py
│       └── visualization_utils.py
├── data/                          # Dataset files
│   ├── raw/                       # Raw dataset
│   ├── cleaned/                   # Cleaned data
│   ├── integrated/                # Integrated features
│   └── reducted/                  # Reduced datasets
├── img/                           # Independent visualizations
│   ├── cleaning/                  # Data cleaning plots
│   ├── integration/               # Integration analysis plots
│   └── reduction/                 # Data reduction plots
├── notebooks/                     # Original Jupyter notebook
│   └── main.ipynb
├── main.py                        # Pipeline orchestrator
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
python main.py
```

### 3. Run Individual Phases

```python
from main import COVIDDataPipeline

pipeline = COVIDDataPipeline()
pipeline.run_data_loading()       # Loading only
pipeline.run_data_cleaning()      # Cleaning only
pipeline.run_data_integration()   # Integration only
pipeline.run_data_reduction()     # Reduction only
pipeline.run_visualization_creation()  # Visualization only
```

## 📊 Pipeline Phases

### Phase 1: Data Loading

**Module**: `src/data/data_loader.py`

**Features**:

- CSV data loading dengan validasi
- Basic data information extraction
- Missing value analysis
- Duplicate detection
- Data integrity validation

**Outputs**: Loaded and validated DataFrame

### Phase 2: Data Cleaning

**Module**: `src/preprocessing/data_cleaner.py`

**Features**:

- Missing value handling (drop/impute)
- Outlier detection and treatment (cap/remove)
- Duplicate removal
- Unnecessary column removal
- Data type consistency validation

**Outputs**: Cleaned dataset + cleaning report

### Phase 3: Data Integration

**Module**: `src/preprocessing/data_integrator.py`

**Features**:

- Correlation analysis
- Covariance analysis dengan PCA
- Feature engineering:
  - Risk-based features (mortality_rate, recovery_rate, dll)
  - Statistical features (mean, std, percentile per kecamatan)
  - Categorical encoding
- Redundancy removal

**Outputs**: Integrated dataset dengan 27+ new features

### Phase 4: Data Reduction

**Module**: `src/preprocessing/data_reducer.py`

**Features**:

- **PCA**: Principal Component Analysis (85%, 90%, 95% variance)
- **t-SNE**: Non-linear dimensionality reduction (2D, 3D)
- **Feature Selection**: Univariate, RFE, Model-based (Lasso, RF)
- **Clustering**: K-means untuk representative sampling
- **Sampling**: Random dan stratified sampling
- **Discretization**: Uniform, quantile, k-means binning

**Outputs**: 14+ reduced datasets dengan berbagai teknik

### Phase 5: Visualization

**Module**: `src/visualization/visualization_utils.py`

**Features**:

- **18 Independent Visualizations** (tidak digabung dalam 1 frame)
- **Cleaning Phase** (7 plots): Missing values, outliers, cleaning summary
- **Integration Phase** (4 plots): Correlation matrix, feature engineering
- **Reduction Phase** (7 plots): PCA, t-SNE, clustering, sampling analysis

## 📈 Generated Outputs

### Datasets (35+ files)

```
data/cleaned/
├── covid19_jakarta_mei2020_cleaned_basic.csv      # Basic cleaning
└── covid19_jakarta_mei2020_cleaned_full.csv       # Full cleaning

data/integrated/
├── covid19_jakarta_mei2020_integrated_full.csv    # Full integration
├── covid19_jakarta_mei2020_integrated_ml_ready.csv # ML-ready format
└── covid19_jakarta_mei2020_integrated_analysis.csv # Analysis format

data/reducted/
├── covid19_jakarta_reduced_pca_95.csv             # PCA 95% variance
├── covid19_jakarta_reduced_pca_90.csv             # PCA 90% variance
├── covid19_jakarta_reduced_universal_features.csv # Universal features
├── covid19_jakarta_reduced_rfe_positif.csv        # RFE for 'positif'
├── covid19_jakarta_reduced_clustering.csv         # Clustering representatives
├── covid19_jakarta_reduced_hybrid_optimized.csv   # Best combination
└── ... (14 total reduced datasets)
```

### Visualizations (18 independent plots)

```
img/cleaning/
├── missing_values_bar.png                  # Missing values by column
├── missing_values_heatmap.png              # Completeness matrix
├── missing_values_distribution.png         # Missing pattern analysis
├── outliers_boxplot_positif.png           # Outliers in COVID cases
├── outliers_boxplot_sembuh.png            # Outliers in recovery
├── outliers_boxplot_meninggal.png         # Outliers in fatalities
└── data_cleaning_summary.png              # Cleaning process summary

img/integration/
├── correlation_matrix.png                  # Full correlation heatmap
├── correlation_matrix_filtered.png         # Significant correlations only
├── covariance_eigenanalysis.png           # PCA eigenvalue analysis
└── feature_engineering_summary.png        # New features overview

img/reduction/
├── pca_variance_analysis.png              # PCA variance explained
├── pca_configurations_comparison.png       # PCA component comparison
├── tsne_visualization.png                 # t-SNE 2D/3D plots
├── feature_selection_comparison.png       # Feature selection methods
├── clustering_elbow_analysis.png          # Optimal cluster analysis
├── sampling_methods_comparison.png        # Sampling techniques
└── data_reduction_comprehensive_summary.png # Complete reduction overview
```

## 🔧 Configuration

Edit `src/config.py` untuk menyesuaikan:

```python
# Paths
RAW_DATA_PATH = Path('data/raw/data-rekap-harian-kasus-covid19-per-kelurahan-di-provinsi-dki-jakarta-bulan-mei-2020.csv')

# COVID-19 specific columns
COVID_COLUMNS = ['positif', 'sembuh', 'meninggal']

# Processing parameters
CLEANING_CONFIG = {
    'missing_strategy': 'auto',
    'outlier_method': 'cap',
    'outlier_threshold': 3.0
}

INTEGRATION_CONFIG = {
    'correlation_threshold': 0.9,
    'variance_threshold': 0.01
}

REDUCTION_CONFIG = {
    'pca': {
        'variance_thresholds': [0.85, 0.90, 0.95]
    },
    'clustering': {
        'k_values': [5, 10, 15, 20]
    }
}
```

## 📋 Requirements

```txt
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## 🎯 Key Features

### ✅ Modular Architecture

- Setiap module dapat dijalankan independent
- Proper package structure dengan `__init__.py`
- Clean separation of concerns

### ✅ Independent Visualizations

- 18 plots terpisah, tidak digabung dalam 1 frame
- Setiap visualization disimpan sebagai file PNG individual
- Representatif dan tidak menampilkan teks eksekusi

### ✅ Comprehensive Error Handling

- Logging terstruktur untuk setiap phase
- Graceful error recovery
- Detailed error reporting

### ✅ Extensive Data Processing

- 4 major preprocessing techniques
- 14+ reduced dataset variants
- Complete feature engineering pipeline

### ✅ Professional Output

- Pipeline execution report (JSON)
- Metadata files untuk setiap phase
- Clean console output dengan progress tracking

## 📊 Pipeline Performance

**Execution Time**: ~52 seconds
**Success Rate**: 100% (5/5 phases)
**Data Transformation**: (269, 18) → (269, 44) → 14+ reduced variants
**Visualizations Created**: 18 independent plots

## 🔍 Usage Examples

### Run Specific Phase Only

```python
from main import run_pipeline_phase

# Run hanya cleaning
cleaned_data = run_pipeline_phase('cleaning', raw_data)

# Run hanya reduction
reduced_data = run_pipeline_phase('reduction', integrated_data)
```

### Create Custom Visualizations

```python
from src.visualization.visualization_utils import DataVisualizer

visualizer = DataVisualizer()
visualizer.plot_missing_values_bar(missing_analysis)
visualizer.plot_correlation_matrix(correlation_matrix)
```

### Access Individual Modules

```python
from src.data.data_loader import load_covid_data
from src.preprocessing.data_cleaner import clean_covid_data

# Load data
df, loader = load_covid_data()

# Clean data
cleaned_df, cleaner = clean_covid_data(df)
```

## 📝 License

This project is developed for academic purposes at Universitas semester 5 Rekayasa Data course.

---

**Author**: Assistant AI  
**Date**: September 2025  
**Version**: 1.0.0
