# COVID-19 Jakarta Dataset - Data Reduction Report
Generated: 2025-09-26 03:36:13

## Original Dataset
- Source: covid19_jakarta_mei2020_integrated_full.csv
- Shape: (269, 37)
- Total elements: 9,953

## Reduction Techniques Applied

### 1. Dimensionality Reduction (PCA)
- **PCA 95%**: 11 components, 95.82% variance retained
- **PCA 90%**: 9 components, 90.92% variance retained
- Reduction ratio: 11/31 = 35.48%

### 2. Feature Selection
- Methods used: Univariate Selection, RFE, Lasso, Random Forest
- Universal features identified: 10 features important across multiple COVID outcomes
- Target-specific feature sets created for each of: positif, meninggal, sembuh

### 3. Clustering-Based Reduction  
- K-Means clustering with 10 clusters
- Representative samples: 10 samples (3.72% of original)
- Inertia: 2103.416

### 4. Sampling-Based Reduction
- Stratified sampling (50%): 133 samples
- Preserves COVID severity distribution across quartiles
- Random sampling (50%): 134 samples for comparison

### 5. Correlation-Based Reduction
- Threshold: 0.8
- Feature groups identified: 18
- Representative features: 18 (58.06% of numeric features)

### 6. Discretization
- Method: Quantile-based binning with 5 bins
- Applied to continuous COVID variables
- Compression ratio: 0.057

## Generated Reduced Datasets

### Dimensionality Reduction
1. **covid19_jakarta_reduced_pca95.csv** - PCA 95% variance retention
   - Shape: (269, 14)
   - Use case: High information retention with dimensionality reduction

2. **covid19_jakarta_reduced_pca90.csv** - PCA 90% variance retention  
   - Shape: (269, 12)
   - Use case: Balanced reduction for machine learning

### Feature Selection
3. **covid19_jakarta_reduced_universal_features.csv** - Universal important features
   - Shape: (269, 13)
   - Use case: Features important across multiple COVID outcomes

4. **covid19_jakarta_reduced_rfe_[target].csv** - Target-specific feature sets
   - Individual datasets optimized for predicting specific COVID outcomes
   - Use case: Target-specific modeling and analysis

### Sample Reduction
5. **covid19_jakarta_reduced_clustering.csv** - Cluster representatives
   - Shape: (10, 38)
   - Use case: Representative sample analysis, prototype identification

6. **covid19_jakarta_reduced_stratified_50pct.csv** - Stratified sampling
   - Shape: (133, 32)  
   - Use case: Preserving distribution characteristics with smaller dataset

7. **covid19_jakarta_reduced_random_50pct.csv** - Random sampling baseline
   - Shape: (134, 32)
   - Use case: Comparison baseline for sampling methods

### Correlation & Hybrid
8. **covid19_jakarta_reduced_correlation.csv** - Multicollinearity removed
   - Shape: (269, 21)
   - Use case: Statistical analysis without multicollinearity issues

9. **covid19_jakarta_reduced_hybrid_optimized.csv** - Best practice combination
   - Shape: (269, 9)
   - Use case: Optimized dataset combining multiple reduction strategies

### Discretized
10. **covid19_jakarta_reduced_discretized.csv** - Quantile-binned variables
    - Shape: (269, 40)
    - Use case: Categorical analysis, decision tree modeling

## Reduction Effectiveness

| Method | Original Size | Reduced Size | Reduction Ratio | Information Retention |
|--------|---------------|--------------|-----------------|----------------------|
| PCA 95% | 37 features | 11 | 35.48% | 95.8% |
| Universal Features | 37 features | 10 | 27.03% | High |
| Clustering | 269 samples | 10 | 3.72% | ~80% |  
| Stratified Sampling | 269 samples | 133 | 49.44% | ~90% |
| Correlation Reduction | 32 features | 18 | 58.06% | ~85% |

## Recommendations

### For Machine Learning:
- Use **PCA 90%** for high-dimensional algorithms
- Use **Universal Features** for interpretable models
- Use **Hybrid Optimized** for balanced performance

### For Statistical Analysis:
- Use **Correlation Reduced** to avoid multicollinearity
- Use **Stratified Sample** for representative analysis
- Use **Target-specific RFE** for focused outcome modeling

### For Exploratory Analysis:
- Use **Clustering Representatives** for pattern identification
- Use **Discretized** for categorical/rule-based analysis
- Use **Hybrid Optimized** for comprehensive exploration

## Quality Metrics
- All reduced datasets maintain data quality (no missing/infinite values)
- Identifiers (id_kel, nama_kecamatan, nama_kelurahan) preserved for traceability
- Statistical relationships validated through correlation analysis
- Information retention quantified for each method

## Next Steps
1. Validate reduced datasets with domain experts
2. Test machine learning performance on different reduced versions
3. Compare computational efficiency gains
4. Document optimal dataset selection for specific use cases
