# Forest Cover Type Classification with Support Vector Machines (SVM)

A comprehensive machine learning project implementing Support Vector Machine (SVM) algorithms to classify forest cover types using the Covertype dataset. This project demonstrates advanced data preprocessing, dimensionality reduction techniques, and SVM optimization for multi-class classification.

## 🌲 Project Overview

This project classifies forest cover types based on cartographic variables like elevation, slope, aspect, distance to water sources, and soil characteristics. The dataset contains 54 features and 7 different cover types from four wilderness areas in Northern Colorado.

### Cover Types
1. **Spruce/Fir**
2. **Lodgepole Pine**
3. **Ponderosa Pine**
4. **Cottonwood/Willow**
5. **Aspen**
6. **Douglas-fir**
7. **Krummholz**

## 📊 Dataset Information

- **Dataset**: Forest Cover Type (Covertype) from UCI Machine Learning Repository
- **Total Samples**: 581,012 instances
- **Features**: 54 cartographic variables
- **Classes**: 7 forest cover types
- **Data Type**: Multivariate classification
- **Source**: US Forest Service (USFS) Region 2 Resource Information System (RIS)

### Key Features
- **Elevation**: 1,859 - 3,858 meters
- **Aspect**: 0 - 360 degrees (azimuth)
- **Slope**: 0 - 52 degrees
- **Distance Features**: Horizontal distances to hydrology, roadways, and fire points
- **Soil Types**: 40 binary soil type variables
- **Wilderness Areas**: 4 binary wilderness area designations
- **Hillshade**: Illumination index at 9am, noon, and 3pm

## 🚀 Features & Implementation

### Data Analysis & Visualization
- **Statistical Analysis**: Comprehensive exploratory data analysis with mean, median, standard deviation
- **Correlation Analysis**: Feature correlation matrices and relationships
- **Terrain Analysis**: Elevation vs. slope patterns, aspect distributions
- **Illumination Analysis**: Hillshade patterns throughout the day
- **Wilderness Area Analysis**: Cover type distribution across different wilderness areas

### Data Preprocessing
- **Scaling Methods Comparison**:
  - StandardScaler (recommended)
  - MinMaxScaler
  - RobustScaler
  - PowerTransformer
  - QuantileTransformer
- **Train-Test Split**: 80% training, 20% testing with stratified sampling
- **Missing Value Handling**: Complete dataset with no missing values

### Dimensionality Reduction
- **Principal Component Analysis (PCA)**:
  - Variance analysis (85%, 90%, 95%, 99% thresholds)
  - Component interpretation and visualization
  - Optimal component selection (43 components for 95% variance)
- **Linear Discriminant Analysis (LDA)**:
  - Supervised dimensionality reduction
  - 6 LDA components for maximum class separation

### SVM Implementation
- **Multiple SVM Configurations**:
  - Original features with RBF kernel
  - PCA-reduced features with RBF kernel
  - LDA-reduced features with Linear kernel
  - LDA-reduced features with RBF kernel
- **Hyperparameter Optimization**: C parameter tuning, kernel selection
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score

## 📁 Project Structure

```
SVM/
├── code.ipynb              # Main Jupyter notebook with complete implementation
├── compressed_data.csv.gz  # Compressed dataset (if available)
├── README.md              # Project documentation (this file)
└── requirements.txt       # Python dependencies (create as needed)
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab

### Required Libraries
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Detailed Dependencies
- **numpy**: Numerical computations
- **pandas**: Data manipulation and analysis
- **matplotlib**: Static plotting and visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms and utilities

## 🏃‍♂️ Usage

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SVM
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter notebook**:
   ```bash
   jupyter notebook code.ipynb
   ```

4. **Execute cells sequentially** to:
   - Load and explore the Covertype dataset
   - Perform comprehensive data analysis
   - Apply preprocessing and scaling techniques
   - Implement dimensionality reduction
   - Train and evaluate SVM models
   - Compare model performances

## 📈 Model Performance

### Best Model Configuration
- **Model**: SVM with optimized hyperparameters
- **Features**: Based on comprehensive preprocessing analysis
- **Accuracy**: High classification accuracy across all cover types
- **Kernels Tested**: Linear, RBF (Radial Basis Function)
- **Preprocessing**: StandardScaler with optional PCA/LDA

### Model Comparison
The project evaluates multiple configurations:
1. **SVM_Original_RBF**: All 54 features with RBF kernel
2. **SVM_PCA_RBF**: 43 PCA components with RBF kernel
3. **SVM_LDA_Linear**: 6 LDA components with Linear kernel
4. **SVM_LDA_RBF**: 6 LDA components with RBF kernel

## 🔍 Key Insights

### Data Insights
- **Elevation**: Strong predictor of cover type (different types thrive at different altitudes)
- **Wilderness Areas**: Each area has distinct cover type distributions
- **Soil Types**: Significant impact on forest cover classification
- **Illumination Patterns**: Hillshade variations affect forest growth patterns

### Model Insights
- **Feature Scaling**: Critical for SVM performance
- **Dimensionality Reduction**: PCA maintains most information with significant feature reduction
- **Kernel Selection**: RBF kernel generally outperforms linear for this dataset
- **Class Balance**: Stratified sampling ensures representative train/test splits

## 📊 Visualizations

The project includes comprehensive visualizations:
- **Correlation Heatmaps**: Feature relationships
- **Distribution Plots**: Elevation, slope, and aspect patterns
- **Box Plots**: Cover type characteristics by wilderness area
- **Scatter Plots**: PCA component analysis
- **Confusion Matrix**: Model prediction accuracy
- **Performance Comparisons**: Model accuracy and training time analysis

## 🎯 Applications

This forest cover type classification model can be applied to:
- **Forest Management**: Planning and resource allocation
- **Conservation**: Identifying critical habitats
- **Environmental Monitoring**: Tracking forest changes over time
- **Ecological Research**: Understanding forest ecosystem patterns
- **Land Use Planning**: Informed decision-making for development

## 🔮 Future Enhancements

- **Ensemble Methods**: Random Forest, Gradient Boosting comparison
- **Deep Learning**: Neural network implementations
- **Feature Engineering**: Creating new meaningful features
- **Cross-Validation**: K-fold validation across geographical regions
- **Real-time Prediction**: Web application deployment
- **Temporal Analysis**: Multi-year forest change detection

## 📚 Technical Details

### Preprocessing Pipeline
1. **Data Loading**: Fetch Covertype dataset from sklearn
2. **Exploratory Analysis**: Statistical summaries and visualizations
3. **Feature Scaling**: Multiple scaling method comparison
4. **Dimensionality Reduction**: PCA and LDA implementation
5. **Train-Test Split**: Stratified 80-20 split

### Model Training Pipeline
1. **Model Configuration**: Multiple SVM setups
2. **Hyperparameter Selection**: Optimized C and gamma values
3. **Training**: Fit models on preprocessed data
4. **Evaluation**: Comprehensive performance metrics
5. **Comparison**: Model selection based on accuracy and efficiency

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features or improvements
- Submit pull requests
- Share insights and optimizations

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Covertype dataset
- **US Forest Service** for the original forest cover data
- **Scikit-learn** community for excellent machine learning tools
- **Open-source community** for the various visualization and analysis libraries

## 📞 Contact

For questions, suggestions, or collaborations, please reach out through:
- GitHub Issues
- Project Repository

---

