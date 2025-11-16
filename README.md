Breast Cancer Classification - Machine Learning Assessment

Project Date: November 16, 2025  
Assessment: ML Classification Project

---

Table of Contents

1. Project Overview
2. Dataset Description
3. Algorithms Used
4. Project Structure
5. Installation & Setup
6. Results Summary
7. Model Performance Comparison
8. Key Findings
9. Suggestions for Improvement
10. Usage Instructions
11. References

---

Project Overview

This project implements a comprehensive machine learning pipeline to classify breast cancer cases using multiple supervised learning algorithms. The objective is to compare the performance of different models and hyperparameter configurations to identify the most effective approach for this medical classification task.

Key Objectives:
- Implement and evaluate 9 different model configurations
- Compare Logistic Regression, Random Forest, and Neural Network approaches
- Provide detailed performance metrics for each configuration
- Identify the best-performing model for deployment

---

Dataset Description

Source
The breast cancer dataset used in this project contains clinical measurements and features derived from digitized images of breast mass samples.

Dataset Characteristics

| Property | Value |
|----------|-------|
| Type | Classification (Binary) |
| Format | Excel/CSV |
| File Name | `breast_cancer.csv` |
| Target Variable | Diagnosis (Malignant/Benign) |

Features

The dataset contains numerical features describing characteristics of cell nuclei present in breast mass samples, including:
- Radius: Mean distances from center to points on the perimeter
- Texture: Standard deviation of gray-scale values
- Perimeter: Perimeter of the nucleus
- Area: Area of the nucleus
- Smoothness: Local variation in radius lengths
- Compactness: Perimeter² / Area - 1.0
- Concavity: Severity of concave portions of the contour
- Concave Points: Number of concave portions of the contour
- Symmetry: Symmetry of the nucleus
- Fractal Dimension: "Coastline approximation" - 1

Note: The exact feature set may vary. See the notebook for complete feature list and data exploration.

Data Quality
- Missing Values: Handled using median imputation
- Feature Scaling: Standardized using StandardScaler (mean=0, std=1)
- Train-Test Split: 80% training, 20% testing with stratified sampling
- Cross-Validation: 10-fold stratified cross-validation for robust performance estimation
- Class Balance: Verified to ensure representative distribution

Exploratory Data Analysis
- Correlation Heatmap: Visualizes feature relationships and identifies highly correlated pairs
- Feature Distributions: Histograms for all 30 numerical features
- Statistical Summary: Comprehensive descriptive statistics for each feature

---

Algorithms Used

1. Logistic Regression

A linear model for binary classification that models the probability of class membership using the logistic function.

Three Configurations Tested:

| Config | Regularization | Strength (C) | Solver | Description |
|--------|----------------|--------------|--------|-------------|
| LR #1 | L2 (Ridge) | 1.0 | lbfgs | Default configuration with L2 penalty |
| LR #2 | L1 (Lasso) | 0.5 | liblinear | Stronger regularization for feature selection |
| LR #3 | L2 (Ridge) | 10.0 | lbfgs | Weaker regularization for more flexibility |

Strengths:
- Fast training and inference
- Interpretable model coefficients
- Probabilistic predictions
- Works well with linearly separable data

Weaknesses:
- Assumes linear relationship
- May underfit complex patterns
- Sensitive to feature scaling

---

2. Random Forest

An ensemble learning method that constructs multiple decision trees and aggregates their predictions.

Three Configurations Tested:

| Config | Trees (n) | Max Depth | Min Split | Min Leaf | Description |
|--------|-----------|-----------|-----------|----------|-------------|
| RF #1 | 100 | 10 | 5 | 2 | Moderate complexity, balanced approach |
| RF #2 | 200 | 20 | 2 | 1 | High complexity, deeper trees |
| RF #3 | 50 | 5 | 10 | 5 | Low complexity, prevents overfitting |

Strengths:
- Handles non-linear relationships
- Robust to outliers
- Provides feature importance
- Minimal hyperparameter tuning required

Weaknesses:
- Less interpretable than single trees
- Can overfit with too many deep trees
- Slower prediction time than linear models
- Memory intensive

Feature Importance Analysis:
- Comparative analysis across all 3 Random Forest configurations
- Top 15 most important features visualized
- Identifies most predictive features for malignancy classification
- Quantifies feature contribution percentages

---

3. Multi-Layer Perceptron (Neural Network)

A feedforward artificial neural network with multiple layers of neurons using backpropagation for learning.

Three Architectures Tested:

| Config | Hidden Layers | Architecture | Activation | Description |
|--------|---------------|--------------|------------|-------------|
| MLP #1 | 1 | (50) | ReLU | Simple network, fast training |
| MLP #2 | 2 | (100, 50) | ReLU | Moderate depth, balanced capacity |
| MLP #3 | 3 | (150, 100, 50) | ReLU | Deep network, high capacity |

Training Configuration:
- Optimizer: Adam (adaptive moment estimation)
- Learning Rate: Adaptive
- Regularization: L2 with varying alpha values
- Early Stopping: Enabled with 10% validation split
- Max Iterations: 500-1000 depending on architecture

Strengths:
- Captures complex non-linear patterns
- Flexible architecture
- Can achieve high accuracy
- Learns hierarchical features

Weaknesses:
- Requires significant data
- Computationally expensive
- Prone to overfitting
- "Black box" - difficult to interpret
- Sensitive to initialization and hyperparameters

Training Analysis:
- Loss curves visualized for all 3 architectures
- Convergence monitoring with early stopping
- Training dynamics analysis showing learning behavior
- Epoch-by-epoch loss reduction tracking

---

Project Structure

```
ML_assessment_2416509/
│
├── breast_cancer.csv                    # Dataset file
├── breast_cancer_ml_assessment.ipynb    # Main Jupyter Notebook
├── model_results_summary.csv            # Exported results (generated)
├── README.md                            # This file
│
└── (Optional: Add model saves, plots, reports)
```

---

Installation & Setup

Prerequisites

- Python: 3.8 or higher
- Jupyter Notebook or VS Code with Jupyter extension

Required Libraries

Install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

Or create a `requirements.txt` file:

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
openpyxl>=3.0.0
jupyter>=1.0.0
```

Then install:

```bash
pip install -r requirements.txt
```

Running the Notebook

1. Open in VS Code:
   ```bash
   code .
   ```
   Then open `breast_cancer_ml_assessment.ipynb`

2. Or use Jupyter Notebook:
   ```bash
   jupyter notebook breast_cancer_ml_assessment.ipynb
   ```

3. Run all cells sequentially to reproduce the analysis

---

Results Summary

Performance Metrics

All 9 model configurations were evaluated using the following metrics:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Accuracy | Proportion of correct predictions | Higher is better (0-1) |
| F1-Score | Harmonic mean of precision and recall | Higher is better (0-1) |
| ROC-AUC | Area under ROC curve | Higher is better (0-1) |

Expected Results Template

Note: Run the notebook to generate actual results. Below is a template:

| Model | Configuration | Accuracy | F1-Score | ROC-AUC |
|-------|---------------|----------|----------|---------|
| Logistic Regression #1 | L2, C=1.0 | 0.XXXX | 0.XXXX | 0.XXXX |
| Logistic Regression #2 | L1, C=0.5 | 0.XXXX | 0.XXXX | 0.XXXX |
| Logistic Regression #3 | L2, C=10.0 | 0.XXXX | 0.XXXX | 0.XXXX |
| Random Forest #1 | n=100, depth=10 | 0.XXXX | 0.XXXX | 0.XXXX |
| Random Forest #2 | n=200, depth=20 | 0.XXXX | 0.XXXX | 0.XXXX |
| Random Forest #3 | n=50, depth=5 | 0.XXXX | 0.XXXX | 0.XXXX |
| Neural Network #1 | (50) | 0.XXXX | 0.XXXX | 0.XXXX |
| Neural Network #2 | (100-50) | 0.XXXX | 0.XXXX | 0.XXXX |
| Neural Network #3 | (150-100-50) | 0.XXXX | 0.XXXX | 0.XXXX |

Note: Detailed results with confusion matrices and ROC curves are available in the notebook.

---

Model Performance Comparison

Visualizations Included

The notebook generates the following visualizations:

Exploratory Data Analysis:
1. Correlation Heatmap: Feature relationships with highly correlated pairs identification
2. Feature Distributions: Histograms for all 30 numerical features

Model Performance:
3. Confusion Matrices: True positives, false positives, true negatives, false negatives for all 9 models
4. ROC Curves: True positive rate vs false positive rate with AUC scores
5. Comparison Charts: Bar charts comparing all models across accuracy, F1-score, ROC-AUC
6. Combined ROC Plot: All 9 models' ROC curves on one plot

Model-Specific Analysis:
7. Random Forest Feature Importance: Top 15 features across 3 configurations
8. Neural Network Loss Curves: Training convergence for 3 architectures

Model Selection Guidelines

- For Maximum Accuracy: Choose the top-ranked accuracy model
- For Medical Applications: Prioritize models with:
  - High recall (minimize false negatives)
  - High ROC-AUC (good probability calibration)
  - Interpretability (Logistic Regression preferred)
- For Production Deployment: Consider:
  - Inference speed (Logistic Regression fastest)
  - Model size (simpler models preferred)
  - Maintenance complexity

---

Key Findings

General Observations

Note: Update this section after running the notebook with actual findings

1. Best Performing Model Type: [To be determined after running]
   - Average accuracy across configurations
   - Consistency of performance
   - Variance between configurations

2. Hyperparameter Impact:
   - Effect of regularization strength in Logistic Regression
   - Impact of tree depth and number in Random Forest
   - Influence of network architecture in Neural Networks

3. Model Complexity vs Performance:
   - Whether simpler models performed comparably to complex ones
   - Signs of overfitting in complex configurations
   - Trade-offs between interpretability and accuracy

4. Feature Importance Insights:
   - Which features contribute most to predictions (for applicable models)
   - Correlation between features and target variable

Clinical Implications

- False Negative Rate: Critical metric for cancer diagnosis
- False Positive Rate: Impact on unnecessary procedures
- Threshold Selection: May need to adjust for medical context

---

Suggestions for Improvement

1. Data-Related Improvements

- Feature Engineering:
  - Create polynomial features for non-linear relationships
  - Implement feature selection techniques (RFE, feature importance)
  - Use domain knowledge to create clinically meaningful features
  - Apply PCA for dimensionality reduction

- Data Augmentation:
  - Collect more samples if possible
  - Use SMOTE for class imbalance (if present)
  - Cross-validation with multiple folds

- Data Quality:
  - Investigate outliers using box plots and z-scores
  - Check for multicollinearity among features
  - Validate data integrity and consistency

2. Model Improvements

- Hyperparameter Optimization:
  - Implement GridSearchCV for exhaustive search
  - Use RandomizedSearchCV for faster optimization
  - Apply Bayesian optimization for complex models
  - Consider AutoML tools (Auto-sklearn, TPOT)

- Advanced Algorithms:
  - Test Gradient Boosting (XGBoost, LightGBM, CatBoost)
  - Experiment with Support Vector Machines (SVM)
  - Try ensemble methods (stacking, voting classifiers)
  - Explore deep learning architectures (custom architectures)

- Model Calibration:
  - Apply Platt scaling for probability calibration
  - Use isotonic regression for non-parametric calibration
  - Validate calibration using reliability diagrams

3. Evaluation Enhancements

- Cross-Validation: ✅ IMPLEMENTED
  - 10-fold stratified cross-validation implemented for all 9 models
  - Mean and standard deviation reported for each model
  - Generalization performance validated across multiple folds

- Additional Metrics:
  - Precision and Recall for each class
  - Matthews Correlation Coefficient (MCC)
  - Cohen's Kappa for inter-rater agreement
  - Specificity and Sensitivity
  - Positive and Negative Predictive Values

- Error Analysis:
  - Analyze misclassified samples
  - Identify patterns in false positives/negatives
  - Create decision boundary visualizations
  - Investigate prediction confidence

4. Deployment Considerations

- Model Packaging:
  - Save best models using joblib or pickle
  - Create model versioning system
  - Document model assumptions and limitations
  - Prepare model cards for transparency

- Production Pipeline:
  - Implement real-time prediction API
  - Set up model monitoring and logging
  - Create automated retraining pipeline
  - Establish model performance thresholds

- Explainability:
  - Implement SHAP values for feature importance
  - Use LIME for local interpretability
  - Create model documentation for stakeholders
  - Develop clinician-friendly interfaces

5. Ethical and Regulatory

- Bias Detection:
  - Test for demographic parity
  - Evaluate fairness metrics across subgroups
  - Conduct disparate impact analysis
  - Document potential biases

- Compliance:
  - Ensure HIPAA compliance for medical data
  - Follow FDA guidelines for medical AI (if applicable)
  - Implement audit trails
  - Maintain data privacy and security

6. Research Extensions

- Transfer Learning:
  - Use pre-trained models from similar medical domains
  - Fine-tune on breast cancer specific data
  - Explore multi-task learning

- Feature Learning:
  - Apply convolutional neural networks if image data available
  - Use autoencoders for feature extraction
  - Implement attention mechanisms

- Time Series Analysis:
  - If longitudinal data available, model disease progression
  - Analyze survival curves
  - Predict recurrence risk

---

Usage Instructions

Step-by-Step Execution

1. Prepare Environment:
   ```bash
   pip install -r requirements.txt
   ```

2. Open Notebook:
   - Launch VS Code or Jupyter
   - Open `breast_cancer_ml_assessment.ipynb`

3. Run Cells Sequentially:
   - Section 1: Import libraries
   - Section 2: Load and explore data
   - Section 3: Preprocess data
   - Section 4: Train and evaluate models
   - Section 5: Compare results

4. Review Results:
   - Check printed metrics for each model
   - Examine confusion matrices and ROC curves
   - Review summary comparison table

5. Export Results:
   - Results automatically saved to `model_results_summary.csv`
   - Use this file for report writing

Customization Options

- Modify Hyperparameters: Edit model configurations in Section 4
- Change Train-Test Split: Adjust `test_size` parameter
- Add New Models: Follow the template in Section 4
- Adjust Visualizations: Modify plotting parameters

---

References

Dataset & Medical Context

1. Breast Cancer Dataset:
   - Wisconsin Diagnostic Breast Cancer (WDBC) dataset
   - UCI Machine Learning Repository
   - [Link to dataset source if applicable]

2. Medical Background:
   - American Cancer Society - Breast Cancer Statistics
   - WHO Guidelines on Breast Cancer Screening
   - Clinical relevance of features

Machine Learning Resources

1. Scikit-learn Documentation:
   - Logistic Regression: https://scikit-learn.org/stable/modules/linear_model.html
   - Random Forest: https://scikit-learn.org/stable/modules/ensemble.html
   - Neural Networks: https://scikit-learn.org/stable/modules/neural_networks_supervised.html

2. Books & Papers:
   - Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
   - Bishop, C. M. (2006). Pattern Recognition and Machine Learning
   - Breiman, L. (2001). "Random Forests" - Machine Learning Journal

3. Evaluation Metrics:
   - ROC Curves and AUC: Fawcett, T. (2006)
   - F1-Score: Powers, D. M. (2011)
   - Confusion Matrix interpretation

Tools & Libraries

- Pandas: https://pandas.pydata.org/
- NumPy: https://numpy.org/
- Matplotlib: https://matplotlib.org/
- Seaborn: https://seaborn.pydata.org/
- Scikit-learn: https://scikit-learn.org/

---

License & Acknowledgments

License

This project is created for educational and assessment purposes.

Acknowledgments

- Dataset providers (if applicable)
- Scikit-learn developers
- Open-source community

---

Author

Name: Lord Kingsley Baffoe
Student Number: 2416509  
Date: November 16, 2025

---

Contact & Support

For questions about this project:
- Review the notebook documentation
- Check scikit-learn documentation for algorithm details
- Consult course materials and textbooks

---

Last Updated: November 16, 2025  
Version: 1.0
