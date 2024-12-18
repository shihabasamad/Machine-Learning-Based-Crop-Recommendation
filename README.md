# Machine Learning-Based Crop Recommendation System

This project builds a machine learning-based crop recommendation system to support agriculture by predicting the most suitable crops based on soil and environmental parameters. The web application ensures ease of use and accessibility for all stakeholders in agriculture.

## Dataset Overview
- **Source**: Secondary source ([Kaggle](https://www.kaggle.com/datasets/sanchitagholap/crop-and-fertilizer-dataset-for-westernmaharashtra)).
- **Size**: 4,513 records with 11 attributes.
- **Key Attributes**:
  - **Soil Characteristics**: Nitrogen, Phosphorus, Potassium (NPK), pH level, and soil color.
  - **Environmental Factors**: Temperature, rainfall, and humidity.
  - **Crop & Fertilizer Information**: Target crops and optimal fertilizer types.


## How It Works
1. **Data Input**: Users provide details like soil color, NPK levels, pH, and rainfall.
2. **Model Prediction**: The machine learning model processes the input and predicts the optimal crop.
3. **Output**: The result is displayed as a crop recommendation with details.


## Tools & Technologies
- **Machine Learning**: Python (Pandas, NumPy, Scikit-learn).
- **Algorithms Used**:
  - Random Forest (Achieved 97% accuracy).
  - Decision Tree.
  - Support Vector Machine (SVM).
  - Logistic Regression.
- **Web Deployment**: Flask framework for real-time crop recommendations.
- **Model Storage**: Pickle files (`crop_model.pkl`, `scaler.pkl`).


## Project Highlights
### Data Preprocessing:
- Addressed missing values with statistical imputation (e.g., mean for numerical features).
- Encoded categorical features (e.g., soil color, fertilizers) into numerical formats.
- Balanced the dataset using SMOTE to handle class imbalance.
- Scaled features using StandardScaler.

### Exploratory Data Analysis:
- Identified relationships between attributes using correlation analysis.
- Conducted outlier detection and removal (e.g., potassium outliers and rainfall extremes).

### Model Evaluation:
- Evaluated using accuracy, precision, recall, and F1-score.
- Random Forest outperformed others, achieving 97% F1-score across all metrics.

### Deployment:
- Developed a Flask web application for user interaction.
- Users input soil and environmental parameters to get crop recommendations.
- Real-time predictions ensure practicality for farmers.


## Key Findings
- Nitrogen and Phosphorus have strong correlations with Potassium.
- Random Forest offers robust performance in varied agricultural contexts.
- Soil color and fertilizer significantly influence crop recommendations.


## Future Scope
- Integration with IoT devices for real-time data collection.
- Expansion to include region-specific datasets for better generalization.
- Deployment on cloud platforms for scalability and accessibility.

