# Crop variety recommendation using Predictive Modeling and Analysis

> Predicting the ideal crop variety is essential for farmers because it helps maximize yield, minimize risks, and efficiently use resources. By choosing the right variety tailored to local soil, climate, and environmental conditions, farmers can increase productivity and reduce the likelihood of crop failure due to pests, diseases, or extreme weather. This approach also supports sustainable farming by optimizing inputs like water and fertilizers, making agriculture more resilient to climate change and ensuring better food security and profitability.


## Table of Contents

- [Overview](#overview)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [How It Works](#how-it-works)
- [Code Snippets](#code-snippets)
- [Features](#features)
- [Status](#status)
- [Challenges](#challenges)
- [Learnings](#learnings)
- [Contributors](#contributors)
- [Contact](#contact)

## Overview

This project focuses on building predictive models using Python to analyze datasets effectively. The primary goal is to preprocess data, engineer features, and train models that yield actionable insights. This notebook was developed during a Datathon challenge. The project won the **2nd Runner-Up Prize**. View the badge [here](https://badgr.com/public/assertions/rpl3BidYQJKToosP9B4jLg?identity__email=ogupta@horizon.csueastbay.edu).

## Screenshots

### Visualizations from the Project:

1. **EDA and Trends**:
   ![EDA](./img/EDA.png)
   - This chart explores the distribution of plant varieties across different zones and highlights key trends.

2. **Predictive Model Accuracy**:
   ![Model Accuracy](./img/ModelAccuracy.png)
   ![Categories](./img/MA2.png)
   - A confusion matrix and accuracy metrics for the Random Forest model achieving 80% accuracy.

4. **Recommendations**:
   ![Recommendations](./img/Recommendations.png)
   - This dashboard showcases crop recommendations tailored for zones, soil types, and environmental conditions.

## Technologies Used

- **Python**
- **Pandas**
- **Numpy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone [repo-url]
   ```
2. Navigate to the project directory and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Datathon.ipynb
   ```

## How It Works

1. **Data Loading and Exploration**: Load datasets and explore descriptive statistics.
2. **Preprocessing**: Handle missing values, encode categorical data, and normalize features.
3. **Feature Engineering**: Select and transform features for model input.
4. **Model Training**: Train a random forest classifier and optimize hyperparameters using grid search.
5. **Evaluation**: Assess model accuracy and visualize results.

## Code Snippets

```python
# Example: Random Forest Classifier with Grid Search
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best Params: {grid_search.best_params_}")
```

## Features

### Current Features

- Real-time crop variety predictions based on input parameters
- Dashboard visualization for trends and insights
- Recommendations tailored to soil, zone, and environmental conditions

### Future Enhancements

- Integrate an AI-powered chatbot to answer farmer queries interactively
- Expand prediction capabilities to include weather-driven crop forecasting

## Status

- **Completed**: Initial version finalized, but further improvements are welcome.

## Challenges

- Imbalanced datasets impacted model performance.
- Fine-tuning hyperparameters required extensive compute resources.

## Learnings

- Improved data preprocessing and feature engineering skills.
- Gained expertise in hyperparameter tuning using Grid Search.
- Learned effective ways to visualize and interpret results.

## Contact

Feel free to reach out for collaboration or feedback. **Created by:** Me and my teammates - Komal Nayak, Mahima Advilkar, Vibha Gupta

Connect with me:

- **Email**: [osheengupta1994@gmail.com](mailto\:osheengupta1994@gmail.com)
- **GitHub**: [osheengupta](https://github.com/osheengupta)
- **LinkedIn**: [Osheen Gupta](https://linkedin.com/in/osheengupta/)
