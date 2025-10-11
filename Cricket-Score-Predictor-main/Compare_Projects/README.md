## Cricket-score-predictor
# A XgBoost based Cricket Score Predictor

Dataset: https://www.kaggle.com/veeralakrishna/cricsheet-a-retrosheet-for-cricket?select=t20s

## Overview
This project is a Cricket Score Predictor that leverages machine learning techniques to forecast the final score a batting team is likely to achieve in a cricket match. It is trained on historical cricket match data, and the prediction model is built using the XGBoost algorithm.

## Features
- **Feature Engineering:** The code employs extensive feature engineering, including cumulative scores, over details, wicket-related features, and a rolling sum of runs in the last five overs.
- **Data Cleaning:** Missing city values are handled by extracting information from the venue column, and the dataset is filtered to include only matches in cities with substantial occurrences.
- **Model Construction:** A machine learning pipeline is utilized, incorporating one-hot encoding, standard scaling, and an XGBoost Regressor for accurate score predictions.
- **Model Persistence:** The trained model is saved using Pickle for future use, allowing for quick deployment without the need for retraining on historical data.

## Instructions
1. **Data Loading:** The code loads a preprocessed dataset (`dataset_level2.pkl`) using the Pandas library. Make sure to have this dataset available.
2. **Run the Code:** Execute the provided code to preprocess the data, engineer features, train the model, and evaluate its performance.
3. **Trained Model:** The trained model is saved as `pipe.pkl`. You can use this model for score predictions without retraining.

## Metrics
- **R-squared (R²):** The R-squared value provides insight into the proportion of variance in the target variable explained by the model. A higher R-squared indicates better predictive performance.
- **Mean Absolute Error (MAE):** MAE measures the average absolute difference between predicted and actual values, offering a metric for the accuracy of score predictions.

## Evaluation Results
The output of the model evaluation metrics is as follows:
- **R-squared (R²):** 0.98
- **Mean Absolute Error (MAE):** 1.712

## Version Information
- **XGBoost Version:** The code uses XGBoost version [2.0.3].

Feel free to customize the README with specific details about your project and environment.
