#!/usr/bin/env python
# coding: utf-8

# 

# #### **Project Outline for Predicting Exam Score**
# 
# ---
# 
# #### 1. Project Setup
# - **Define the goal**: Predict exam scores based on various medical diagnosis and lifestyle variable inputs.
# - **Identify resources**: Use the dataset (`medrecords.csv`) from the provided URL.
# - **Install dependencies**: Install necessary Python libraries such as Pandas, scikit-learn, xgboost, and others needed for machine learning.
# - **Reference**: Code Cell 1
# 
# ---
# 
# #### 2. Data Loading
# - **Download and load the dataset**: Download the dataset from the GitHub repository and load it into a Pandas DataFrame.
# - **Verify data integrity**: Check the first few rows of the dataset to ensure it's correctly loaded.
# - **Reference**: Code Cell 2
# 
# ---
# 
# #### 3. Data Preprocessing
# - **One-hot encoding**: Convert categorical variables like `Gender`, `Age Group`, and `Exam Age` into numerical representations.
# - **Handle missing data**: Identify missing values and handle them by either dropping rows or imputing values.
# - **Feature scaling**: Standardize numerical features (e.g., `BMI`, `Age`) using `StandardScaler`.
# - **Reference**: Code Cells 3–5
# 
# ---
# 
# #### 4. Model Development
# - **Model selection**: Train Random Forest and XGBoost models.
# - **Train-test split**: Split the dataset into training and testing sets (80/20 split) for model evaluation.
# - **Define problem type**: Confirm that the project is a regression problem with a continuous target variable (Exam Score).
# - **Reference**: Code Cells 6, 10
# 
# ---
# 
# #### 5. Model Training
# - **Train Random Forest model**: Train a Random Forest model using the training data and evaluate its performance using MSE and R-squared.
# - **Train XGBoost model**: Train an XGBoost model using the training data and evaluate its performance.
# - **Reference**: Code Cells 6, 10
# 
# ---
# 
# #### 6. Model Evaluation
# - **Evaluate Random Forest on the test set**: Use the test set to calculate MSE and R-squared for the Random Forest model.
# - **Evaluate Random Forest on the first 5 rows**: Evaluate the first 5 rows with the Random Forest model and calculate MSE.
# - **Evaluate Random Forest with cross-validation**: Perform 5-fold cross-validation for Random Forest and compute MSE.
# - **Evaluate XGBoost on the test set**: Use the test set to calculate MSE and R-squared for the XGBoost model.
# - **Evaluate XGBoost with cross-validation**: Perform 5-fold cross-validation for XGBoost and compute MSE.
# - **Evaluate XGBoost on the first 5 and 1000 rows**: Make predictions for the first 5 and 1000 rows, calculate MSE for both.
# - **Reference**: Code Cells 7–9, 11–13
# 
# ---
# 
# #### 7. Feature Importance Evaluation
# - **Feature importance visualization**: Visualize feature importance in XGBoost using weight, gain, and cover metrics.
# - **Iterative feature removal**: Systematically remove the least important features and evaluate the model’s performance after each removal.
# - **Residual analysis**: Analyze residuals by comparing predictions with and without the `Strength` feature.
# - **Reference**: Code Cells 14–18
# 
# ---
# 
# #### 8. Model Tuning
# - **Evaluate XGBoost without key features**: Test the performance of the XGBoost model after removing significant features like `Strength` and `Age`.
# - **Hyperparameter tuning**: Optionally, adjust model hyperparameters for better performance.
# - **Reference**: Code Cells 16, 17
# 
# ---
# 
# #### 9. Model Comparison
# - **Comparison to original algorithm**: Compare model predictions to the scoring system used to generate the target variable and assess accuracy.
# - **Reference**: Code Cells 18
# 
# ---
# 
# #### 10. Documentation and Reporting
# - **Document findings**: Summarize results, including key performance metrics, error analysis, and feature importance evaluations.
# - **Present results**: Report the final model’s performance and any key insights derived from the iterative feature removal process.
# - **Future work**: Suggest improvements such as using additional data, refining feature engineering, or deploying the model in a production environment.
# - **Reference**: Final conclusions based on Code Cells 18
# 

# #### Code Cell 1: Install dependencies
# Install the necessary packages (e.g., `requests`) to work with the dataset.
# 
# #### Code Cell 2: Download the dataset
# Download the dataset from a GitHub repository using `requests` and load it into a Pandas DataFrame.
# 
# #### Code Cell 3: One-hot encoding categorical variables
# Convert categorical variables like `Gender`, `Age Group`, and `Exam Age` into numerical form using one-hot encoding.
# 
# #### Code Cell 4: Handle missing values
# Identify missing values, drop them or fill them with the mean value for numerical columns.
# 
# #### Code Cell 5: Feature scaling
# Standardize the numerical columns like `Age` and `BMI` using `StandardScaler` to normalize the values.
# 
# #### Code Cell 6: Train Random Forest model
# Train a Random Forest model and evaluate its performance on the test set using MSE and R-squared as evaluation metrics.
# 
# #### Code Cell 7: Evaluate Random Forest on the first 5 rows
# Use the Random Forest model to predict the exam scores for the first 5 rows of the dataset and calculate MSE.
# 
# #### Code Cell 8: Random Forest cross-validation
# Perform 5-fold cross-validation on the Random Forest model and compute cross-validated MSE and its mean.
# 
# #### Code Cell 9: Model evaluation on the full dataset
# Evaluate the trained Random Forest model on the entire dataset and calculate overall MSE.
# 
# #### Code Cell 10: Train XGBoost model
# Train an XGBoost model on the training dataset and evaluate its MSE and R-squared on the test set.
# 
# #### Code Cell 11: XGBoost cross-validation
# Perform 5-fold cross-validation on the XGBoost model and compute the cross-validated MSE and its standard deviation.
# 
# #### Code Cell 12: XGBoost predictions for the first 5 rows
# Predict exam scores for the first 5 rows using the XGBoost model and calculate the MSE for these predictions.
# 
# #### Code Cell 13: XGBoost predictions for the first 1000 rows
# Predict exam scores for the first 1000 rows using the XGBoost model and evaluate the MSE for these predictions.
# 
# #### Code Cell 14: Feature importance visualization in XGBoost
# Generate and visualize feature importance for the XGBoost model using different importance metrics (weight, gain, cover).
# 
# #### Code Cell 15: Residual analysis by Strength feature
# Perform residual analysis by calculating the mean absolute residuals for the `Strength` feature when it's 1 or 0.
# 
# #### Code Cell 16: Evaluate XGBoost without the 'Strength' feature
# Retrain the XGBoost model without the `Strength` feature and evaluate its performance.
# 
# ##### Code Cell 17: Evaluate XGBoost without the 'Age' feature
# Retrain the XGBoost model without the `Age` feature and evaluate its performance, including cross-validation.
# 
# #### Code Cell 18: Feature removal process and evaluation
# Iteratively remove the least important features and evaluate the XGBoost model's performance after each removal.
# 

# #### Install Dependencies and Load Data from Google Cloud Storage
# 
# - **Installed dependencies**: scikit-learn, matplotlib, and seaborn.
# - **Loaded CSV file** (`medrecords.csv`) from Google Cloud Storage using `google.cloud.storage`.
# - **Displayed first few rows of the dataset** to verify successful loading.
# 
# #### Output:
# - Dependencies were already installed.
# - File `medrecords.csv` downloaded to `local_medrecords.csv`.
# - Displayed a snippet of the dataset with columns such as `Gender`, `Age`, `BMI`, and `Exam Score`, showing values like:
# 
# | Gender | Age | Age Group | BMI  | Exam Score |
# |--------|-----|-----------|------|------------|
# | Male   | 27  | 18-44     | 22.7 | 81.0       |
# | Female | 54  | 45-64     | 28.5 | 72.0       |
# | Male   | 21  | 18-44     | 21.3 | 110.0      |
# 

# In[93]:


get_ipython().system('pip install requests')


# In[94]:


import requests
import pandas as pd

# URL to the raw CSV file on GitHub
url = 'https://raw.githubusercontent.com/Compcode1/medical-dx-exam-scores/refs/heads/master/medrecords.csv'

# Send a GET request to download the file
response = requests.get(url)
open('medrecords.csv', 'wb').write(response.content)

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('medrecords.csv')

# Display the first few rows of the dataset
df.head()


# #### Data Loading and Exploratory Data Analysis (EDA)
# 
# 1. **Data Loading**:
#    - Downloaded the dataset from GitHub and loaded it into a Pandas DataFrame.
#    - Previewed the first few rows to verify the dataset structure, which includes features such as `Gender`, `Age`, `BMI`, `Health Indicators`, `Exam Age`, and `Exam Score`.
# 
# 2. **Exploratory Data Analysis (EDA)**:
#    - Examined the basic structure and data types of the dataset.
#    - Checked for missing values in the dataset.
#    - Analyzed distributions of key features like `Exam Score`, `Age`, and `BMI` for further insights.
# 
# #### Data Preprocessing
# 
# 1. **One-Hot Encoding**:
#    - Applied one-hot encoding to categorical variables (`Gender`, `Age Group`, and `Exam Age`), converting them into numerical values for machine learning models.
# 
# 2. **Handling Missing Values**:
#    - Checked for missing values and handled them by either dropping rows with missing values or imputing with the mean for numerical columns.
# 
# 3. **Feature Scaling**:
#    - Standardized numerical features such as `BMI` and `Age` using `StandardScaler` to ensure all numerical features are on the same scale, which is important for many machine learning algorithms.
# 

# In[95]:


# One-hot encode categorical variables: 'Gender', 'Age Group', and 'Exam Age'
df_encoded = pd.get_dummies(df, columns=['Gender', 'Age Group', 'Exam Age'], drop_first=True)

# Check the result
df_encoded.head()


# In[ ]:


# Check for missing values
missing_values = df_encoded.isnull().sum()
print(missing_values)

# Option 1: Drop rows with missing values (if missing values are few)
df_cleaned = df_encoded.dropna()

# Option 2: Impute missing values (mean for numerical columns)
df_cleaned = df_encoded.fillna(df_encoded.mean())

# Check again for any missing values after handling them
df_cleaned.isnull().sum()


# In[ ]:


from sklearn.preprocessing import StandardScaler

# Select numerical columns for scaling
numerical_columns = ['BMI', 'Age']

# Initialize the scaler
scaler = StandardScaler()

# Apply the scaling to numerical columns
df_cleaned[numerical_columns] = scaler.fit_transform(df_cleaned[numerical_columns])

# Check the scaled values
df_cleaned.head()


# #### **Drop the Exam Age and Age Group columns not necessary for analysis (Redundant)**
# 

# In[ ]:


# Drop the Exam Age and Age Group columns
columns_to_drop = [col for col in df_cleaned.columns if 'Exam Age' in col or 'Age Group' in col]
df_cleaned_no_age_groups = df_cleaned.drop(columns=columns_to_drop)

# Check the updated DataFrame without Exam Age and Age Group columns
df_cleaned_no_age_groups.head()


# #### **Drop the 'Obesity' column as it is redundant with BMI**
# 

# In[ ]:


# Drop the 'Obesity' column as it is redundant with BMI
df_cleaned_no_obesity = df_cleaned_no_age_groups.drop(columns=['Obesity'])

# Display the first few rows to verify the column has been dropped
df_cleaned_no_obesity.head()


# #### **Summary of Random Forest Model Strategy and Output**
# 
# #### Code Strategy/Logistics:
# 1. **Model Initialization and Training**:
#    - A **Random Forest Regressor** is initialized with a fixed random seed (`random_state=42`) to ensure consistent results.
#    - The model is trained on the training dataset (`X_train`, `y_train`).
# 
# 2. **Model Evaluation on Test Set**:
#    - After training, predictions are made using the **test set** (`X_test`), and key performance metrics are calculated:
#      - **Mean Squared Error (MSE)**: Measures the average squared difference between the predicted and actual exam scores.
#      - **R-squared (R²)**: Measures how well the model explains the variance in the exam scores (values close to 1 indicate strong predictive power).
# 
# 3. **First 5 Rows Evaluation**:
#    - The first 5 rows of the dataset are evaluated using the trained model.
#    - Both the original exam scores and the predicted exam scores for these rows are displayed.
#    - The **MSE** for these 5 rows is calculated separately to check how well the model predicts for a small sample of data.
# 
# 4. **First 1,000 Rows Evaluation**:
#    - Predictions are made on the first 1,000 rows of the dataset.
#    - The original and predicted exam scores are compared for the first 10 rows.
#    - The **MSE** for these 1,000 rows is also calculated to analyze the model's performance on a larger sample.
# 
# #### Output Interpretation:
# 1. **Random Forest Model on Test Set**:
#    - **MSE on the test set**: `0.03095`, indicating that, on average, the squared difference between predicted and actual exam scores is very small.
#    - **R-squared on the test set**: `0.99990`, which indicates that the Random Forest model explains almost all the variance in the test data, showing excellent predictive performance.
# 
# 2. **First 5 Rows**:
#    - The predicted exam scores for the first 5 rows are significantly lower than the original exam scores.
#    - **MSE for the first 5 rows**: `145.282`, which is much higher than the test set MSE, suggesting that the model struggles to predict accurately for this small sample.
# 
# 3. **First 1,000 Rows**:
#    - The model’s predictions for the first 1,000 rows are fairly close to the actual values, with minor deviations.
#    - **MSE for the first 1,000 rows**: `150.761`, indicating that while the model performs well on average, it may still struggle with some data points.
# 
# #### Key Takeaways:
# - The Random Forest model performs **exceptionally well** on the test set, showing high accuracy and predictive power (low MSE, high R²).
# - The performance on the first 5 rows is **much worse**, likely due to the small sample size.
# - On the first 1,000 rows, the model’s performance is **more consistent** with a slightly elevated MSE, indicating it may struggle with specific cases but generally performs well.
# 

# In[102]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Initialize the Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Step 2: Train the Random Forest model
rf_model.fit(X_train, y_train)

# Step 3: Make predictions on the test set to evaluate the model
y_pred_rf = rf_model.predict(X_test)

# Step 4: Evaluate the Random Forest model performance
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Random Forest Model MSE on the test set: {mse_rf}')
print(f'Random Forest Model R-squared on the test set: {r2_rf}')

# Step 5: Take the first 5 rows from the dataset
# Ensure feature consistency between training and prediction sets
X_first_5 = X_train[X_train.columns].head(5)
y_true_first_5 = y[:5]

# Step 6: Run them through the Random Forest model to get the predicted Exam Score
y_pred_first_5_rf = rf_model.predict(X_first_5)

# Step 7: Compare the predicted values to the original Exam Scores
print("Original Exam Scores for the first 5 rows:", y_true_first_5.values)
print("Predicted Exam Scores by Random Forest for the first 5 rows:", y_pred_first_5_rf)

# Step 8: Calculate the MSE for the first 5 rows
mse_first_5_rf = mean_squared_error(y_true_first_5, y_pred_first_5_rf)
print(f"Random Forest MSE for the first 5 rows: {mse_first_5_rf}")

# Step 9: Take the first 1,000 rows from the dataset
# Again, ensure feature consistency between training and prediction sets
X_first_1000 = X_train[X_train.columns].head(1000)
y_true_first_1000 = y[:1000]

# Step 10: Run them through the Random Forest model to get the predicted Exam Score
y_pred_first_1000_rf = rf_model.predict(X_first_1000)

# Step 11: Compare the predicted values to the original Exam Scores
print("Original Exam Scores for the first 1,000 rows:", y_true_first_1000.values[:10])  # Display only first 10 for brevity
print("Predicted Exam Scores by Random Forest for the first 1,000 rows:", y_pred_first_1000_rf[:10])  # Display only first 10 for brevity

# Step 12: Calculate the MSE for the first 1,000 rows
mse_first_1000_rf = mean_squared_error(y_true_first_1000, y_pred_first_1000_rf)
print(f"Random Forest MSE for the first 1,000 rows: {mse_first_1000_rf}")


# ### Cross-Validation with Random Forest Model
# 
# #### Code Strategy/Logistics:
# 1. **Model Initialization**:
#    - A **Random Forest Regressor** is initialized with a fixed random seed (`random_state=42`) to ensure consistent and reproducible results.
# 
# 2. **5-Fold Cross-Validation**:
#    - The **cross-validation** process is performed with 5 folds, splitting the dataset into 5 different subsets.
#    - The model is trained and evaluated on these subsets, with the **Mean Squared Error (MSE)** being computed for each fold.
#    - Cross-validation helps in evaluating the model’s performance on different parts of the data, ensuring that the results are not overly dependent on a particular data split.
# 
# 3. **Performance Metrics**:
#    - **Negative MSE** scores from cross-validation are converted to positive values to make them interpretable.
#    - The mean and standard deviation of the MSE across the 5 folds are calculated to give a robust measure of model performance.
# 
# #### Output Interpretation:
# 1. **Cross-Validated MSE for Each Fold**:
#    - The MSE values for each fold are: 
#      - Fold 1: `0.02194`
#      - Fold 2: `0.03120`
#      - Fold 3: `0.02699`
#      - Fold 4: `0.03542`
#      - Fold 5: `0.02307`
#    - The variation between these values shows how the model performs on different subsets of the data. Lower values indicate better performance, and the relatively low MSE across all folds suggests the model is performing well on all parts of the data.
# 
# 2. **Mean Cross-Validated MSE**:
#    - The mean MSE across all 5 folds is `0.02773`, indicating the average error between the predicted and actual exam scores is very low, which demonstrates good predictive power.
# 
# 3. **Standard Deviation of Cross-Validated MSE**:
#    - The standard deviation of `0.00504` shows that the MSE values across the 5 folds are relatively close to each other. A low standard deviation indicates that the model’s performance is consistent across different splits of the data.
# 
# #### Key Takeaways:
# - The **Random Forest model** performs consistently well across different data splits, as shown by the low and consistent cross-validated MSE.
# - The mean cross-validated MSE (`0.02773`) demonstrates strong predictive power, with little variation between folds, indicating the model is generalizing well to unseen data.
# 

# In[103]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Initialize the Random Forest model
rf_model_cv = RandomForestRegressor(random_state=42)

# Perform 5-fold cross-validation
cv_scores_rf = cross_val_score(rf_model_cv, X, y, cv=5, scoring='neg_mean_squared_error')

# Convert negative MSE to positive
cv_mse_scores_rf = -cv_scores_rf

# Display the cross-validated MSE for each fold
print(f'Cross-Validated MSE for each fold: {cv_mse_scores_rf}')

# Display the mean and standard deviation of the cross-validated MSE
print(f'Mean Cross-Validated MSE: {cv_mse_scores_rf.mean()}')
print(f'Standard Deviation of MSE: {cv_mse_scores_rf.std()}')


# ### Running the Whole Dataset Through the Model
# 
# #### Code Strategy/Logistics:
# 1. **Full Dataset Prediction**:
#    - The entire dataset (`X`) is passed through the trained model (in this case, Random Forest.
#    - Predictions (`y_pred_all`) are generated for all rows in the dataset.
# 
# 2. **Model Performance on the Full Dataset**:
#    - The **Mean Squared Error (MSE)** is calculated for the entire dataset by comparing the predicted exam scores to the actual exam scores (`y`).
#    - MSE provides an overall measure of how well the model is performing across the entire dataset.
# 
# #### Output Interpretation:
# - **MSE for the Entire Dataset**: `148.54440731017408`
#   - This relatively high MSE suggests that the model is struggling to capture the underlying patterns across the whole dataset. Despite performing well during cross-validation and on smaller subsets, it indicates the possibility of:
#     - **Overfitting**: The model might have memorized patterns in the training data but isn’t generalizing well to the entire dataset.
#     - **High Variability**: There could be substantial variability in the dataset that the model isn’t accounting for, leading to larger errors on a broader scale.
# 
# #### Key Takeaways:
# - Although the cross-validated results were strong, the high overall MSE for the entire dataset suggests that the model might be **overfitting** or that there are other data characteristics (e.g., outliers, noise) impacting its performance. Further investigation (e.g., residual analysis) would be needed to pinpoint the exact cause of the discrepancy.
# 

# In[105]:


# Ensure the feature set used for prediction matches the one used during training
X_all_aligned = X_train.columns  # Get the feature names from the trained model's data

# Select the same columns from the full dataset X
X_aligned_for_prediction = X[X_all_aligned]

# Run the whole dataset through the model
y_pred_all = rf_model.predict(X_aligned_for_prediction)

# Calculate the overall MSE for the entire dataset
mse_all = mean_squared_error(y, y_pred_all)
print(f"MSE for the entire dataset: {mse_all}")


# #### **XGBoost Model: Predicting Exam Scores**
# 
# #### Code Strategy/Logistics:
# 1. **Data Preprocessing**:
#    - Categorical variable 'Gender' is one-hot encoded, converting it into numerical format to make it suitable for the XGBoost model.
#    - Unnecessary columns like 'Exam Age', 'Age Group', and 'Obesity' (which is redundant due to the presence of BMI) are dropped.
#    - The dataset is then split into features (X) and the target variable ('Exam Score').
# 
# 2. **Train-Test Split**:
#    - The dataset is split into training and testing sets using an 80/20 ratio. This ensures that the model is trained on 80% of the data and evaluated on the remaining 20%, simulating real-world generalization.
# 
# 3. **Feature Scaling**:
#    - Numerical features like 'Age' and 'BMI' are standardized using the `StandardScaler` to ensure that they are on the same scale, which is important for models like XGBoost that can be sensitive to feature magnitudes.
# 
# 4. **XGBoost Model Initialization and Training**:
#    - XGBoost, a powerful gradient boosting algorithm, is used for training the model on the preprocessed data. It is initialized with a random state for reproducibility and trained on the scaled training set.
# 
# 5. **Prediction and Evaluation**:
#    - The trained XGBoost model is used to predict the exam scores on the test data, and performance is evaluated using:
#      - **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted scores.
#      - **R-squared (R²)**: Indicates how much variance in the target variable is explained by the model.
# 
# #### Output Interpretation:
# - **XGBoost Model MSE**: 0.0234
#   - This low MSE value indicates that the model predictions are very close to the actual exam scores on average, suggesting high accuracy.
#   
# - **XGBoost Model R-squared**: 0.9999
#   - The R-squared value is nearly 1, showing that the model explains almost all the variance in the exam scores, indicating excellent performance.
# 
# #### Key Takeaways:
# - The XGBoost model performs exceptionally well with very low prediction error and almost perfect R-squared. This suggests that the model has learned the underlying relationships in the data very effectively, and its predictions align closely with the actual scores.
# 

# In[106]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Load the dataset
df = pd.read_csv('medrecords.csv')

# Step 1: Preprocess the data

# One-hot encode categorical variables: 'Gender'
df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Drop the columns that are not required, like 'Exam Age' and 'Age Group'
columns_to_drop = [col for col in df_encoded.columns if 'Exam Age' in col or 'Age Group' in col]
df_cleaned = df_encoded.drop(columns=columns_to_drop)

# Drop the 'Obesity' column since it's redundant with BMI
df_cleaned_no_obesity = df_cleaned.drop(columns=['Obesity'])

# Step 2: Define X (features) and y (target)
# X includes all columns except the target column 'Exam Score'
X = df_cleaned_no_obesity.drop(columns=['Exam Score'])
y = df_cleaned_no_obesity['Exam Score']

# Step 3: Perform train-test split before scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Apply scaling to the numerical columns
scaler = StandardScaler()

# Fit the scaler only on the numerical columns (e.g., Age, BMI)
numerical_columns = ['Age', 'BMI']
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Step 5: Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(random_state=42)

# Step 6: Train the model on the scaled training data
xgb_model.fit(X_train, y_train)

# Step 7: Make predictions on the scaled test data
y_pred_xgb = xgb_model.predict(X_test)

# Step 8: Evaluate the XGBoost model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Step 9: Print the results
print(f'XGBoost Model MSE: {mse_xgb}')
print(f'XGBoost Model R-squared: {r2_xgb}')


# #### **XGBoost Model: Cross-Validation Results**
# 
# #### Code Strategy/Logistics:
# 1. **Cross-Validation Setup**:
#    - The XGBoost model is initialized with a random state for reproducibility.
#    - **5-Fold Cross-Validation** is performed, meaning the dataset is split into 5 subsets, and the model is trained 5 times, each time using a different subset as the test set while the remaining subsets serve as the training set.
#    - The metric used is **Mean Squared Error (MSE)**, but because cross-validation uses negative scoring for consistency, we convert the negative MSE scores back to positive for interpretation.
# 
# 2. **Cross-Validation Output**:
#    - After each fold, the MSE is calculated, and the mean and standard deviation across all folds are reported.
#    - **Mean MSE** provides an average measure of model performance across different subsets of data.
#    - **Standard Deviation** gives insight into how consistent the model performance across the different data splits.
# 
# #### Output Interpretation:
# - **Cross-Validated MSE for each fold**: 
#   - [0.0250, 0.0173, 0.0148, 0.0227, 0.0222]
#   - These values indicate that the model’s performance remains consistently low in error across all 5 folds.
#   
# - **Mean Cross-Validated MSE**: 0.0204
#   - This low average MSE suggests that the model is performing well overall across different data splits.
#   
# - **Standard Deviation of MSE**: 0.0038
#   - The small standard deviation means that the model's performance is very consistent across different folds, showing that it generalizes well to various parts of the data.
# 
# #### Key Takeaways:
# - Cross-validation demonstrates the stability and generalizability of the XGBoost model.
# - The low and consistent MSE values across all folds indicate that the model is not overfitting to any particular subset of the data.
# 

# In[107]:


from sklearn.model_selection import cross_val_score
import xgboost as xgb

# Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(random_state=42)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')

# Convert negative MSE to positive
cv_mse_scores = -cv_scores

# Display the cross-validated MSE for each fold
print(f'Cross-Validated MSE for each fold: {cv_mse_scores}')

# Display the mean and standard deviation of the cross-validated MSE
print(f'Mean Cross-Validated MSE: {cv_mse_scores.mean()}')
print(f'Standard Deviation of MSE: {cv_mse_scores.std()}')


# #### **XGBoost Model: Predictions for the First 5 Rows**
# 
# #### Code Strategy/Logistics:
# 1. **Final Model Training**:
#    - The XGBoost model is trained on the entire dataset (`X` for features and `y` for target variable: Exam Score).
#    - This ensures the model has learned from all available data.
# 
# 2. **First 5 Rows**:
#    - The first 5 rows of the dataset are selected (`X_first_5`), and the model is used to predict their Exam Scores.
#    - The predicted scores are then compared with the actual Exam Scores (`y_true_first_5`).
# 
# 3. **Mean Squared Error (MSE)**:
#    - The MSE is calculated for the first 5 rows, which provides a measure of how well the model predicts these specific cases. The smaller the MSE, the closer the predictions are to the actual values.
# 
# #### Output Interpretation:
# - **Original Exam Scores**: [81, 72, 110, 77, 77]
#   - These are the actual Exam Scores from the dataset for the first 5 rows.
# 
# - **Predicted Exam Scores**: [80.99, 72.02, 109.99, 77.01, 77.02]
#   - The predicted Exam Scores are extremely close to the original values, indicating that the model is making accurate predictions.
# 
# - **MSE for the first 5 rows**: 0.000239
#   - This very small MSE demonstrates that the model has learned the underlying relationships in the data well for these rows, making highly accurate predictions.
# 
# #### Key Takeaway:
# - The XGBoost model performs excellently on the first 5 rows, with minimal error between the predicted and actual Exam Scores, as evidenced by the very low MSE.
# 

# In[108]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Train the final XGBoost model
xgb_model_final = xgb.XGBRegressor(random_state=42)
xgb_model_final.fit(X, y)

# 2. Take the first 5 rows from the dataset
X_first_5 = X[:5]
y_true_first_5 = y[:5]

# 3. Run them through the model to get the predicted Exam Score
y_pred_first_5 = xgb_model_final.predict(X_first_5)

# 4. Compare the predicted values to the original Exam Scores
print("Original Exam Scores:", y_true_first_5.values)
print("Predicted Exam Scores:", y_pred_first_5)

# 5. Calculate the MSE for the first 5 rows
mse_first_5 = mean_squared_error(y_true_first_5, y_pred_first_5)
print(f"MSE for the first 5 rows: {mse_first_5}")


# #### **XGBoost Model: Predictions for the First 1,000 Rows**
# 
# #### Code Strategy/Logistics:
# 1. **First 1,000 Rows Selection**:
#    - The first 1,000 rows from the dataset are selected as `X_first_1000` for features and `y_true_first_1000` for the actual Exam Scores.
#    
# 2. **Prediction**:
#    - The XGBoost model, already trained on the entire dataset, is used to predict the Exam Scores for these first 1,000 rows (`y_pred_first_1000`).
#    - The predicted scores are compared with the original Exam Scores from the dataset.
# 
# 3. **Mean Squared Error (MSE)**:
#    - The MSE for the first 1,000 rows is calculated to measure the difference between the predicted and actual Exam Scores.
#    - A lower MSE indicates better performance.
# 
# #### Output Interpretation:
# - **Original Exam Scores**: For brevity, the output displays the first 10 actual Exam Scores, such as `[81, 72, 110, 77, 77, 85.5, 26.13, 58.91, 47.57, 87]`.
#   
# - **Predicted Exam Scores**: The model's predicted scores for the first 10 rows are displayed, such as `[80.99, 72.02, 109.99, 77.01, 77.02, 85.50, 26.21, 59.47, 47.51, 87]`.
# 
# - **MSE for the first 1,000 rows**: The MSE is `0.01053`, which indicates that the model is making highly accurate predictions, with very little error across the first 1,000 rows.
# 
# #### Key Takeaway:
# - The XGBoost model continues to perform exceptionally well even on a larger subset of the dataset (the first 1,000 rows), as demonstrated by the low MSE.
# - This indicates the model's robustness in predicting Exam Scores over a larger portion of the dataset, maintaining a high level of accuracy.
# 

# In[109]:


# 1. Take the first 1,000 rows from the dataset
X_first_1000 = X[:1000]
y_true_first_1000 = y[:1000]

# 2. Run them through the model to get the predicted Exam Score
y_pred_first_1000 = xgb_model_final.predict(X_first_1000)

# 3. Compare the predicted values to the original Exam Scores
print("Original Exam Scores for the first 1,000 rows:", y_true_first_1000.values[:10])  # Display only first 10 for brevity
print("Predicted Exam Scores for the first 1,000 rows:", y_pred_first_1000[:10])  # Display only first 10 for brevity

# 4. Calculate the MSE for the first 1,000 rows
mse_first_1000 = mean_squared_error(y_true_first_1000, y_pred_first_1000)
print(f"MSE for the first 1,000 rows: {mse_first_1000}")


# #### **XGBoost Model for Predicting Exam Scores**
# 
# #### Code Strategy/Logistics:
# 1. **Data Preprocessing**:
#    - One-hot encoding was applied to convert categorical variables (like Gender) into numerical form.
#    - The 'Exam Age', 'Age Group', and 'Obesity' columns were dropped as they were either redundant or unnecessary for the model.
#    - Feature scaling was applied to numerical variables (Age and BMI) to normalize their range, which is particularly useful for gradient-based methods like XGBoost.
# 
# 2. **Model Training**:
#    - An XGBoost model was trained on the training dataset (80% of the data).
#    - Predictions were made on the test dataset (remaining 20%).
#    
# 3. **Model Evaluation**:
#    - **Mean Squared Error (MSE)** and **R-squared** were calculated for the model's predictions on the test data to measure performance. The model demonstrated excellent accuracy with very low error and an R-squared value close to 1.
#    
# 4. **Cross-Validation**:
#    - The model's performance was evaluated using 5-fold cross-validation. The **Mean Cross-Validated MSE** and its standard deviation were calculated, showing consistency across different splits of the data.
# 
# 5. **Full Dataset Prediction**:
#    - The model was retrained on the full dataset, and predictions were made on specific subsets: 
#       - **First 5 rows** and **first 1,000 rows**: These were used to compare the model's predictions to the original Exam Scores, showing very close alignment between predicted and actual values.
#       - **Entire dataset**: The model's overall MSE for the entire dataset was calculated to assess its performance on all data points.
# 
# #### Output Interpretation:
# - **XGBoost Model MSE**: `0.02338906874154974`
#   - The model’s low MSE indicates it is performing extremely well, with minimal error between predicted and actual Exam Scores on the test set.
#   
# - **XGBoost Model R-squared**: `0.9999245421116839`
#   - The R-squared value suggests the model explains nearly all the variance in the target variable, indicating very strong predictive power.
#   
# - **Cross-Validated MSE for each fold**: `[0.02500604, 0.01728912, 0.01483328, 0.02271097, 0.02217558]`
#   - The cross-validation results show stable performance across different splits of the data, with MSE values in a similar range.
#   
# - **Mean Cross-Validated MSE**: `0.02040299750467838`
#   - The average MSE from cross-validation shows that the model performs consistently well.
#   
# - **Standard Deviation of MSE**: `0.003751659821773931`
#   - The low standard deviation indicates that the model's performance is stable across different folds of the dataset.
#   
# - **First 5 Rows**:
#   - The original Exam Scores for the first 5 rows closely match the predicted Exam Scores, with a very small MSE of `0.00023898585932329296`.
#   
# - **First 1,000 Rows**:
#   - Similarly, for the first 1,000 rows, the MSE is low (`0.010526539734485856`), showing that the model is consistently accurate even on a larger subset.
#   
# - **MSE for the Entire Dataset**: `0.016013743906791843`
#   - The overall MSE for the entire dataset demonstrates that the model continues to perform well across all data points.
#   
# #### Key Takeaways:
# - The XGBoost model performs exceptionally well across both the test set and the entire dataset.
# - Cross-validation results further substantiate the model's robustness, as performance remains consistent across different data splits.
# - The predictions for the first 5 and first 1,000 rows align closely with the actual values, further reinforcing the model's accuracy.
# - The final model's low overall MSE indicates that it generalizes well to the entire dataset, showing no signs of overfitting or underperformance.
# 

# In[110]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Load the dataset
df = pd.read_csv('medrecords.csv')

# Step 1: Preprocess the data

# One-hot encode categorical variables: 'Gender'
df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Drop the columns that are not required, like 'Exam Age' and 'Age Group'
columns_to_drop = [col for col in df_encoded.columns if 'Exam Age' in col or 'Age Group' in col]
df_cleaned = df_encoded.drop(columns=columns_to_drop)

# Drop the 'Obesity' column since it's redundant with BMI
df_cleaned_no_obesity = df_cleaned.drop(columns=['Obesity'])

# Step 2: Define X (features) and y (target)
# X includes all columns except the target column 'Exam Score'
X = df_cleaned_no_obesity.drop(columns=['Exam Score'])
y = df_cleaned_no_obesity['Exam Score']

# Step 3: Perform train-test split before scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Apply scaling to the numerical columns
scaler = StandardScaler()

# Fit the scaler only on the numerical columns (e.g., Age, BMI)
numerical_columns = ['Age', 'BMI']
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Step 5: Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(random_state=42)

# Step 6: Train the model on the scaled training data
xgb_model.fit(X_train, y_train)

# Step 7: Make predictions on the scaled test data
y_pred_xgb = xgb_model.predict(X_test)

# Step 8: Evaluate the XGBoost model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Step 9: Print the results
print(f'XGBoost Model MSE: {mse_xgb}')
print(f'XGBoost Model R-squared: {r2_xgb}')

# Step 10: Perform 5-fold cross-validation
cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')

# Convert negative MSE to positive
cv_mse_scores = -cv_scores

# Display the cross-validated MSE for each fold
print(f'Cross-Validated MSE for each fold: {cv_mse_scores}')

# Display the mean and standard deviation of the cross-validated MSE
print(f'Mean Cross-Validated MSE: {cv_mse_scores.mean()}')
print(f'Standard Deviation of MSE: {cv_mse_scores.std()}')

# Step 11: Train the final XGBoost model on the full dataset
xgb_model_final = xgb.XGBRegressor(random_state=42)
xgb_model_final.fit(X, y)

# Step 12: Take the first 5 rows from the dataset
X_first_5 = X[:5]
y_true_first_5 = y[:5]

# Run them through the model to get the predicted Exam Score
y_pred_first_5 = xgb_model_final.predict(X_first_5)

# Compare the predicted values to the original Exam Scores
print("Original Exam Scores:", y_true_first_5.values)
print("Predicted Exam Scores:", y_pred_first_5)

# Calculate the MSE for the first 5 rows
mse_first_5 = mean_squared_error(y_true_first_5, y_pred_first_5)
print(f"MSE for the first 5 rows: {mse_first_5}")

# Step 13: Take the first 1,000 rows from the dataset
X_first_1000 = X[:1000]
y_true_first_1000 = y[:1000]

# Run them through the model to get the predicted Exam Score
y_pred_first_1000 = xgb_model_final.predict(X_first_1000)

# Compare the predicted values to the original Exam Scores
print("Original Exam Scores for the first 1,000 rows:", y_true_first_1000.values[:10])  # Display only first 10 for brevity
print("Predicted Exam Scores for the first 1,000 rows:", y_pred_first_1000[:10])  # Display only first 10 for brevity

# Calculate the MSE for the first 1,000 rows
mse_first_1000 = mean_squared_error(y_true_first_1000, y_pred_first_1000)
print(f"MSE for the first 1,000 rows: {mse_first_1000}")

# Step 14: Run the model on the entire dataset
y_pred_full = xgb_model_final.predict(X)

# Calculate the MSE for the entire dataset
mse_full = mean_squared_error(y, y_pred_full)
print(f'MSE for the entire dataset: {mse_full}')


# #### **Feature Importance Visualization using XGBoost**
# 
# This code block generates visualizations for feature importance in an XGBoost model based on three different metrics: weight, gain, and cover.
# 
# #### Code Strategy/Logistics:
# 1. **Train the XGBoost Model**:
#    - If not already trained, the final XGBoost model is trained on the entire dataset (`X`, `y`).
# 
# 2. **Plot Feature Importance**:
#    - **Weight**: This shows how frequently a feature is used in the decision trees of the model.
#    - **Gain**: Gain represents the improvement in accuracy brought by a feature to the branches it is used in. Higher gain means more predictive power.
#    - **Cover**: This metric represents the relative number of observations impacted by splits on this feature.
#    
# 3. **Visualizations**:
#    - The feature importance for all three metrics is visualized side-by-side for comparison, using `matplotlib` to generate the plots.
# 
# #### Purpose and Insights:
# - **Weight**: Helps identify which features the model frequently uses for splits. A feature with high weight appears often in the decision trees.
# - **Gain**: Provides insight into which features contribute the most to the model’s performance, showing their impact on prediction accuracy.
# - **Cover**: Shows which features influence the largest number of data points, meaning they play a crucial role in the model’s decision-making process for a broader set of observations.
# 
# By analyzing these visualizations, you can better understand how each feature contributes to the model's overall performance, and assess whether the model might be over-relying on specific features.
# 
# #### Output:
# The output will be three side-by-side plots showing feature importance by:
# - Weight
# - Gain
# - Cover
# 

# In[111]:


import matplotlib.pyplot as plt
import xgboost as xgb

# Step 1: Train the final XGBoost model (if not done already)
xgb_model_final = xgb.XGBRegressor(random_state=42)
xgb_model_final.fit(X, y)

# Step 2: Plot feature importance based on weight, gain, and cover
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Feature importance by weight
xgb.plot_importance(xgb_model_final, importance_type='weight', ax=ax[0], title='Feature importance by Weight')
ax[0].set_title('Feature importance by Weight')

# Feature importance by gain
xgb.plot_importance(xgb_model_final, importance_type='gain', ax=ax[1], title='Feature importance by Gain')
ax[1].set_title('Feature importance by Gain')

# Feature importance by cover
xgb.plot_importance(xgb_model_final, importance_type='cover', ax=ax[2], title='Feature importance by Cover')
ax[2].set_title('Feature importance by Cover')

plt.tight_layout()
plt.show()


# #### **Residual Analysis by Strength Feature**
# 
# This code calculates and compares the mean absolute residuals for two groups in the dataset: individuals with Strength = 1 and Strength = 0.
# 
# #### Code Strategy/Logistics:
# 1. **Calculate Predicted Exam Scores**: 
#    - The final XGBoost model is used to predict exam scores, which are stored in a new column (`'Predicted Exam Score'`).
#    
# 2. **Calculate Residuals**:
#    - The residuals are calculated by subtracting the predicted scores from the actual exam scores (`'Residuals'` = `'Exam Score'` - `'Predicted Exam Score'`).
#    
# 3. **Separate Residuals by Strength**:
#    - The residuals are divided into two groups based on the `Strength` feature: those where Strength = 1 and those where Strength = 0.
# 
# 4. **Mean Absolute Residuals**:
#    - The mean absolute residuals for both groups are calculated and displayed. This helps us understand how well the model performs when Strength is present or absent.
# 
# #### Interpretation of the Output:
# - **Mean Absolute Residuals for Strength = 1**: 0.0047
#    - The model performs very well when Strength = 1, as the mean absolute residual is quite low, indicating minimal error.
#    
# - **Mean Absolute Residuals for Strength = 0**: 0.0613
#    - The model has a higher error when Strength = 0, showing that the predictions are less accurate in the absence of Strength.
# 
# This analysis suggests that the model may be disproportionately reliant on the Strength feature, as the error is significantly smaller when Strength is present.
# 

# In[112]:


# Calculate residuals for the dataset
df['Predicted Exam Score'] = xgb_model_final.predict(X)
df['Residuals'] = df['Exam Score'] - df['Predicted Exam Score']

# Separate residuals by Strength
strength_positive_residuals = df[df['Strength'] == 1]['Residuals']
strength_negative_residuals = df[df['Strength'] == 0]['Residuals']

# Calculate the mean absolute residuals for each group
strength_positive_residuals_mean = strength_positive_residuals.abs().mean()
strength_negative_residuals_mean = strength_negative_residuals.abs().mean()

print(f"Mean Absolute Residuals for Strength = 1: {strength_positive_residuals_mean:.4f}")
print(f"Mean Absolute Residuals for Strength = 0: {strength_negative_residuals_mean:.4f}")


# #### **XGBoost Model Evaluation Without the 'Strength' Feature**
# 
# #### Code Strategy/Logistics:
# 1. **Preprocessing**: 
#    - The dataset is cleaned by one-hot encoding categorical variables like Gender, and columns like 'Exam Age', 'Age Group', and 'Obesity' are dropped. 
#    - Additionally, the 'Strength' feature is specifically removed from the dataset for this analysis.
#    
# 2. **Scaling**: 
#    - The numerical features ('Age', 'BMI') are scaled to ensure they are normalized before model training.
# 
# 3. **Model Training**: 
#    - An XGBoost model (`xgb_model_no_strength`) is trained using the cleaned dataset without the 'Strength' feature.
#    
# 4. **Evaluation**: 
#    - The model is evaluated on the test data, and the Mean Squared Error (MSE) and R-squared are calculated to assess performance.
# 
# #### Interpretation of the Results:
# - **MSE without Strength**: 26.9884
#    - The Mean Squared Error increases significantly when the 'Strength' feature is removed, indicating that 'Strength' is a critical feature for accurate predictions.
#    
# - **R-squared without Strength**: 0.9129
#    - The R-squared value decreases to 0.9129, showing that the model explains less variance in the target variable without the 'Strength' feature.
# 
# These results suggest that the model relies heavily on the 'Strength' feature for making accurate predictions.
# 

# In[113]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Load the dataset
df = pd.read_csv('medrecords.csv')

# Step 1: Preprocess the data

# One-hot encode categorical variables: 'Gender'
df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Drop the columns that are not required, like 'Exam Age' and 'Age Group'
columns_to_drop = [col for col in df_encoded.columns if 'Exam Age' in col or 'Age Group' in col]
df_cleaned = df_encoded.drop(columns=columns_to_drop)

# **NEW STEP**: Drop the 'Strength' feature from the dataset
df_cleaned_no_strength = df_cleaned.drop(columns=['Strength'])

# Drop the 'Obesity' column since it's redundant with BMI
df_cleaned_no_obesity = df_cleaned_no_strength.drop(columns=['Obesity'])

# Step 2: Define X (features) and y (target)
# X includes all columns except the target column 'Exam Score'
X = df_cleaned_no_obesity.drop(columns=['Exam Score'])
y = df_cleaned_no_obesity['Exam Score']

# Step 3: Perform train-test split before scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Apply scaling to the numerical columns
scaler = StandardScaler()

# Fit the scaler only on the numerical columns (e.g., Age, BMI)
numerical_columns = ['Age', 'BMI']
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Step 5: Initialize the XGBoost model
xgb_model_no_strength = xgb.XGBRegressor(random_state=42)

# Step 6: Train the model on the scaled training data
xgb_model_no_strength.fit(X_train, y_train)

# Step 7: Make predictions on the scaled test data
y_pred_xgb_no_strength = xgb_model_no_strength.predict(X_test)

# Step 8: Evaluate the XGBoost model without the 'Strength' feature
mse_xgb_no_strength = mean_squared_error(y_test, y_pred_xgb_no_strength)
r2_xgb_no_strength = r2_score(y_test, y_pred_xgb_no_strength)

# Step 9: Print the results without 'Strength'
print(f'XGBoost Model MSE (without Strength): {mse_xgb_no_strength}')
print(f'XGBoost Model R-squared (without Strength): {r2_xgb_no_strength}')


# #### **XGBoost Model Evaluation Without the 'Age' Feature**
# 
# #### Code Strategy/Logistics:
# 1. **Preprocessing**:
#    - The dataset is cleaned by one-hot encoding the 'Gender' variable and dropping irrelevant columns like 'Exam Age', 'Age Group', 'Obesity', and 'Age'.
#    
# 2. **Scaling**:
#    - The numerical column 'BMI' is scaled using `StandardScaler` to normalize the values.
# 
# 3. **Model Training**:
#    - An XGBoost model is trained using the cleaned dataset with the 'Age' feature removed.
# 
# 4. **Evaluation**:
#    - The model is evaluated on the test data using Mean Squared Error (MSE) and R-squared to assess performance.
#    - Cross-validation is performed to check model robustness.
# 
# #### Interpretation of Results:
# - **MSE without Age**: 16.0124
#    - The Mean Squared Error is noticeably higher compared to when 'Age' is included, indicating that 'Age' is an important feature for predicting exam scores.
#    
# - **R-squared without Age**: 0.9483
#    - The R-squared value remains high, suggesting that the model still explains a large portion of the variance in the target variable, though not as effectively as with 'Age'.
#    
# - **Cross-Validated MSE**: 
#    - Mean: 16.0782
#    - Standard Deviation: 0.1003
#    - The cross-validated MSE across different folds shows consistency, with a small variation across the folds, indicating a stable model performance without 'Age'.
# 
# These results highlight that 'Age' is a valuable feature for making accurate predictions, though the model still performs reasonably well without it.
# 

# In[114]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Load the dataset
df = pd.read_csv('medrecords.csv')

# Step 1: Preprocess the data

# One-hot encode categorical variables: 'Gender'
df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Drop the columns that are not required, like 'Exam Age' and 'Age Group'
columns_to_drop = [col for col in df_encoded.columns if 'Exam Age' in col or 'Age Group' in col]
df_cleaned = df_encoded.drop(columns=columns_to_drop)

# Drop the 'Obesity' column since it's redundant with BMI
df_cleaned_no_obesity = df_cleaned.drop(columns=['Obesity'])

# Drop 'Age' but keep 'Strength'
df_cleaned_no_age = df_cleaned_no_obesity.drop(columns=['Age'])

# Step 2: Define X (features) and y (target)
# X includes all columns except the target column 'Exam Score'
X = df_cleaned_no_age.drop(columns=['Exam Score'])
y = df_cleaned_no_age['Exam Score']

# Step 3: Perform train-test split before scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Apply scaling to the numerical columns
scaler = StandardScaler()

# Fit the scaler only on the numerical columns (e.g., BMI)
numerical_columns = ['BMI']
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Step 5: Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(random_state=42)

# Step 6: Train the model on the scaled training data
xgb_model.fit(X_train, y_train)

# Step 7: Make predictions on the scaled test data
y_pred_xgb = xgb_model.predict(X_test)

# Step 8: Evaluate the XGBoost model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Step 9: Print the results
print(f'XGBoost Model MSE (without Age): {mse_xgb}')
print(f'XGBoost Model R-squared (without Age): {r2_xgb}')

# Step 10: Cross-validation for robustness
cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_mse_scores = -cv_scores

# Display cross-validated MSE
print(f'Cross-Validated MSE for each fold (without Age): {cv_mse_scores}')
print(f'Mean Cross-Validated MSE (without Age): {cv_mse_scores.mean()}')
print(f'Standard Deviation of MSE (without Age): {cv_mse_scores.std()}')


# #### **Feature Importance Evaluation by Removing Features**
# 
# #### Code Strategy:
# 1. **Preprocessing**:
#    - The dataset is cleaned by one-hot encoding the 'Gender' variable and dropping irrelevant columns like 'Exam Age', 'Age Group', and 'Obesity'.
#    
# 2. **Scaling**:
#    - The numerical columns 'Age' and 'BMI' are scaled using `StandardScaler` to normalize the values.
# 
# 3. **Feature Removal Process**:
#    - For each feature in the dataset, we iteratively remove it, retrain the XGBoost model, and evaluate the model's performance using Mean Squared Error (MSE) and R-squared.
#    
# 4. **Model Reintroduction**:
#    - After evaluating feature removal, the model is retrained with all features included to get the final baseline metrics.
# 
# #### Interpretation of Results:
# The table below shows the MSE and R-squared values when each feature is removed from the dataset.
# 
# | Removed Feature    | MSE         | R-squared  |
# |--------------------|-------------|------------|
# | Age                | 16.012358   | 0.948341   |
# | BMI                | 0.016445    | 0.999947   |
# | Smoking            | 0.022409    | 0.999928   |
# | High Alcohol       | 0.023304    | 0.999925   |
# | Heart Disease      | 3.235148    | 0.989563   |
# | Cancer             | 6.739839    | 0.978256   |
# | COPD               | 0.979488    | 0.996840   |
# | Alzheimers         | 9.763297    | 0.968502   |
# | Diabetes           | 2.478050    | 0.992005   |
# | CKD                | 3.802506    | 0.987732   |
# | High Blood Pressure| 1.155466    | 0.996272   |
# | Stroke             | 10.138606   | 0.967291   |
# | Liver Dx           | 5.273491    | 0.982987   |
# | Strength           | 26.988428   | 0.912930   |
# | Gender_Male        | 0.023107    | 0.999925   |
# 
# - **Age**: Removing the 'Age' feature results in a significant increase in MSE (16.01), indicating that 'Age' is a crucial feature.
# - **Strength**: Removing the 'Strength' feature drastically increases MSE (26.99), indicating heavy reliance on this feature.
# - **BMI**: Despite being an important feature, removing BMI has only a minor impact on model performance.
# 
# #### Final Model Performance with All Features:
# - **MSE**: 0.0234
# - **R-squared**: 0.9999
# 
# These results show that the model performs exceptionally well when all features are included and demonstrate the importance of features such as Age and Strength.
# 

# #### **Removing the 5 Lowest-Performing Features from the Model**
# 
# We are evaluating how the XGBoost model performs when the 5 least important features are removed:
# 
# 1. **Features Removed**:
#     - High Alcohol
#     - Gender_Male
#     - Smoking
#     - BMI
#     - High Blood Pressure
# 
# 2. **Steps**:
#     - The dataset is preprocessed by one-hot encoding categorical variables and dropping irrelevant columns like 'Exam Age' and 'Age Group'.
#     - We then remove the five least important features from the dataset.
#     - The model is trained using XGBoost on the reduced dataset, and the Mean Squared Error (MSE) and R-squared (R²) values are calculated.
# 
# 3. **Results**:
#     - **XGBoost Model MSE (without the 5 lowest-performing features)**: 2.239843848851294
#     - **XGBoost Model R-squared (without the 5 lowest-performing features)**: 0.9927738086171873
# 
# This analysis helps assess how much impact removing the least important features has on the overall performance of the model.
# 

# In[115]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Load the dataset
df = pd.read_csv('medrecords.csv')

# Step 1: Preprocess the data

# One-hot encode categorical variables: 'Gender'
df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Drop the columns that are not required, like 'Exam Age' and 'Age Group'
columns_to_drop = [col for col in df_encoded.columns if 'Exam Age' in col or 'Age Group' in col]
df_cleaned = df_encoded.drop(columns=columns_to_drop)

# Drop the 'Obesity' column since it's redundant with BMI
df_cleaned_no_obesity = df_cleaned.drop(columns=['Obesity'])

# Step 2: Define X (features) and y (target)
# X includes all columns except the target column 'Exam Score'
X = df_cleaned_no_obesity.drop(columns=['Exam Score'])

# Step 3: Remove the 5 least important features
features_to_remove = ['High Alcohol', 'Gender_Male', 'Smoking', 'BMI', 'High Blood Pressure']
X_reduced = X.drop(columns=features_to_remove)

# Step 4: Define y (target)
y = df_cleaned_no_obesity['Exam Score']

# Step 5: Perform train-test split before scaling
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Step 6: Apply scaling to the numerical columns
scaler = StandardScaler()
numerical_columns = ['Age']  # Only scale 'Age' since 'BMI' was removed
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Step 7: Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(random_state=42)

# Step 8: Train the model on the scaled training data
xgb_model.fit(X_train, y_train)

# Step 9: Make predictions on the scaled test data
y_pred_xgb = xgb_model.predict(X_test)

# Step 10: Evaluate the XGBoost model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Step 11: Print the results
print(f'XGBoost Model MSE (without the 5 lowest-performing features): {mse_xgb}')
print(f'XGBoost Model R-squared (without the 5 lowest-performing features): {r2_xgb}')


# #### **Removing the 8 Lowest-Performing Features from the Model**
# 
# In this step, we are evaluating the XGBoost model's performance after removing the 8 least important features:
# 
# 1. **Features Removed**:
#     - High Alcohol
#     - Gender_Male
#     - Smoking
#     - BMI
#     - High Blood Pressure
#     - Heart Disease
#     - Cancer
#     - Stroke
# 
# 2. **Steps**:
#     - The dataset is preprocessed by one-hot encoding categorical variables and dropping irrelevant columns like 'Exam Age' and 'Age Group'.
#     - We drop the 8 lowest-performing features from the dataset.
#     - The model is trained using XGBoost on the reduced dataset, and performance is evaluated using Mean Squared Error (MSE) and R-squared (R²).
# 
# 3. **Results**:
#     - **XGBoost Model MSE (without the 8 lowest-performing features)**: 0.02250054506846148
#     - **XGBoost Model R-squared (without the 8 lowest-performing features)**: 0.9999274086695974
# 
# This analysis demonstrates that even after removing the 8 least important features, the model maintains excellent performance.
# 

# In[116]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Load the dataset
df = pd.read_csv('medrecords.csv')

# Step 1: Preprocess the data

# One-hot encode categorical variables: 'Gender'
df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Drop the columns that are not required, like 'Exam Age' and 'Age Group'
columns_to_drop = [col for col in df_encoded.columns if 'Exam Age' in col or 'Age Group' in col]
df_cleaned = df_encoded.drop(columns=columns_to_drop)

# Drop the 'Obesity' column since it's redundant with BMI
df_cleaned_no_obesity = df_cleaned.drop(columns=['Obesity'])

# Drop the 3 additional lowest-performing features: 'Gender_Male', 'High Alcohol', and 'Smoking'
df_cleaned_reduced = df_cleaned_no_obesity.drop(columns=['Gender_Male', 'High Alcohol', 'Smoking'])

# Step 2: Define X (features) and y (target)
X = df_cleaned_reduced.drop(columns=['Exam Score'])
y = df_cleaned_reduced['Exam Score']

# Step 3: Perform train-test split before scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Apply scaling to the numerical columns
scaler = StandardScaler()

# Fit the scaler only on the numerical columns (e.g., Age, BMI)
numerical_columns = ['Age', 'BMI']
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Step 5: Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(random_state=42)

# Step 6: Train the model on the scaled training data
xgb_model.fit(X_train, y_train)

# Step 7: Make predictions on the scaled test data
y_pred_xgb = xgb_model.predict(X_test)

# Step 8: Evaluate the XGBoost model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Step 9: Print the results
print(f'XGBoost Model MSE (without the 8 lowest-performing features): {mse_xgb}')
print(f'XGBoost Model R-squared (without the 8 lowest-performing features): {r2_xgb}')


# #### **Removing 3 More Lowest-Performing Features from the Model**
# 
# In this step, we are evaluating the XGBoost model's performance after removing 3 additional low-performing features:
# 
# 1. **Features Removed**:
#     - Heart Disease
#     - Alzheimer's
#     - Diabetes
# 
# 2. **Steps**:
#     - The dataset is preprocessed, with one-hot encoding of categorical variables and scaling applied to numerical columns (Age, BMI).
#     - The 3 lowest-performing features are removed from the dataset.
#     - The XGBoost model is trained on the reduced dataset, and its performance is evaluated using Mean Squared Error (MSE) and R-squared (R²).
# 
# 3. **Results**:
#     - **XGBoost Model MSE (after removing 3 more features)**: 17.059005215961303
#     - **XGBoost Model R-squared (after removing 3 more features)**: 0.9449641828584806
# 
# This step highlights that removing these additional features results in a noticeable decline in the model's performance, suggesting that they contribute meaningfully to accurate predictions.
# 

# In[117]:


# Step 1: Remove 3 more lowest-performing features
features_to_drop = ['Heart Disease', 'Alzheimers', 'Diabetes']
X_selected = X.drop(columns=features_to_drop)

# Step 2: Perform train-test split before scaling
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Step 3: Apply scaling to the remaining numerical columns
scaler = StandardScaler()
numerical_columns = ['Age', 'BMI']
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Step 4: Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(random_state=42)

# Step 5: Train the model on the scaled training data
xgb_model.fit(X_train, y_train)

# Step 6: Make predictions on the scaled test data
y_pred_xgb = xgb_model.predict(X_test)

# Step 7: Evaluate the XGBoost model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Step 8: Print the results
print(f'XGBoost Model MSE (after removing 3 more features): {mse_xgb}')
print(f'XGBoost Model R-squared (after removing 3 more features): {r2_xgb}')


# #### **Feature Importance and Step-by-Step Removal**
# 
# In this step, we evaluate the performance of the XGBoost model as we systematically remove features in order of their importance, starting with the least important.
# 
# ## Steps:
# 1. **Initial Model**: Train the XGBoost model on all available features and record the performance.
# 2. **Feature Removal**: Begin removing features, starting with the lowest-importance ones, and track the impact on model performance after each removal.
# 3. **Performance Metrics**:
#     - **MSE**: Mean Squared Error
#     - **R-squared (R²)**: A measure of the proportion of variance explained by the model.
# 
# ## Results:
# 
# | Removed Feature         | MSE        | R-squared |
# |-------------------------|------------|-----------|
# | None                    | 0.023389   | 0.999925  |
# | Gender_Male             | 0.023107   | 0.999925  |
# | High Alcohol            | 0.020897   | 0.999933  |
# | Smoking                 | 0.022501   | 0.999927  |
# | BMI                     | 0.012337   | 0.999960  |
# | High Blood Pressure     | 2.239844   | 0.992774  |
# | Liver Dx                | 8.891483   | 0.971314  |
# | COPD                    | 16.913325  | 0.945434  |
# | Diabetes                | 23.187607  | 0.925192  |
# | CKD                     | 34.150340  | 0.889824  |
# | Stroke                  | 46.973287  | 0.848455  |
# | Heart Disease           | 55.038160  | 0.822436  |
# | Alzheimers              | 65.010613  | 0.790263  |
# | Cancer                  | 71.571075  | 0.769097  |
# | Strength                | 121.632734 | 0.607588  |
# 
# #### **Conclusion**:
# - As we removed features like **Gender_Male** and **High Alcohol**, the model's performance remained nearly perfect.
# - However, removing more critical features such as **Heart Disease**, **Stroke**, and **Strength** caused the model's performance to degrade significantly, showing that these features are essential for accurate predictions.
# 

# In[118]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Load the dataset
df = pd.read_csv('medrecords.csv')

# Step 1: Preprocess the data
df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Drop irrelevant columns and 'Obesity'
df_cleaned = df_encoded.drop(columns=['Obesity', 'Age Group', 'Exam Age'])

# Define X (features) and y (target)
X = df_cleaned.drop(columns=['Exam Score'])
y = df_cleaned['Exam Score']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features
scaler = StandardScaler()
numerical_columns = ['Age', 'BMI']
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Function to train XGBoost and return MSE and R-squared
def train_xgboost(X_train, y_train, X_test, y_test):
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    return mse_xgb, r2_xgb

# Record the initial performance with all features
results = []
mse, r2 = train_xgboost(X_train, y_train, X_test, y_test)
results.append({'Removed Feature': 'None', 'MSE': mse, 'R-squared': r2})

# Now systematically remove the lowest importance features and track performance
features_sorted_by_importance = ['Gender_Male', 'High Alcohol', 'Smoking', 'BMI', 
                                 'High Blood Pressure', 'Liver Dx', 'COPD', 'Diabetes', 
                                 'CKD', 'Stroke', 'Heart Disease', 'Alzheimers', 
                                 'Cancer', 'Strength', 'Age']

# Step-by-step feature removal
remaining_features = X_train.columns.tolist()

for feature in features_sorted_by_importance:
    if feature in remaining_features:
        remaining_features.remove(feature)
        
        # Ensure we don't run into an empty feature set
        if not remaining_features:
            print(f"Stopping as no more features are available to remove.")
            break
        
        # Subset data to remaining features
        X_train_dropped = X_train[remaining_features]
        X_test_dropped = X_test[remaining_features]
        
        # Train and evaluate
        mse, r2 = train_xgboost(X_train_dropped, y_train, X_test_dropped, y_test)
        
        # Record the results
        results.append({'Removed Feature': feature, 'MSE': mse, 'R-squared': r2})

# Convert results to a DataFrame for easy visualization
results_df = pd.DataFrame(results)
print(results_df)


# ### **Key Insights and Analysis**:
# 
# - **Strength**: The project reveals that Strength is one of the most important features. Removing it significantly increases the MSE, showing that it has a strong influence on the model’s predictive capability.
# 
# - **Heart Disease**, **Alzheimer's**, and **Cancer**: These features are also highly impactful. Removing them leads to a noticeable performance drop, reflecting their importance in predicting the Exam Score.
# 
# - **BMI**, **Smoking**, and **High Blood Pressure**: These features, while included in the model, seem to have a relatively smaller impact on the model’s performance. Removing them does not substantially affect the model’s accuracy, suggesting that they may be less critical for accurate predictions.
# 
# ---
# 
# ### **Mathematical and Logical Comparison**:
# 
# Several metrics are used to assess how well the machine learning model learns the algorithm:
# 
# 1. **Mean Squared Error (MSE)**:  
#    Measures the average squared difference between the predicted and actual exam scores. A low MSE indicates that the model is successfully learning the underlying relationships.
# 
# 2. **R-squared (R²)**:  
#    Measures the proportion of variance in the Exam Score that the model explains using the features. An R² close to 1 suggests that the model is explaining most of the variance and performing well.
# 
# 3. **Feature Importance Scores**:  
#    By evaluating how MSE and R² change as features are removed, we can pinpoint which features the model relies on most. If removing a feature causes a significant drop in performance, the feature is considered crucial.
# 
# ---
# 
# ### **Explanation of Key Concepts**:
# 
# - **Overfitting**:  
#   This refers to when a model learns to memorize the noise or specific patterns in the training data that do not generalize well to new, unseen data. Overfitting occurs when the model becomes too complex and too reliant on the training data, making it perform worse on new data. The process of iterative feature removal helps in simplifying the model and avoiding overfitting by retaining only the most valuable features.
# 
# - **Generalization**:  
#   This is the model's ability to perform well on new, unseen data. By removing unnecessary features and focusing only on the most important ones, we encourage the model to generalize better, reducing the risk of overfitting.
# 
# ---
# 
# ### **Conclusion**:
# 
# This project demonstrates the effectiveness of using **iterative feature removal** to analyze which features the XGBoost model relies on the most. Through the feature importance analysis, we can evaluate whether the model is learning the correct relationships or if it is overfitting to certain variables.
# 
# - The **original full model** shows **very high accuracy** (MSE and R-squared scores), which indicates that the model is doing a good job of learning the algorithm. 
# - However, by removing features iteratively and seeing how the model’s performance changes, we can ensure that the model isn’t overly dependent on specific features.
# 
# #### **Does this substantiate the original model’s high accuracy?**
# Yes, the iterative removal process proves that the model isn’t simply memorizing patterns. The results show that even after removing certain less important features, the model still performs well, substantiating that the original model is making accurate predictions by relying on a strong set of features. However, removing key features like **Strength** or **Heart Disease** does significantly degrade the model’s performance, confirming their critical role in accurate prediction.
# 
# This analysis ensures that the model is correctly generalizing and learning the algorithm-driven relationships without being overly dependent on any single feature, thereby reinforcing the accuracy of the full model’s high metrics.
# 

# In[119]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Load the dataset
df = pd.read_csv('medrecords.csv')

# Step 1: Preprocess the data
df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)
df_cleaned = df_encoded.drop(columns=['Obesity', 'Age Group', 'Exam Age'])

# Define X (features) and y (target)
X = df_cleaned.drop(columns=['Exam Score'])
y = df_cleaned['Exam Score']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features
scaler = StandardScaler()
numerical_columns = ['Age', 'BMI']
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Function to train XGBoost and return MSE and R-squared
def train_xgboost(X_train, y_train, X_test, y_test):
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    return mse_xgb, r2_xgb

# Record the initial performance with all features
results = []
mse, r2 = train_xgboost(X_train, y_train, X_test, y_test)
results.append({'Removed Feature': 'None', 'MSE': mse, 'R-squared': r2})

# Initialize list of remaining features
remaining_features = X.columns.tolist()

# Step-by-step feature removal
features_sorted_by_importance = ['Gender_Male', 'High Alcohol', 'Smoking', 'BMI', 
                                 'High Blood Pressure', 'Liver Dx', 'COPD', 'Diabetes', 
                                 'CKD', 'Stroke', 'Heart Disease', 'Alzheimers', 
                                 'Cancer', 'Strength', 'Age']

# Track features as they are removed
for feature in features_sorted_by_importance:
    if feature in remaining_features:
        remaining_features.remove(feature)
        
        # Ensure we don't run into an empty feature set
        if len(remaining_features) == 0:
            print(f"Stopping as no more features are available to remove.")
            break
        
        # Subset data to remaining features
        X_train_reduced = X_train[remaining_features]
        X_test_reduced = X_test[remaining_features]

        # Train and evaluate, only if remaining features exist
        if len(remaining_features) > 0:
            mse, r2 = train_xgboost(X_train_reduced, y_train, X_test_reduced, y_test)
            
            # Record the results
            results.append({'Removed Feature': feature, 'MSE': mse, 'R-squared': r2})

# Convert results to a DataFrame for easy visualization
results_df = pd.DataFrame(results)
print(results_df)


# In[ ]:




