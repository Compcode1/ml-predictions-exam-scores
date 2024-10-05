#### **Introduction**

The goal of this project was to develop a machine learning model capable of accurately predicting exam scores based on various medical diagnoses and lifestyle factors. Two primary models were tested: Random Forest and XGBoost, with a focus on analyzing model performance and understanding the importance of different features in predicting exam scores.

- **Random Forest**: This model works by constructing multiple decision trees during training and outputs the average prediction of the individual trees, reducing overfitting and improving accuracy by combining the results of many weak learners into a strong one.

- **XGBoost**: A gradient boosting algorithm, XGBoost builds an ensemble of trees sequentially, where each tree attempts to correct the errors of the previous ones. It is highly optimized for both speed and performance and excels in handling imbalanced and large datasets.

The dataset, `medrecords.csv`, was preprocessed to handle categorical variables, missing values, and standardized features such as BMI and Age. The project incorporated various steps, including model training, evaluation, cross-validation, and experimentation with iterative feature removal to assess model robustness and identify key variables.

This experimentation provided insights into feature importance, model generalization under varying conditions, and the potential for over-reliance on specific features. For example, when key variables like "Strength" or "Age" were removed in the XGBoost model, we observed a sharp decline in performance, indicating the model's sensitivity to those features. 

#### Model Sensitivity and Feature Importance

This process helped to identify how certain features significantly impacted the model's predictive accuracy. For instance, removing "Strength" or "Age" from the dataset led to a noticeable drop in performance metrics like **Mean Squared Error (MSE)** and **R-squared**, showing that these features were essential to the model’s predictions. Without these key variables, XGBoost's ability to make accurate predictions was diminished, suggesting that these variables captured vital relationships in the data that the model depended on.

However, the removal of less significant features like "BMI" or "High Alcohol" did not substantially affect the model’s performance, demonstrating that the model was not overfitting to irrelevant features. This iterative feature removal approach ensured that the model was focusing on meaningful patterns in the data rather than noise, allowing for better generalization to new, unseen data. By systematically testing the model’s robustness under various feature configurations, the project confirmed which features were truly necessary for maintaining high predictive accuracy, thereby avoiding over-reliance on any single feature. 

The analysis highlighted the critical variables for the model's success and demonstrated how performance metrics evolved as we removed or retained certain features, reinforcing the model's ability to generalize well across different subsets of data.
