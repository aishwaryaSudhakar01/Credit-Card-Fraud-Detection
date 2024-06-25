# Comparative Study of Machine Learning Algorithms for Credit Card Fraud Detection

# Objective
The primary objective of this project is to develop and evaluate various machine-learning models for detecting fraudulent credit card transactions. This project involves a comparative study of model performance on both imbalanced and balanced datasets, showcasing my expertise in handling class imbalance, data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and visualization.

# Methodology
Data Preprocessing
Data Loading: The dataset was loaded using pandas.
Missing Values: Checked for missing values to ensure data quality.
Data Splitting: Split the dataset into training and testing sets using train_test_split from scikit-learn.

# Exploratory Data Analysis (EDA)
Data Exploration: Inspected the first ten rows, data types, and summary statistics of the dataset.
Class Distribution: Found that the dataset is highly imbalanced with the majority of transactions being legitimate.
Legitimate transactions: 99.83%
Fraudulent transactions: 0.17%
Descriptive Statistics: Generated summary statistics for both classes.
Visualization: Created count plots and scatter plots to visualize the distribution and relationships between features.

# Handling Class Imbalance
Imbalanced Dataset: Initially trained models on the original imbalanced dataset.
Balanced Dataset: Performed random undersampling to balance the dataset, reducing the number of legitimate transactions to match the number of fraudulent transactions.

# Feature Engineering
Feature Selection: Removed the target variable 'Class' from the feature set.
Dimensionality Reduction: Ensured relevant features were included to improve model performance.

# Model Training and Evaluation
Trained and evaluated the following models on both imbalanced and balanced datasets:
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Support Vector Machine (SVM)
Naive Bayes

Evaluation Metrics: Used classification reports, accuracy scores, precision, recall, F1 scores, ROC curves, and AUC to evaluate model performance.

# Comparative Analysis
Imbalanced Dataset: Models generally achieved high accuracy but struggled with recall for class 1, indicating difficulty in correctly identifying all fraudulent transactions.
Balanced Dataset: Models showed improved recall and F1 scores for class 1, demonstrating better overall performance in detecting fraudulent transactions.
The Random Forest Classifier consistently outperformed other models on both datasets, making it the most reliable model. It achieved an accuracy of 95.93%, precision of 0.98, recall of 0.94, and an F1 score of 0.96 on the balanced dataset.

# Skills Demonstrated
Python Programming
Data manipulation with pandas
Data visualization with seaborn and matplotlib
Machine learning with scikit-learn
Handling class imbalance
Model evaluation and selection
Repository Contents
ml-based-credit-card-fraud-detection.ipynb: Jupyter notebook containing the complete analysis and model development.
creditcard.csv: Dataset used for the project (provide a link or instruction to download if applicable).
Acknowledgements
The dataset used in this project is sourced from Kaggle and consists of transactions made by credit cards in September 2013 by European cardholders.

Feel free to explore the notebook and provide any feedback or suggestions!
