# Singapore-Resale-Flat-Prices-Prediction
This project is to develop a machine-learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.

# Housing and Development Board - An Overview

The Housing & Development Board (HDB; often referred to as the Housing Board), is a statutory board under the Ministry of National Development responsible for the public housing in Singapore. Established in 1960 as a result of efforts in the late 1950s to set up an authority to take over the Singapore Improvement Trust's (SIT) public housing responsibilities, the HDB focused on the construction of emergency housing and the resettlement of kampong residents into public housing in the first few years of its existence.

# Problem Statement:

The objective of this project is to develop a machine-learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.

# Approach:

**1.Data Wrangling:**

It involves the following steps:

  * Data Cleaning
  * Data Transforamtion
  * Data Enrichment

**Data Cleaning:**

  * Handle missing values with mean/median/mode.
  * Treat Outliers using IQR
  * Identify Skewness in the dataset and treat skewness with appropriate data transformations,
    such as log transformation.
  * Encode categorical variables using suitable techniques, such as one-hot encoding, label encoding,
    or ordinal encoding, based on their nature and relationship with the target variable.

**Data transformation:**

  *  Changing the structure or format of data, which can include normalizing data, scaling features, or encoding categorical variables.

**Exploaratory Data Analysis:**

  * Try visualizing outliers and skewness(before and after treating skewness using Seaborn’s boxplot.

**Feature Engineering:**

  * Engineer new features if applicable, such as aggregating or transforming existing features to create more informative representations of the data.

**Model Building and Evaluation:**

* Split the dataset into training and testing/validation sets.
* Train the different regression models and evaluate the result with suitable metrics such as MAE - Mean Absolute Error, MSE - Mean Squared Error and
  RMSE - Root Mean Squared Error.

**Model Deployment using Streamlit:**
* Develop interactive GUI using streamlit.
* Task input( Regression )
* create an input field where the user can enter each column value except resale price.




