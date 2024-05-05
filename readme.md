## Credit Card Fraud Detection with Machine Learning

This Jupyter Notebook explores machine learning techniques for credit card fraud detection. The focus is on identifying fraudulent transactions from historical data.  

**Key Techniques:**

* **Data Preparation:**
    * Feature Selection (dropping irrelevant features)
    * Handling Missing Values 
    * Addressing Class Imbalance (undersampling with RandomUnderSampler)
    * Train-Test Splitting
* **Machine Learning Models:**
    * Multi-Layer Perceptron (MLP) Classifier
    * Autoencoder for Anomaly Detection with Logistic Regression Classification

**Notebook Structure:**

1. **Data Loading and Exploration:**
    * Loads the credit card transaction dataset.
    * Explores data characteristics (shape, missing values, etc.)

2. **Data Preprocessing:**
    * Selects relevant features for fraud prediction.
    * Handles missing values using median imputation.
    * Addresses class imbalance (many more normal transactions than fraudulent ones) using random undersampling.
    * Splits data into training and testing sets.

3. **Multi-Layer Perceptron (MLP) Classifier:**
    * Implements an MLP classifier with a single hidden layer for fraud detection.
    * Evaluates model performance using accuracy score.

4. **Autoencoder for Anomaly Detection:**
    * Builds an autoencoder, a neural network that learns compressed representations of data suitable for anomaly detection.
    * Trains the autoencoder on normal transactions, assuming fraudulent transactions will deviate significantly from the learned patterns.
    * Extracts hidden representations from both normal and fraudulent transactions.
    * Trains a Logistic Regression classifier on the hidden representations to distinguish between normal and fraudulent transactions.
    * Evaluates the Logistic Regression model's performance using accuracy score.

5. **Conclusion:**
    * Briefly summarizes the findings and potential improvements.

**Running the Notebook**

1. Make sure you have the required libraries installed (`pandas`, `numpy`, `scikit-learn`, `tensorflow`, `imblearn`, `seaborn`, etc.). You can install them using `pip install <library_name>`.
2. Install NannyML using:
`pip install -U nannyml`
or run `!python -m pip install git+https://github.com/NannyML/nannyml` in a Jupyter cell.
3. Download the credit card transaction dataset:
[creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)
4. Open the Jupyter Notebook and run the cells one by one.

**Further Exploration**

* Try different hyperparameter tuning techniques to improve model performance.
* Implement cost-sensitive learning to emphasize the importance of correctly classifying fraudulent transactions.
