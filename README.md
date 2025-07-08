# **Sales Forecasting Using Linear Regression**

This project demonstrates how to build a **Sales Forecasting Model** using **Linear Regression** in Python. The dataset includes daily sales data for various products. The primary objective is to predict revenue based on the quantity of products sold using data science techniques.

---

## Project Requirements

This project requires a well-structured pipeline starting from raw data to predictive modeling. The input should be a CSV file that includes four key attributes: the `Date` of the transaction, the `Product` name, the `Quantity` sold, and the `Revenue` generated. First, the data must be preprocessed to identify and handle any missing or null values, ensuring model reliability. Then, it's essential to convert the `Date` into a `datetime` format to extract new time-based features like month and year, which can help in trend analysis. After preprocessing, the core step is applying **Linear Regression**, a supervised learning algorithm that helps model the relationship between quantity sold and revenue earned. The data is split into training and testing subsets to allow model evaluation. Once trained, the model generates predictions on the test data. Finally, the predicted sales are compared with actual values using plots and are validated using metrics such as R² Score and Mean Squared Error to assess prediction accuracy.

### Dataset Overview

The dataset (`sales_data.csv`) includes daily sales information across four fields:

* `Date`: Date of sale in `DD-MM-YYYY` format
* `Product`: Name of the product sold
* `Quantity`: Number of units sold
* `Revenue`: Revenue earned on that day

Sa
---

##  Libraries Used and Why

```python
import pandas as pd              # For data manipulation
import numpy as np               # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.linear_model import LinearRegression  # For model creation
from sklearn.model_selection import train_test_split  # For splitting dataset
```

---

## 🧹 Data Preprocessing in Detail

### Step 1: Load the Dataset

```python
df = pd.read_csv("sales_data.csv")
```

Loads the CSV data into a pandas DataFrame.

### Step 2: Explore Dataset Structure

```python
print(df.shape)     # Rows and columns
print(df.info())    # Data types and nulls
print(df.describe())  # Summary stats of numerical columns
```

This helps us understand data distribution and missing values.

### Step 3: Handle Missing Values

```python
print(df.isnull().sum())  # Check nulls per column
df = df.dropna()          # Remove rows with any missing data
```

Essential to avoid errors and inaccurate predictions.

### Step 4: Feature Engineering with Date

```python
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
```

We convert the Date column to datetime format and extract useful time-based features like month and year.

---

## 📊 Exploratory Data Analysis (EDA)

### Revenue vs Quantity

```python
plt.scatter(df['Quantity'], df['Revenue'])
plt.xlabel('Quantity')
plt.ylabel('Revenue')
plt.title('Revenue vs Quantity')
plt.grid(True)
plt.show()
```

This plot checks for linearity between Quantity and Revenue. A visible linear trend supports the use of linear regression.

---

## 🤖 Building the Linear Regression Model

### Step 1: Define Feature and Target

```python
X = df[['Quantity']]   # Independent variable
y = df['Revenue']      # Dependent variable (target)
```

### Step 2: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

We split the dataset into training (80%) and testing (20%) subsets to evaluate model performance.

### Step 3: Train the Model

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

The model learns the relationship between quantity and revenue.

### Step 4: Make Predictions

```python
y_pred = model.predict(X_test)
```

We use the trained model to predict revenue from test quantities.

---

## 📉 Model Results & Visualization

### Compare Actual vs Predicted Revenue

```python
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Revenue', marker='o')
plt.plot(y_pred, label='Predicted Revenue', marker='x')
plt.legend()
plt.title('Actual vs Predicted Revenue')
plt.xlabel('Index')
plt.ylabel('Revenue')
plt.grid(True)
plt.show()
```

This visualization helps us see how well the model performs by comparing real vs predicted revenue.

### Evaluate Model Accuracy

```python
from sklearn.metrics import r2_score, mean_squared_error
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
```

* **R² Score**: Measures model goodness-of-fit. Closer to 1 means better fit.
* **MSE**: Lower values mean better predictions.

---

## Conclusion :

* The linear regression model successfully predicts revenue based on the quantity of products sold.
* Proper preprocessing and EDA are crucial before training models.
