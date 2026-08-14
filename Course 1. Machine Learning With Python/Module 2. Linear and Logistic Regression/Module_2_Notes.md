# Module 2: Linear and Logistic Regression

## 1. Introduction to Regression

- **Regression:** A type of supervised learning model that models a relationship between a continuous target variable and one or more explanatory features.
- **Applications of Regression:**
  - Sales forecasting (predicting yearly sales based on leads).
  - Predicting maintenance needs for machinery.
  - Estimating real estate prices based on property size/bedrooms.
  - Environmental protection (rainfall estimation, wildfire severity).
  - Public health (disease spread prediction).

### Regression Comparisons

| Feature                     | Simple Regression                                             | Multiple Regression                                                        |
| --------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Definition**        | A single independent variable estimates a dependent variable. | Two or more independent variables estimate the dependent variable.         |
| **Example**           | Predicting CO2 emissions using*Engine Size*.                | Predicting CO2 emissions using*Engine Size* and *Number of Cylinders*. |
| **Relationship Type** | Can be Linear or Nonlinear.                                   | Can be Linear or Nonlinear.                                                |

### Common Regression Algorithms

- **Classical Models:** Linear Regression, Polynomial Regression.
- **Modern ML Models:** Random Forest, XGBoost, k-Nearest Neighbors (KNN), Support Vector Machines (SVM), Neural Networks.

## 2. Simple Linear Regression

- **Simple Linear Regression:** Models a linear relationship between a single independent variable ($x_1$) and a continuous dependent variable ($\hat{y}$).
- **The Equation:** $\hat{y} = \theta_0 + \theta_1x_1$
  - $\hat{y}$: Predicted response (e.g., CO2 Emissions)
  - $x_1$: Predictor/Independent variable (e.g., Engine Size)
  - $\theta_0$: y-intercept (Bias coefficient)
  - $\theta_1$: Slope (Coefficient for the independent variable)
- **Residual Error:** The vertical distance from the actual data point to the fitted regression line.
- **Mean Squared Error (MSE):** The average of all residual errors. It measures how poorly the regression line fits the data.

### Model Fitting Process (Ordinary Least Squares - OLS)

Linear regression aims to find the line that minimizes the Mean Squared Error (MSE). This specific approach is commonly known as **Ordinary Least Squares (OLS) regression**.

1. Calculate the means ($\bar{x}$ and $\bar{y}$) of the independent and dependent variables.
2. Calculate the slope coefficient ($\theta_1$) using the data points' distances from the means.
3. Calculate the intercept ($\theta_0$) using $\bar{y}$, $\bar{x}$, and $\theta_1$.
4. Plug the independent variable into the linear equation to make a prediction.

### OLS Advantages and Limitations

| Advantages                                                     | Limitations                                                                     |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Easy to understand and interpret.                              | Far too simplistic for capturing complex or nonlinear relationships.            |
| Solution is a straightforward calculation (no complex tuning). | Highly sensitive to outliers, which can significantly skew the regression line. |
| Very fast to compute, especially for smaller datasets.         |                                                                                 |

## Code Examples

```Python
# Simple Linear Regression Workflow (Derived from Lab Notebook)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the Data
# url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/.../FuelConsumptionCo2.csv"
# df = pd.read_csv(url)
# X = df['ENGINESIZE'].to_numpy()
# y = df['CO2EMISSIONS'].to_numpy()

# 2. Train/Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train Model
regressor = LinearRegression()
# Scikit-learn expects a 2D array for features; reshape 1D array to 2D
regressor.fit(X_train.reshape(-1, 1), y_train)

# View coefficients
print('Coefficients:', regressor.coef_[0])
print('Intercept:', regressor.intercept_)

# 4. Make Predictions
y_pred = regressor.predict(X_test.reshape(-1, 1))

# 5. Evaluate the Model
print("Mean absolute error (MAE): %.2f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error (MSE): %.2f" % mean_squared_error(y_test, y_pred))
print("Root mean squared error (RMSE): %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2-score: %.2f" % r2_score(y_test, y_pred))
```

## 3. Intro To Multiple Linear Regression

- **Multiple Linear Regression:** An extension of simple linear regression that uses two or more independent variables to predict a dependent variable.
- **Model Structure:** Models the dependent variable as a linear combination of multiple independent variables, each with its own weight/coefficient, plus an intercept term.
- **Geometric Interpretation:**
  - Simple Linear Regression fits a *line*.
  - Two-feature regression fits a *plane*.
  - Multiple features define a *hyperplane*.
- **Applications:** Measuring the effect strength of each independent variable on the dependent variable (e.g., predicting car CO2 emissions from *engine size*, *cylinders*, and *fuel consumption*).
- **Pitfalls & Considerations:**
  - **Overfitting:** Adding too many variables can cause the model to memorize the training data but perform poorly on new data.
  - **Categorical Variables:** Can be included by converting them into numerical or Boolean features.
  - **Collinearity:** Correlated independent variables should be avoided or removed to prevent misleading results.

### Code Examples: Multiple Linear Regression

```python
# Multiple Linear Regression Workflow (Derived from Lab Notebook)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load the Data
# url = "..."
# df = pd.read_csv(url)

# Select multiple features for the model
# X = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']].to_numpy()
# y = df['CO2EMISSIONS'].to_numpy()

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train Model
regressor = LinearRegression()
# Scikit-learn can handle 2D arrays natively for multiple features
regressor.fit(X_train, y_train)

# View coefficients (will return an array of coefficients, one for each feature)
print('Coefficients:', regressor.coef_)
print('Intercept:', regressor.intercept_)

# 4. Make Predictions
y_pred = regressor.predict(X_test)

# 5. Evaluate the Model
print("Mean squared error (MSE): %.2f" % mean_squared_error(y_test, y_pred))
```

## 4. Non-Linear Regression

- **Non-Linear Regression:** A statistical method used to model a complex relationship between a dependent variable and independent variables using a non-linear equation (e.g., polynomial, exponential, logarithmic, sinusoidal).
- **When to Use:** When the underlying data trend follows a smoothed curve rather than a straight line, and linear regression would underfit the data.
- **Common Non-Linear Models:**
  - **Exponential/Compound Growth:** (e.g., GDP growth over time).
  - **Logarithmic:** Diminishing returns (e.g., incremental gains in productivity vs. consecutive hours worked).
  - **Periodicity:** Sinusoidal seasonal variations (e.g., monthly rainfall or temperature).
- **Model Selection & Optimization:**
  - Visual analysis of scatter plots helps identify the relationship type.
  - Optimization techniques like Gradient Descent can estimate parameters.
  - If a specific mathematical form is unknown, modern ML models (Regression Trees, Random Forests, Neural Networks, etc.) are powerful alternatives.

### Polynomial Regression

- **Definition:** A special form of non-linear regression that fits data using polynomial expressions of the features (e.g., x^2, x^3).
- **How it Works:** It introduces new variables representing powers of the original features. Because this transforms the problem, it expresses a *non-linear* dependence on the input features but a *linear* dependence on the regression coefficients. Thus, it can be solved using ordinary multiple linear regression.
- **Equation:** y = theta_0 + theta_1*x_1 + theta_2*x_2 + theta_3*x_3 (where x_1 = x, x_2 = x^2, x_3 = x^3)
- **Overfitting Warning:** While it can fit data very well, picking an arbitrarily high degree polynomial will cause the model to capture random noise rather than underlying patterns (overfitting).

### Code Examples: Polynomial Regression

```python
# Polynomial Regression Workflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Assume X and y are loaded 1D arrays, e.g., Engine Size and CO2 Emissions

# 1. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Transform the Data into Polynomial Features
# Here we choose a degree of 2 (Quadratic)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))

# 3. Fit a Linear Regression Model on the Polynomial Features
# The problem is now linearized
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# View the coefficients for the polynomial terms
print('Coefficients:', poly_reg.coef_)
print('Intercept:', poly_reg.intercept_)

# 4. Make Predictions
# Must transform test data the exact same way before predicting
X_test_poly = poly.fit_transform(X_test.reshape(-1, 1))
y_pred = poly_reg.predict(X_test_poly)

# 5. Evaluate the Model
print("R2-score: %.2f" % r2_score(y_test, y_pred))
```

## 5. Introduction to Logistic Regression
- **Logistic Regression:** A supervised machine learning algorithm used primarily for **binary classification** tasks (e.g., True/False, Pass/Fail, Spam/Not Spam), despite having "regression" in its name.
- **How it Works:** Instead of predicting a continuous numerical value, it predicts the **probability** that an input belongs to a particular class. It uses the **Sigmoid Function** (or Logit function) to convert any real number into a probability between 0 and 1.
  - *Equation:* $P = 1 / (1 + e^{-z})$ where $z = \theta_0 + \theta_1 x_1 + \dots$
  - The model maps predictions to probabilities, and a **decision boundary** (threshold, typically 0.5) is used to assign the final class (e.g., $P \ge 0.5 \rightarrow 1$, $P < 0.5 \rightarrow 0$).
- **Why Not Linear Regression?** Linear regression can predict values less than 0 or greater than 1, making it unsuitable for probabilities.
- **Applications:** Customer churn prediction, email spam detection, disease diagnosis, credit card fraud detection.

## 6. Training a Logistic Regression Model
- **Goal of Training:** To find the optimal weights/parameters ($\theta$) that yield the most accurate probabilities and minimize prediction error.
- **Cost Function (Log Loss / Cross-Entropy):** Instead of Mean Squared Error, logistic regression uses Log Loss.
  - Log Loss **rewards** confident, correct predictions (low loss) and **penalizes** confident, incorrect predictions (huge loss).
- **Optimization (Gradient Descent):** 
  - **Gradient Descent:** An iterative optimization algorithm that adjusts parameters in the direction of the steepest descent of the cost function, controlled by a **Learning Rate ($\alpha$)**.
  - **Stochastic Gradient Descent (SGD):** A faster, more scalable variation that uses random subsets (mini-batches) of the training data rather than the entire dataset per iteration. Ideal for large datasets.

## Logistic Regression Comparisons
| Feature | Linear Regression | Logistic Regression |
|---|---|---|
| **Problem Type** | Regression | Classification |
| **Output** | Continuous number (any real value) | Probability (0 to 1) -> Class Label |
| **Equation Base** | Straight Line ($y = mx + c$) | Sigmoid Curve |
| **Cost Function** | Mean Squared Error (MSE) | Log Loss (Cross-Entropy) |

## Code Examples: Logistic Regression
```python
# Logistic Regression - Customer Churn Prediction
# Modern Best Practices Version
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# 1. Load Dataset
# df = pd.read_csv("ChurnData.csv")
# X = df[['tenure', 'age', 'income', 'employ']]
# y = df['churn']

# 2. Train-Test Split (Split BEFORE scaling to prevent Data Leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 3. Feature Scaling (Fit ONLY on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Logistic Regression Model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# 5. Make Predictions
y_pred = model.predict(X_test_scaled) # Class prediction (0 or 1)
y_prob = model.predict_proba(X_test_scaled) # Probability of each class

# 6. Evaluate Model
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Log Loss : {log_loss(y_test, y_prob):.4f}")

# View learned weights (Coefficients)
print("Coefficients:", model.coef_[0])
```

## Module 2 Cheat Sheet Summary
- **train_test_split:** Splits data into train and test sets to evaluate model generalization.
- **StandardScaler:** Standardizes features to remove mean and scale to unit variance (crucial for models like Logistic Regression).
- **Evaluation Metrics:**
  - *Regression:* `mean_absolute_error` (MAE), `mean_squared_error` (MSE), `root_mean_squared_error` (RMSE), `r2_score`.
  - *Classification:* `log_loss`, `accuracy_score`, `confusion_matrix`, `classification_report`.
