# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the dataset.

2.Convert categorical data and scale features.

3.Train the Logistic Regression model.

4.Predict and evaluate the placement results.


## Program:

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ROHITH VIGNESH
RegisterNumber: 25000290  
*/
```
# Logistic Regression for Student Placement Prediction

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 2️⃣ Load Dataset
data = pd.read_csv("Placement_Data.csv")   

print("Dataset Preview:")
print(data.head())

# 3️⃣ Drop Unnecessary Columns
data = data.drop(["sl_no", "salary"], axis=1)

# 4️⃣ Convert Target Variable (status) to Binary
# Placed = 1, Not Placed = 0
data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})

# 5️⃣ Separate Features and Target
X = data.drop("status", axis=1)
y = data["status"]

# 6️⃣ One-Hot Encode Categorical Variables
X = pd.get_dummies(X, drop_first=True)

print("\nAfter Encoding:")
print(X.head())

# 7️⃣ Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 9️⃣ Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 🔟 Make Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 1️⃣1️⃣ Model Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
<img width="994" height="766" alt="Screenshot 2026-03-17 082658" src="https://github.com/user-attachments/assets/b03e535d-86d2-4ed5-bc7e-a49b9d5ad573" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
