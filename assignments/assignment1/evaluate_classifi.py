import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Classification import LogisticRegression
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data,(iris.target == 2).astype(int)

# Standardizing the input features
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.1, random_state=42, stratify=y)

# Initialize and fit the logistic regression model
log_reg_petal = LogisticRegression(lrn_rate=0.01, epochs=100)
log_reg_petal.fit(X_train[:, 2], y_train)

print("\nActual values",y_test)

# Making predictions on the test set
y_pred_petal = log_reg_petal.predict(X_test[:, 2])
print("\npredicted values", y_pred_petal)

# Evaluating accuracy
accuracy = np.mean(y_pred_petal == y_test)
print(f"\nAccuracy (Petal): {accuracy*100} %")

# plotting graph
plt.figure(figsize=(8, 5))
plot_decision_regions(X_test[:, 2].reshape(-1,1), y_test, clf=log_reg_petal, legend=2)
plt.title('Logistic Regression Decision Regions (Petal length)')
plt.show()