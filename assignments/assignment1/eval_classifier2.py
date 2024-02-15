import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Classification import LogisticRegression
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, (iris.target == 2).astype(int)

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.1, random_state=42, stratify=y)

log_reg_sepal = LogisticRegression(lrn_rate=0.01, epochs=100)
log_reg_sepal.fit(X_train[:, :2], y_train)

print("Actual values",y_test)

y_pred_sepal = log_reg_sepal.predict(X_test[:, :2])
print("Predicted values",y_pred_sepal)

accuracy = np.mean(y_pred_sepal == y_test)
print(f"Accuracy (Sepal): {accuracy*100}%")

plt.figure(figsize=(8, 5))
plot_decision_regions(X_test[:, :2], y_test, clf=log_reg_sepal, legend=2)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Logistic Regression Decision Regions (Sepal)')
plt.show()