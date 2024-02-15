import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Classification import LogisticRegression
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target
X = X[:, [0, 2]]

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.1, random_state=42, stratify=y)

log_reg_all = LogisticRegression(lrn_rate=0.01, epochs=100)
log_reg_all.fit(X_train, y_train)

print("Actual values", y_test)

y_pred_all = log_reg_all.predict(X_test)
print("\nPredicted values", y_pred_all)

accuracy = np.mean(y_pred_all == y_test)
print(f"\nAccuracy (All Features): {accuracy * 100}%")

# # Plotting Decision Regions
# plt.figure(figsize=(8, 5))
# plot_decision_regions(X=X_test, y=y_test, clf=log_reg_all, legend=2)
# plt.title('Logistic Regression of all classes')
# plt.show()
