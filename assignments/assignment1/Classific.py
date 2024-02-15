import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class LogisticRegression:
    def __init__(self, lrn_rate=0.01, epochs=100):
        self.lrn_rate = lrn_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        global z
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        num_samp, num_fea = X.shape
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            pred = self.sigmoid(z)

            grad_wt = np.dot(X.T, (pred - y)) / num_samp
            grad_bias = np.sum(pred - y) / num_samp

            self.weights -= self.lrn_rate * grad_wt
            self.bias -= self.lrn_rate * grad_bias
        print(z)

    def predict(self, X):

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        z = np.dot(X, self.weights) + self.bias
        pred = self.sigmoid(z)
        return (pred >= 0.5).astype(int)


# Loading iris dataset
iris = load_iris()
X, y = iris.data, (iris.target == 2).astype(int)  # Binary classification, we are classifying class 2 vs other classes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# logistic regression model for petal length
log_reg_petal = LogisticRegression(lrn_rate=0.01, epochs=100)
log_reg_petal.fit(X_train[:, 2], y_train)

# logistic regression model for sepal length/width
log_reg_sepal = LogisticRegression(lrn_rate=0.01, epochs=100)
log_reg_sepal.fit(X_train[:, :2], y_train)

# logistic regression model for all features
log_reg_all = LogisticRegression(lrn_rate=0.01, epochs=100)
log_reg_all.fit(X_train, y_train)

# Making predictions on the test set for each variant
y_pred_petal = log_reg_petal.predict(X_test[:, 2])
y_pred_sepal = log_reg_sepal.predict(X_test[:, :2])
y_pred_all = log_reg_all.predict(X_test)
