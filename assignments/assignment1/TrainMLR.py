import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from MLR import MLR
import pickle

# Loading the Iris dataset
iris = load_iris()
X, y = iris.data[:, :2], iris.data[:, 2:] # using sepal length sepal width to predict petal length and petal width

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize and fit the multi-output linear regression model
ip_size = X_train.shape[1]
op_size = y_train.shape[1]

# Without Regularization
NR_MLR_model = MLR(ip_size, op_size, lrn_rate=0.01, regz=0)
training_loss_history, _ = NR_MLR_model.fit(X_train, y_train, epochs=100)

# Plotting
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.plot(training_loss_history, label='Non-Reg Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('NON REG Training Loss Over Epochs')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(X_test[:, 0], y_test[:, 0], label='Actual Petal Length')
plt.scatter(X_test[:, 0], NR_MLR_model.predict(X_test)[:, 0], color='red', label='Predicted Petal Length')
plt.scatter(X_test[:, 1], y_test[:, 1], label='Actual Petal Width')
plt.scatter(X_test[:, 1], NR_MLR_model.predict(X_test)[:, 1], color='blue', label='Predicted Petal Width')
plt.xlabel('Sepal Length and Width')
plt.ylabel('Petal Length and Width')
plt.title('Without Regularization')
plt.legend()

with open('NRMLRmodel_params.pkl', 'wb') as file:
    pickle.dump(NR_MLR_model, file)

# With Regularization
MLR_model_R = MLR(ip_size, op_size, lrn_rate=0.01, regz=0.1)
training_loss_history_r, _ = MLR_model_R.fit(X_train, y_train, epochs=100)

# Plotting
plt.subplot(2, 2, 3)
plt.plot(training_loss_history_r, label='Reg Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('REG Training Loss Over Epochs')
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(X_test[:, 0], y_test[:, 0], label='Actual Petal Length')
plt.scatter(X_test[:, 0], MLR_model_R.predict(X_test)[:, 0], color='red', label='Predicted Petal Length')
plt.scatter(X_test[:, 1], y_test[:, 1], label='Actual Petal Width')
plt.scatter(X_test[:, 1], MLR_model_R.predict(X_test)[:, 1], color='blue', label='Predicted Petal Width')
plt.xlabel('Sepal Length and Width')
plt.ylabel('Petal Length and Width')
plt.title('With Regularization')
plt.legend()
plt.tight_layout()
plt.show()

with open('R_MLR_model_params.pkl', 'wb') as file:
    pickle.dump(MLR_model_R, file)