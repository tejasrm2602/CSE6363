import matplotlib.pyplot as plt
import pickle
from TrainMLR import X_test, y_test

# Load the trained model parameters
with open('NRMLRmodel_params.pkl', 'rb') as file:
    NR_MLR_model = pickle.load(file)

# Predicting on the test set
y_pred = NR_MLR_model.predict(X_test)
print("\nPredicted petal length and width\n", y_pred)
# calculating mean square error
mse = NR_MLR_model.score(y_pred, y_test)
print(f"\n Mean Squared Error: {mse}")

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], y_test[:, 0], label='Actual Petal Length')
plt.scatter(X_test[:, 0], NR_MLR_model.predict(X_test)[:, 0], color='red', label='Predicted Petal Length')
plt.scatter(X_test[:, 1], y_test[:, 1], label='Actual Petal Width')
plt.scatter(X_test[:, 1], NR_MLR_model.predict(X_test)[:, 1], color='blue', label='Predicted Petal Width')
plt.xlabel('Sepal Length and Width ')
plt.ylabel('Petal Length and Width ')
plt.title('Multi-Output Linear Regression')
plt.legend()
plt.show()