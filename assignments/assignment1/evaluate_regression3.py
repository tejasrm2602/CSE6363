import pickle
from TrainRegression3 import X_test_model3, Y_test_model3  # Assuming TrainRegression4.py contains the test data
import matplotlib.pyplot as plt

# Reshaping test data to 2D arrays
X_test_model3_reshaped = X_test_model3.reshape(-1, 1)

# Load the model parameters from the saved file
with open('NRmodel3_params.pkl', 'rb') as file:
    model_params = pickle.load(file)

# Predict on the test set
y_pred_model3 = model_params.predict(X_test_model3_reshaped)

# Calculate and print the mean squared error using the model's score method
mse_model3 = model_params.score(y_pred_model3, Y_test_model3)
print(f"\nMean Squared Error (Model 3): {mse_model3}")

# plotting linear regression graph between petal length and sepal length
plt.scatter(X_test_model3, Y_test_model3, label='Actual Data')
plt.plot(X_test_model3, y_pred_model3, color='red', label='Linear Regression Line')
plt.xlabel('Petal length (Model 3)')
plt.ylabel('Sepal length (Model 3)')
plt.title('Linear Regression Evaluation (Model 3)')
plt.legend()
plt.show()