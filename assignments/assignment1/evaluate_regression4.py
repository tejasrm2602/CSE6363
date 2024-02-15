import pickle
from TrainRegression4 import X_test_model4, Y_test_model4  # Assuming TrainRegression4.py contains the test data
import matplotlib.pyplot as plt

# Reshaping test data to 2D arrays
X_test_model4_reshaped = X_test_model4.reshape(-1, 1)

# Load the model parameters from the saved file
with open('NRmodel4_params.pkl', 'rb') as file:
    model_params = pickle.load(file)

# Predict on the test set
y_pred_model4 = model_params.predict(X_test_model4_reshaped)

# Calculate and print the mean squared error using the model's score method
mse_model4 = model_params.score(y_pred_model4, Y_test_model4)
print(f"\nMean Squared Error (Model 4): {mse_model4}")

# Plotting linear regression graph between petal width and petal length
plt.scatter(X_test_model4, Y_test_model4, label='Actual Data')
plt.plot(X_test_model4, y_pred_model4, color='red', label='Linear Regression Line')
plt.xlabel('Petal Width (Model 4)')
plt.ylabel('Petal Length (Model 4)')
plt.title('Linear Regression Evaluation (Model 4)')
plt.legend()
plt.show()
