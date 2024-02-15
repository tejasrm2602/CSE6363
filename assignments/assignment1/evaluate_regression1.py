import pickle
from TrainRegression1 import X_test_model1, Y_test_model1  # Assuming TrainRegression4.py contains the test data
import matplotlib.pyplot as plt

# Reshaping test data to 2D arrays
X_test_model1_reshaped = X_test_model1.reshape(-1, 1)

# Load the model parameters from the saved file
with open('NRmodel1_params.pkl', 'rb') as file:
    model_params = pickle.load(file)

# Predict on the test set
y_pred_model1 = model_params.predict(X_test_model1_reshaped)

# Calculate and print the mean squared error using the model's score method
mse_model1 = model_params.score(y_pred_model1, Y_test_model1)
print(f"\nMean Squared Error (Model 4): {mse_model1}")

# plotting linear regression graph between sepal length and sepal width
plt.scatter(X_test_model1, Y_test_model1, label='Actual Data')
plt.plot(X_test_model1, y_pred_model1, color='red', label='Linear Regression Line')
plt.xlabel('sepal length (Model 1)')
plt.ylabel('sepal width (Model 1)')
plt.title('Linear Regression Evaluation (Model 1)')
plt.legend()
plt.show()