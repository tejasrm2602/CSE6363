import pickle
from TrainRegression2 import X_test_model2, Y_test_model2  # Assuming TrainRegression4.py contains the test data
import matplotlib.pyplot as plt

# Reshaping test data to 2D arrays
X_test_model2_reshaped = X_test_model2.reshape(-1, 1)

# Load the model parameters from the saved file
with open('NRmodel2_params.pkl', 'rb') as file:
    model_params = pickle.load(file)

# Predict on the test set
y_pred_model2 = model_params.predict(X_test_model2_reshaped)

# Calculate and print the mean squared error using the model's score method
mse_model2 = model_params.score(y_pred_model2, Y_test_model2)
print(f"\nMean Squared Error (Model 2): {mse_model2}")

# plotting linear regression graph between petal length and petal width
plt.scatter(X_test_model2, Y_test_model2, label='Actual Data')
plt.plot(X_test_model2, y_pred_model2, color='red', label='Linear Regression Line')
plt.xlabel('Petal Length (Model 2)')
plt.ylabel('Petal width (Model 2)')
plt.title('Linear Regression Evaluation (Model 2)')
plt.legend()
plt.show()