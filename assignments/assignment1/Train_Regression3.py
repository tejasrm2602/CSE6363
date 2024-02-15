import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearReg import LinearRegression
import pickle

# Load Iris dataset for Model 3
iris_data_model3 = load_iris()
features_model3, labels_model3 = iris_data_model3.data, iris_data_model3.target

feature_index_model3, target_index_model3 = 2, 0  # petal length predicting sepal length
X_selected_model3 = features_model3[:, feature_index_model3]
Y_selected_model3 = features_model3[:, target_index_model3]

# Split the data into training and testing sets for Model 3
X_train_model3, X_test_model3, Y_train_model3, Y_test_model3 = train_test_split(X_selected_model3, Y_selected_model3, test_size=0.1, random_state=42)

# Reshape to 2D arrays for Model 3
X_train_reshaped_model3 = X_train_model3.reshape(-1, 1)
X_test_reshaped_model3 = X_test_model3.reshape(-1, 1)

# Initialize and fit the non-regularized linear regression model for Model 3
non_reg_model3 = LinearRegression(batch_size=32, regularization=0, max_epochs=100, patience=3, lrn_rate=0.01)
training_loss_hist_model3, _ = non_reg_model3.fit(X_train_reshaped_model3, Y_train_model3, lrn_rate=0.01)

# Plotting the training loss history for Model 3
plt.plot(training_loss_hist_model3, label='Non-Reg Training Loss (Model 3)')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('NON REG Training Loss Over Epochs (Model 3)')
plt.legend()
plt.show()

# Save the non-regularized model parameters for Model 3
with open('NRmodel3_params.pkl', 'wb') as file:
    pickle.dump(non_reg_model3, file)

print("\nNon-Regularized Model Parameters (Model 3):")
print("Weights:", non_reg_model3.weights)
print("Bias:", non_reg_model3.bias)

# Initialize and fit the regularized linear regression model for Model 3
reg_model3 = LinearRegression(batch_size=32, regularization=1, max_epochs=100, patience=3, lrn_rate=0.01)
training_loss_hist_r_model3, _ = reg_model3.fit(X_train_reshaped_model3, Y_train_model3, lrn_rate=0.01)

# Plot the training loss history for Model 3
plt.plot(training_loss_hist_r_model3, label='Regularised Training Loss (Model 3)')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('REG Training Loss Over Epochs (Model 3)')
plt.legend()
plt.show()

# Save the regularized model parameters for Model 3
with open('Rmodel3_params.pkl', 'wb') as file:
    pickle.dump(reg_model3, file)

# Print regularized model parameters for Model 3
print("\nRegularized Model Parameters (Model 3):")
print("Weights:", reg_model3.weights)
print("Bias:", reg_model3.bias)

# Calculate and print the difference in parameters between regularized and non-regularized models for Model 3
weights_diff_model3 = reg_model3.weights - non_reg_model3.weights
bias_diff_model3 = reg_model3.bias - non_reg_model3.bias

print("\nDifference in Parameters (Model 3):")
print("Weights Difference:", weights_diff_model3)
print("Bias Difference:", bias_diff_model3)
