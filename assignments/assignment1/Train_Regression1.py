import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearReg import LinearRegression
import pickle

# Load Iris dataset for Model 1
iris_data_model1 = load_iris()
features_model1, labels_model1 = iris_data_model1.data, iris_data_model1.target

feature_index_model1, target_index_model1 = 0, 1  # sepal length predicting sepal width
X_selected_model1 = features_model1[:, feature_index_model1]
Y_selected_model1 = features_model1[:, target_index_model1]

# Split the data into training and testing sets for Model 1
X_train_model1, X_test_model1, Y_train_model1, Y_test_model1 = train_test_split(X_selected_model1, Y_selected_model1, test_size=0.1, random_state=42)

# Reshape to 2D arrays for Model 1
X_train_reshaped_model1 = X_train_model1.reshape(-1, 1)
X_test_reshaped_model1 = X_test_model1.reshape(-1, 1)

# Initialize and fit the non-regularized linear regression model for Model 1
non_reg_model1 = LinearRegression(batch_size=32, regularization=0, max_epochs=100, patience=3, lrn_rate=0.01)
training_loss_hist_model1, _ = non_reg_model1.fit(X_train_reshaped_model1, Y_train_model1)

# Plotting the training loss history for Model 1
plt.plot(training_loss_hist_model1, label='Non-Reg Training Loss (Model 1)')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('NON REG Training Loss Over Epochs (Model 1)')
plt.legend()
plt.show()

# Save the non-regularized model parameters for Model 1
with open('NRmodel1_params.pkl', 'wb') as file:
    pickle.dump(non_reg_model1, file)

print("\nNon-Regularized Model Parameters (Model 1):")
print("Weights:", non_reg_model1.weights)
print("Bias:", non_reg_model1.bias)

# Initialize and fit the regularized linear regression model for Model 1
reg_model1 = LinearRegression(batch_size=32, regularization=1, max_epochs=100, patience=3, lrn_rate=0.01)
training_loss_hist_r_model1, _ = reg_model1.fit(X_train_reshaped_model1, Y_train_model1)

# Plot the training loss history for Model 1
plt.plot(training_loss_hist_r_model1, label='Regularised Training Loss (Model 1)')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('REG Training Loss Over Epochs (Model 1)')
plt.legend()
plt.show()

# Save the regularized model parameters for Model 1
with open('Rmodel1_params.pkl', 'wb') as file:
    pickle.dump(reg_model1, file)

# Print regularized model parameters for Model 1
print("\nRegularized Model Parameters (Model 1):")
print("Weights:", reg_model1.weights)
print("Bias:", reg_model1.bias)

# Calculate and print the difference in parameters between regularized and non-regularized models for Model 1
weights_diff_model1 = reg_model1.weights - non_reg_model1.weights
bias_diff_model1 = reg_model1.bias - non_reg_model1.bias

print("\nDifference in Parameters (Model 1):")
print("Weights Difference:", weights_diff_model1)
print("Bias Difference:", bias_diff_model1)
