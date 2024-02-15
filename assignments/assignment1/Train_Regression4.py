import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearReg import LinearRegression
import pickle

# Load Iris dataset for Model 4
iris_data_model4 = load_iris()
features_model4, labels_model4 = iris_data_model4.data, iris_data_model4.target

feature_index_model4, target_index_model4 = 3, 2  # petal width predicting petal length for Model 4
X_selected_model4 = features_model4[:, feature_index_model4]
Y_selected_model4 = features_model4[:, target_index_model4]

# Split the data into training and testing sets for Model 4
X_train_model4, X_test_model4, Y_train_model4, Y_test_model4 = train_test_split(X_selected_model4, Y_selected_model4, test_size=0.1, random_state=42)

# Reshape to 2D arrays for Model 4
X_train_reshaped_model4 = X_train_model4.reshape(-1, 1)
X_test_reshaped_model4 = X_test_model4.reshape(-1, 1)

# Initialize and fit the non-regularized linear regression model for Model 4
non_reg_model4 = LinearRegression(batch_size=32, regularization=0, max_epochs=100, patience=3, lrn_rate=0.01)
training_loss_hist_model4, _ = non_reg_model4.fit(X_train_reshaped_model4, Y_train_model4, lrn_rate=0.01)

# Plotting the training loss history for Model 4
plt.plot(training_loss_hist_model4, label='Non-Reg Training Loss (Model 4)')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('NON REG Training Loss Over Epochs (Model 4)')
plt.legend()
plt.show()

# Save the non-regularized model parameters for Model 4
with open('NRmodel4_params.pkl', 'wb') as file:
    pickle.dump(non_reg_model4, file)

print("\nNon-Regularized Model Parameters (Model 4):")
print("Weights:", non_reg_model4.weights)
print("Bias:", non_reg_model4.bias)

# Initialize and fit the regularized linear regression model for Model 4
reg_model4 = LinearRegression(batch_size=32, regularization=1, max_epochs=100, patience=3, lrn_rate=0.01)
training_loss_hist_r_model4, _ = reg_model4.fit(X_train_reshaped_model4, Y_train_model4, lrn_rate=0.01)

# Plot the training loss history for Model 4
plt.plot(training_loss_hist_r_model4, label='Regularised Training Loss (Model 4)')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('REG Training Loss Over Epochs (Model 4)')
plt.legend()
plt.show()

# Save the regularized model parameters for Model 4
with open('Rmodel4_params.pkl', 'wb') as file:
    pickle.dump(reg_model4, file)

# Print regularized model parameters for Model 4
print("\nRegularized Model Parameters (Model 4):")
print("Weights:", reg_model4.weights)
print("Bias:", reg_model4.bias)

# Calculate and print the difference in parameters between regularized and non-regularized models for Model 4
weights_diff_model4 = reg_model4.weights - non_reg_model4.weights
bias_diff_model4 = reg_model4.bias - non_reg_model4.bias

print("\nDifference in Parameters (Model 4):")
print("Weights Difference:", weights_diff_model4)
print("Bias Difference:", bias_diff_model4)
