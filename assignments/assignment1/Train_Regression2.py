import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearReg import LinearRegression
import pickle

# Load Iris dataset for Model 2
iris_data_model2 = load_iris()
features_model2, labels_model2 = iris_data_model2.data, iris_data_model2.target

feature_index_model2, target_index_model2 = 2, 3   # Petal length predicting petal width
X_selected_model2 = features_model2[:, feature_index_model2]
Y_selected_model2 = features_model2[:, target_index_model2]

# Split the data into training and testing sets for Model 2
X_train_model2, X_test_model2, Y_train_model2, Y_test_model2 = train_test_split(X_selected_model2, Y_selected_model2, test_size=0.1, random_state=42)

# Reshape to 2D arrays for Model 2
X_train_reshaped_model2 = X_train_model2.reshape(-1, 1)
X_test_reshaped_model2 = X_test_model2.reshape(-1, 1)

# Initialize and fit the non-regularized linear regression model for Model 2
non_reg_model2 = LinearRegression(batch_size=32, regularization=0, max_epochs=100, patience=3, lrn_rate=0.01)
training_loss_hist_model2, _ = non_reg_model2.fit(X_train_reshaped_model2, Y_train_model2, lrn_rate=0.01)

# Plotting the training loss history for Model 2
plt.plot(training_loss_hist_model2, label='Non-Reg Training Loss (Model 2)')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('NON REG Training Loss Over Epochs (Model 2)')
plt.legend()
plt.show()

# Save the non-regularized model parameters for Model 2
with open('NRmodel2_params.pkl', 'wb') as file:
    pickle.dump(non_reg_model2, file)

print("\nNon-Regularized Model Parameters (Model 2):")
print("Weights:", non_reg_model2.weights)
print("Bias:", non_reg_model2.bias)

# Initialize and fit the regularized linear regression model for Model 2
reg_model2 = LinearRegression(batch_size=32, regularization=1, max_epochs=100, patience=3, lrn_rate=0.01)
training_loss_hist_r_model2, _ = reg_model2.fit(X_train_reshaped_model2, Y_train_model2, lrn_rate=0.01)

# Plot the training loss history for Model 2
plt.plot(training_loss_hist_r_model2, label='Regularised Training Loss (Model 2)')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('REG Training Loss Over Epochs (Model 2)')
plt.legend()
plt.show()

# Save the regularized model parameters for Model 2
with open('Rmodel2_params.pkl', 'wb') as file:
    pickle.dump(reg_model2, file)

# Print regularized model parameters for Model 2
print("\nRegularized Model Parameters (Model 2):")
print("Weights:", reg_model2.weights)
print("Bias:", reg_model2.bias)

# Calculate and print the difference in parameters between regularized and non-regularized models for Model 2
weights_diff_model2 = reg_model2.weights - non_reg_model2.weights
bias_diff_model2 = reg_model2.bias - non_reg_model2.bias

print("\nDifference in Parameters (Model 2):")
print("Weights Difference:", weights_diff_model2)
print("Bias Difference:", bias_diff_model2)
