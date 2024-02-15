import numpy as np
import pickle
import matplotlib.pyplot as plt


class MLR:
    def __init__(self, ip_size, op_size, lrn_rate=0.01, regz=0):
        self.ip_size = ip_size
        self.op_size = op_size
        self.lrn_rate = lrn_rate
        self.regz = regz
        self.weights = np.zeros((ip_size, op_size))
        self.bias = np.zeros(op_size)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def gradient(self, X, y):
        num_sam = X.shape[0]

        # Computing gradients for weights
        grad_weights = (-2 / num_sam) * np.dot(X.T, (y - self.predict(X)))

        # Computing gradients for bias
        grad_bias = (-2 / num_sam) * np.sum(y - self.predict(X), axis=0)

        return grad_weights, grad_bias

    def regularization(self):
        # Apply L2 regularization to weights
        self.weights -= self.regz * self.weights
        # Apply L2 regularization to bias
        self.bias -= self.regz * self.bias

    def fit(self, X, y, epochs=100, val_split=0.1, patience=3):
        num_sam = X.shape[0]
        num_val = int(num_sam * val_split)

        X_train, y_train = X[:-num_val], y[:-num_val]
        X_val, y_val = X[-num_val:], y[-num_val:]

        best_weights = np.copy(self.weights)
        best_bias = np.copy(self.bias)
        best_loss = float('inf')
        count = 0
        training_loss_history = []
        val_loss_history = []

        for epoch in range(epochs):

            y_pred = self.predict(X)

            grad_weights, grad_bias = self.gradient(X_train, y_train)

            # Update weights and bias
            self.weights -= self.lrn_rate * grad_weights
            self.bias -= self.lrn_rate * grad_bias

            # Apply regularization
            self.regularization()
            # Evaluate on validation set
            val_loss = self.score(X_val, y_val)

            # Evaluate on training and validation sets
            training_loss = self.score(X_train, y_train)
            val_loss = self.score(X_val, y_val)

            # Append to loss history
            training_loss_history.append(training_loss)
            val_loss_history.append(val_loss)
            print(f"Epoch {epoch + 1}, Training Loss: {training_loss}")
            # Check for early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = np.copy(self.weights)
                best_bias = np.copy(self.bias)
                count = 0
            else:
                count += 1
                if count >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self.weights = best_weights
        self.bias = best_bias

        return training_loss_history, val_loss_history

    # Plot the training and validation loss history.
    def plot_loss_history(self, training_loss_history, val_loss_history=None):

        plt.plot(training_loss_history, label='Training Loss')
        if val_loss_history is not None:
            plt.plot(val_loss_history, label='Validation Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.show()
        print("Final Weights:", self.weights)
        print("Final Bias:", self.bias)

    def score(self, X, y):
        y_pred = self.predict(X)
        n, m = y.shape
        mse = np.sum((y - y_pred) ** 2) / (n * m)
        return mse

    # Save model parameters to a file using pickle.
    def save(self, file_path):

        model_params = {
            'weights': self.weights,
            'bias': self.bias,
            'batch_size': self.batch_size,
            'regz': self.regz,
            'max_epochs': self.max_epochs,
            'patience': self.patience
        }

        with open(file_path, 'wb') as file:
            pickle.dump(model_params, file)

    # Load model parameters from a file using pickle.
    def load(self, file_path):

        with open(file_path, 'rb') as file:
            model_params = pickle.load(file)

        self.weights = model_params['weights']
        self.bias = model_params['bias']
        self.batch_size = model_params['batch_size']
        self.regz = model_params['regz']
        self.max_epochs = model_params['max_epochs']
        self.patience = model_params['patience']
