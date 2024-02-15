import numpy as np
import matplotlib.pyplot as plt
import pickle


class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3, lrn_rate=0.01):
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None
        self.lrn_rate = lrn_rate

    def fit(self, X, y, X_val=None, y_val=None, batch_size=32, regularization=0, max_epochs=100, patience=3, lrn_rate=0.01):

        split_in = int(0.9 * len(X))
        X_train, X_val = X[:split_in], X[split_in:]
        y_train, y_val = y[:split_in], y[split_in:]
        X_val = X_val[:, None]
        y_val = y_val[:, None]

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        training_loss_hist = []
        val_loss_hist = []
        best_weights = np.copy(self.weights)
        best_bias = self.bias
        best_loss = float('inf')
        count = 0

        for epoch in range(max_epochs):
            ind = np.random.permutation(len(X))
            X_train_shfl = X[ind]
            y_train_shfl = y[ind]

            for i in range(0, len(X), batch_size):
                X_batch = X_train_shfl[i:i + batch_size]
                y_batch = y_train_shfl[i:i + batch_size]

                y_pred = self.predict(X_batch)
                loss = np.mean((y_batch - y_pred) ** 2)

                # Gradient Descent Step
                grad_weights = -2 * np.dot(X_batch.T, (y_batch - y_pred)) / len(X_batch) + 2 * self.regularization * self.weights
                grad_bias = -2 * np.sum(y_batch - y_pred) / len(X_batch)

                # Updating weights and bias 
                self.weights -= lrn_rate * grad_weights
                self.bias -= lrn_rate * grad_bias

                # Record training loss for the epoch
                training_loss = np.mean((y - self.predict(X)) ** 2)
                training_loss_hist.append(training_loss)
                print(f"Epoch {epoch + 1}, Training Loss: {training_loss}")

            # Validation loss calculation
            y_val_pred = self.predict(X_val)
            val_loss = self.score(y_val_pred, y_val)
            val_loss_hist.append(val_loss)

            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = np.copy(self.weights)
                best_bias = self.bias
                count = 0
            else:
                count += 1

            if count >= patience:
                print(f"\nStopping early at epoch {epoch + 1} due to no improvement in validation loss.")
                break

        # Set the model parameters to the best values
        self.weights = best_weights
        self.bias = best_bias

        return training_loss_hist, val_loss_hist

    # Plot the training and validation loss history
    def plot_loss_history(self, training_loss_hist, val_loss_hist=None):
        
        plt.plot(training_loss_hist, label='Training Loss')
        if val_loss_hist is not None:
            plt.plot(val_loss_hist, label='Validation Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.show()
        print("Final Weights:", self.weights)
        print("Final Bias:", self.bias)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def score(self, y_pred, y):
        mse = np.mean((y - y_pred) ** 2)
        return mse

    # Save model parameters to a file using pickle.
    def save(self, file_path):
        
        model_params = {
            'weights': self.weights,
            'bias': self.bias,
            'batch_size': self.batch_size,
            'regularization': self.regularization,
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
        self.regularization = model_params['regularization']
        self.max_epochs = model_params['max_epochs']
        self.patience = model_params['patience']
