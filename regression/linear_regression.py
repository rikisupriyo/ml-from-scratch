import numpy as np
import time
import math

class LinearRegression:
    def __init__(self, learning_rate = 1e-3, n_iters = 1000):
        """
        Constructor for the LinearRegression class.

        Parameters:
        - learning_rate: The learning rate for gradient descent.
        - n_iters: The number of iterations for gradient descent.
        """
        # Initialize hyperparameters
        self.learning_rate = learning_rate
        self.n_iters = n_iters

        # Initialize weights and bias
        self.weights = None
        self.bias = None

    def __compute_gradient(self, X, y):
        """
        Private method to compute the gradient of the cost function with respect to weights and bias.

        Parameters:
        - X: Feature matrix.
        - y: Target variable.

        Returns:
        - dj_dw: Gradient of weights.
        - dj_db: Gradient of bias.
        - j_wb: Cost value.
        """
        n_samples, _ = X.shape
        error = np.dot(X, self.weights) + self.bias - y

        # Calculate the cost (mean squared error)
        j_wb =  np.sum(error ** 2) / (2 * n_samples)

        # Calculate the gradients
        dj_dw = np.dot(X.T, error) / n_samples
        dj_db = np.sum(error) / n_samples

        return dj_dw, dj_db, j_wb

    def fit(self, X, y):
        """
        Fit the linear regression model to the given data using gradient descent.

        Parameters:
        - X: Feature matrix.
        - y: Target variable.
        """
        _, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros((n_features,))
        self.bias = 0

        # Record the start time for measuring elapsed time during iterations
        start_time = time.time()

        for i in range(self.n_iters):
            # Compute gradients and cost
            dj_dw, dj_db, j_wb = self.__compute_gradient(X, y)

            # Update weights and bias using gradient descent
            self.weights -= self.learning_rate * dj_dw
            self.bias -= self.learning_rate * dj_db

            # Print progress every 10% of iterations
            if i % math.ceil(self.n_iters / 10) == 0:
                elapsed_time = time.time() - start_time
                print(f"Iteration --> {i:5} | Cost --> {float(j_wb):8.2f} | Elapsed Time --> {elapsed_time:.2f} seconds")

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X: Feature matrix for prediction.

        Returns:
        - y_hat: Predicted values.
        """
        # Check if the model is trained
        if self.weights is None or self.bias is None:
            raise ValueError("fit() method must be called before predict().")

        # Calculate predictions using the linear regression model
        y_hat = np.dot(X, self.weights) + self.bias

        return y_hat