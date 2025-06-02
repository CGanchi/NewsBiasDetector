import numpy as np

# linear regression without libraries
class BareBonesLinearRegression:
    def __init__(self):
        self.coefficients = None # weights
        self.bias = None # bias (or y-intercept)
        self.costs = [] # list to track cost/improvement during training

    def fit(self, X, y, learning_rate=0.01, epochs=1000, verbose=False):
        y_np = y.to_numpy()
        n_samples, n_features = X.shape
        self.coefficients = np.zeros(n_features) # initialize weights to zero
        self.bias = 0 #same with bias

        # Gradient Descent, loop through desired number of epochs
        for epoch in range(epochs):
            y_pred = X.dot(self.coefficients) + self.bias # predict y using current weights and bias
            cost = np.mean((y_pred - y_np) ** 2)  # Mean Squared Error (MSE) cost
            self.costs.append(cost) # store cost for tracking
            residual = y_pred - y_np 
            d_coefficients = (1 / n_samples) * X.T.dot(residual) # gradient for coefficients
            d_bias = (1 / n_samples) * np.sum(residual) # gradient for bias
            # Update coefficients and bias using gradients
            self.coefficients -= learning_rate * d_coefficients
            self.bias -= learning_rate * d_bias

            # Print cost every 100 epochs if verbose is True
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.6f}")

    def predict(self, X):
        return X.dot(self.coefficients) + self.bias