import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)
    def fit_plot(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        m = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01

        # Lists to store the number of iterations and corresponding loss
        iterations_list = []
        loss_list = []

        # Implement gradient descent
        for epoch in range(iterations):
            predictions = np.dot(X, np.hstack([self.intercept, self.coefficients]))
            error = predictions - y

            # Compute gradients
            gradient = (1 / m) * np.dot(np.transpose(X), error)            
            
            # Update parameters
            self.intercept -= learning_rate * gradient[0]
            self.coefficients -= learning_rate * gradient[1:]

            # Calculate and store the loss every 10 epochs
            if epoch % 10 == 0:
                mse = np.mean(error ** 2)
                iterations_list.append(epoch)
                loss_list.append(mse)
                print(f"Epoch {epoch}: MSE = {mse}")

        # Plot the loss over iterations
        plt.plot(iterations_list, loss_list)
        plt.xlabel('Iterations')
        plt.ylabel('Mean Squared Error')
        plt.title('Loss over Iterations')
        plt.show()
    def fit_plot_compare(self, X, y, learning_rate=0.01, iterations=1000, w_optimo = 0, b_optimo = 0):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        m = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01

        # Lists to store the number of iterations and corresponding coefficients and intercept
        iterations_list = []
        coefficients_list = []
        intercept_list = []

        # Implement gradient descent
        for epoch in range(iterations):
            predictions = np.dot(X, np.hstack([self.intercept, self.coefficients]))
            error = predictions - y

            # Compute gradients
            gradient = (1 / m) * np.dot(np.transpose(X), error)            
            
            # Update parameters
            self.intercept -= learning_rate * gradient[0]
            self.coefficients -= learning_rate * gradient[1:]

            # Store the coefficients and intercept every 10 epochs
            if epoch % 10 == 0:
                iterations_list.append(epoch)
                print(self.coefficients[0])
                coefficients_list.append(self.coefficients[0])
                intercept_list.append(self.intercept)
                print(f"Epoch {epoch}: Coefficients = {self.coefficients[0]}, Intercept = {self.intercept}")

        # Plot the coefficient (x-axis) versus the intercept (y-axis) over iterations
        plt.plot(coefficients_list, intercept_list, label='Coefficient vs Intercept')
        plt.scatter([w_optimo], [b_optimo], color='red', marker='x', label='Optimal Point')
        plt.xlabel('Coefficient')
        plt.ylabel('Intercept')
        plt.title('Coefficient vs Intercept over Iterations')
        plt.legend()
        plt.show()
    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Replace this code with the code you did in the previous laboratory session

        # Store the intercept and the coefficients of the model
        X_bias = X
        beta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y

        self.intercept = beta[0]
        self.coefficients = beta[1:]

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        # Initialize the parameters to very small values (close to 0)
        m = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01
        # Implement gradient descent (TODO)
        for epoch in range(iterations):
            predictions = np.dot(X, np.hstack([self.intercept, self.coefficients]))
            error = predictions - y

            # TODO:
            # Compute gradients
            gradient = (1 / m) * np.dot(np.transpose(X), error)            
            
            # Update parameters
            self.intercept -= learning_rate * gradient[0]
            self.coefficients -= learning_rate * gradient[1:]


            # TODO: Calculate and print the loss every 10 epochs
            if epoch % 1000 == 0:
                mse = np.mean(error ** 2)
                print(f"Epoch {epoch}: MSE = {mse}")

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            # TODO: Predict when X is only one variable
            predictions = self.intercept + self.coefficients * X
        else:
            # TODO: Predict when X is more than one variable
            predictions = self.intercept + X @ self.coefficients
        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    # R^2 Score
    # TODO
    # R^2 Score
    # TODO: Calculate R^2
    rss = np.sum((y_true - y_pred) ** 2) #calculate residual sum of squares
    tss = np.sum((y_true - np.mean(y_true)) ** 2) #calculate total sum of squares
    r_squared = 1 - rss / tss

    # Root Mean Squared Error
    # TODO: Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Mean Absolute Error
    # TODO: Calculate MAE
    mae = np.mean(np.abs(y_true - y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}



def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()
    for index in sorted(categorical_indices, reverse=True):
        # TODO: Extract the categorical column
        categorical_column = X_transformed[:, index]

        # TODO: Find the unique categories (works with strings)
        unique_values = np.unique(categorical_column)

        # TODO: Create a one-hot encoded matrix (np.array) for the current categorical column
        one_hot = []
        for category in unique_values:
            one_hot_column = (categorical_column == category).astype(int)
            one_hot.append(one_hot_column)
        one_hot = np.array(one_hot).T

        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]

        # TODO: Delete the original categorical column from X_transformed and insert new one-hot encoded columns
        X_transformed = np.delete(X_transformed, index, axis=1)
        for i in range(one_hot.shape[1]):
            X_transformed = np.insert(X_transformed, index + i, one_hot[:, i], axis=1)

    return X_transformed
