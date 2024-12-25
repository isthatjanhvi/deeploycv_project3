import numpy as np
import argparse

def generate_data(num_features, num_data_points):
    """
    Generates synthetic data for the ridge regression example.
    """
    X = np.random.rand(num_data_points, num_features)
    true_weights = np.random.rand(num_features, 1)
    y = X @ true_weights + 0.1 * np.random.randn(num_data_points, 1)  # Adding some noise
    return X, y, true_weights

def ridge_regression(X, y, lambda_param):
    """
    Performs ridge regression to compute the weights.
    """
    num_features = X.shape[1]
    identity_matrix = np.eye(num_features)
    regularization_term = lambda_param * identity_matrix
    weights = np.linalg.inv(X.T @ X + regularization_term) @ X.T @ y
    return weights

def main():
    parser = argparse.ArgumentParser(description="Perform Ridge Regression.")
    parser.add_argument('--num_features', type=int, default=3, help='Number of features (default: 3).')
    parser.add_argument('--num_data_points', type=int, default=5, help='Number of data points (default: 5).')
    parser.add_argument('--lambda_param', type=float, default=1.0, help='Regularization parameter (default: 1.0).')

    args = parser.parse_args()

    # Load arguments
    num_features = args.num_features
    num_data_points = args.num_data_points
    lambda_param = args.lambda_param

    print(f"Generating data with {num_features} features and {num_data_points} data points...")
    X, y, true_weights = generate_data(num_features, num_data_points)

    print("Performing Ridge Regression...")
    estimated_weights = ridge_regression(X, y, lambda_param)

    print("\nTrue Weights:")
    print(true_weights.flatten())

    print("\nEstimated Weights:")
    print(estimated_weights.flatten())

if __name__ == "__main__":
    main()
