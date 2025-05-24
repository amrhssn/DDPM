import numpy as np
from sklearn.utils import check_random_state
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def gaussian_mixture_pdf(X, means, covariances, weights):
    """
    Compute the probability density function of a Gaussian mixture model.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
    means : array-like of shape (n_components, n_features)
        The component means.
    covariances : array-like of shape (n_components, n_features, n_features)
        The component covariance matrices.
    weights : array-like of shape (n_components,)
        The component weights, should sum to 1.

    Returns:
    --------
    pdf : array-like of shape (n_samples,)
        The PDF values for each sample.
    """
    n_components = len(weights)
    n_samples = X.shape[0]
    pdf = np.zeros(n_samples)

    for i in range(n_components):
        pdf += weights[i] * multivariate_normal.pdf(X, mean=means[i], cov=covariances[i])

    return pdf


def make_gaussian_mixture(
        n_samples=100,
        n_features=2,
        means=None,
        covariances=None,
        weights=None,
        return_pdf=False,
        random_state=None):
    """
    Generate a random dataset from a mixture of Gaussians.

    Parameters:
    -----------
    n_samples : int, default=100
        The number of samples.
    n_features : int, default=2
        The number of features.
    means : array-like of shape (3, n_features), default=None
        The means of the Gaussian components. If None, random means are generated.
    covariances : array-like of shape (3, n_features, n_features), default=None
        The covariance matrices. If None, random covariance matrices are generated.
    weights : array-like of shape (3,), default=None
        The mixture weights. If None, equal weights are used.
    return_pdf : bool, default=False
        If True, return the PDF values for each sample.
    random_state : int or RandomState, default=None
        Determines random number generation for dataset creation.

    Returns:
    --------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.
    y : ndarray of shape (n_samples,)
        The integer labels (0, 1, or 2) for class membership of each sample.
    pdf_values : ndarray of shape (n_samples,), optional
        The PDF values for each sample. Only returned if return_pdf is True.
    """
    random_state = check_random_state(random_state)

    # Set default values if not provided
    if means is None:
        means = random_state.uniform(-10, 10, size=(3, n_features))

    if covariances is None:
        covariances = []
        for _ in range(3):
            # Generate a random positive definite matrix
            A = random_state.normal(0, 1, size=(n_features, n_features))
            cov = np.dot(A, A.T) + np.eye(n_features)  # Ensure positive definiteness
            covariances.append(cov)
        covariances = np.array(covariances)

    if weights is None:
        weights = np.ones(3) / 3  # Equal weights
    else:
        weights = np.array(weights) / np.sum(weights)  # Normalize weights

    # Generate class labels based on weights
    y = random_state.choice(3, size=n_samples, p=weights)

    # Generate samples for each class
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        component = y[i]
        X[i] = random_state.multivariate_normal(means[component], covariances[component])

    if return_pdf:
        pdf_values = gaussian_mixture_pdf(X, means, covariances, weights)
        return X, y, pdf_values
    else:
        return X, y
