import numpy as np
import pandas as pd

class RFF():
    def __init__(self, gamma: float = 2, n: int = 6000):
        """
        Initializes the Random Fourier Features (RFF) object.

        Args:
            gamma (float, optional): Parameter for the random features. Defaults to 2.
            n (int, optional): Number of features. Defaults to 6000.
        """
        if not isinstance(gamma, (int, float)) or gamma <= 0:
            raise ValueError("gamma should be a positive number.")
        
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n should be a positive integer.")

        self.gamma = gamma
        self.n = n

    def features(self, data: pd.DataFrame, seed: int = None) -> np.ndarray:
        """
        Generate Random Fourier Features for the input data.

        Args:
            data (pd.DataFrame): Input data. Each row is an observation, each column is a feature.
            seed (int, optional): Seed for random number generator. Defaults to None.

        Returns:
            np.ndarray: Generated Random Fourier Features.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data should be a pandas DataFrame.")

        if seed is not None:
            np.random.seed(seed)
        
        num_features = len(data.columns)
        if num_features == 0:
            raise ValueError("data should have at least one column.")

        # Create omegas based on a normal distribution
        omegas = np.random.randn(num_features, self.n) * self.gamma

        # Calculate sine and cosine features
        sine_features = np.sin(data @ omegas)
        cosine_features = np.cos(data @ omegas)

        # Concatenate features and return
        features = np.concatenate([sine_features, cosine_features], axis=1)
        return features
