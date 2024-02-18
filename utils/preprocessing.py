import numpy as np

class ZScoreNormalization:
    """
    Z-Score Normalization class for standardizing features.

    Attributes:
        mean (numpy.ndarray): Mean values for each feature.
        std (numpy.ndarray): Standard deviation values for each feature.
    """
    
    def __init__(self):
        """
        Initializes the ZScoreNormalization object.
        """
        self.mean = None
        self.std = None
    
    def fit(self, X):
        """
        Computes the mean and standard deviation of each feature in the given dataset.

        Args:
            X (numpy.ndarray): Input dataset.

        Returns:
            tuple: A tuple containing mean and standard deviation arrays.
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
    
    def transform(self, X):
        """
        Transforms the input dataset using Z-Score normalization.

        Args:
            X (numpy.ndarray): Input dataset.

        Returns:
            numpy.ndarray: Normalized dataset.
        
        Raises:
            ValueError: If fit() method has not been called before transform().
            ValueError: If standard deviation is zero for one or more features.
        """
        if self.mean is None or self.std is None:
            raise ValueError("fit() method must be called before transform().")
        
        if np.any(self.std == 0):
            raise ValueError("Standard deviation is zero for one or more features.")
        
        X_normalized = (X - self.mean) / self.std
        return X_normalized