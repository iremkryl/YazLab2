import numpy as np


def add_gaussian_noise(X, mean: float = 0.0, std: float = 0.05, seed: int = 42):
    """
    Veriye Gaussian noise ekler.

    Bu fonksiyon, gürültülü veri senaryosu için kullanılacaktır.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=mean, scale=std, size=X.shape)

    return X + noise