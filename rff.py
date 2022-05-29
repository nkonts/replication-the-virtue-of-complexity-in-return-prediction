import numpy as np

class RFF():
    def __init__(self, **kwargs):
        self.gamma = 2
        self.n = 6000 
        
        for key, value in kwargs.items():
            setattr(self, key, value)

    def features(self, data, seed=None):
        if seed is not None:
            np.random.seed(seed)
        omegas = np.random.randn(len(data.columns), self.n) * self.gamma
        features = np.concatenate([np.sin(data @ omegas), np.cos(data @ omegas)], axis=1)
        return features
