from os import path
import numpy as np
from sklearn.decomposition import NMF, FastICA, PCA
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Make wrapper classes for FastICA and NMF, so we can interface to them identically
class ICA_Factorizer(FastICA):
    def __init__(self, n_components=None, max_iter=200, 
                 random_state=42, fun='logcosh'):
        FastICA.__init__(self, n_components=n_components, max_iter=max_iter,
                        random_state=random_state, fun=fun)
        self.V = None
        self.W = None
        self.H = None
        self.recovered_V = None
        
    def fit(self, V):
        self.V = V
        self.W = self.fit_transform(V)
        return self
        
    def get_W(self):
        assert self.V is not None
        if self.W is None:
            self.W = self.fit_transform(self.V)
        return self.W
    
    def get_H(self):
        assert self.V is not None
        if self.H is None:
            self.H = self.mixing_.T
        return self.H
    
    def get_recovered_V(self):
        assert self.V is not None
        if self.recovered_V is None:
            W = self.get_W()
            H = self.get_H()
            mean = self.mean_
            self.recovered_V = np.dot(W, H) + mean
            #print(self.recovered_V)
        return self.recovered_V

class NMF_Factorizer(NMF):
    def __init__(self, n_components=None, max_iter=200, random_state=42):
        NMF.__init__(self, n_components=n_components, max_iter=max_iter, 
                     random_state=random_state)
        self.V = None
        self.W = None
        self.H = None
        self.recovered_V = None
        
    def fit(self, V):
        self.V = V
        self.W = self.fit_transform(V)
        return self
        
    def get_W(self):
        assert self.V is not None
        if self.W is None:
            self.W = self.fit_transform(self.V)
        return self.W
    
    def get_H(self):
        assert self.V is not None
        if self.H is None:
            self.H = self.components_
        return self.H
    
    def get_recovered_V(self):
        assert self.V is not None
        if self.recovered_V is None:
            W = self.get_W()
            H = self.get_H()
            self.recovered_V = np.dot(W, H)
            #print(self.recovered_V)
        return self.recovered_V

class PCA_Factorizer(PCA):
    def __init__(self, n_components=None, max_iter=None, random_state=None):
        PCA.__init__(self, n_components=n_components)
        self.V = None
        self.W = None
        self.H = None
        self.recovered_V = None
        
    def fit(self, V):
        self.V = V
        self.W = self.fit_transform(V)
        return self
        
    def get_W(self):
        assert self.V is not None
        if self.W is None:
            self.W = self.fit_transform(self.V)
        return self.W
    
    def get_H(self):
        assert self.V is not None
        if self.H is None:
            self.H = self.components_
        return self.H
    
    def get_recovered_V(self):
        assert self.V is not None
        if self.recovered_V is None:
            W = self.get_W()
            H = self.get_H()
            self.recovered_V = np.dot(W,H) + self.mean_
        return self.recovered_V

def example_V(n_genes=100):
    # Generate example expression matrix, useful in tests
    np.random.seed(0)
    time = np.linspace(0, 8, n_genes)

    s1 = np.sin(time) + 1.1  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time)) + 1.1  # Signal 2: square signal
    s3 = np.sin(2 * np.pi * time) + 1.1  # Signal 3: saw tooth signal
    s4 = np.cos(0.5 * np.pi * time) + 1.1  # Signal 4: cosine
    s5 = np.sin(0.2 * np.pi * time) + 1.1  # Signal 5: higher freq sine

    W = np.c_[s1, s2, s3, s4, s5]
    W += 0.1 * np.random.normal(size=W.shape)  # Add noise

    W /= W.std(axis=0)  # Standardize data
    # Mix data
    H = np.array([[1, 1, 1, 1, 1], [0.5, 0/6, 1, 1.2, 1], [1.5, 1, 2, 1, 1.1],
                 [1, 0.4, 1, 1.1, 0.1], [1, 0.2, 0.8, 1, 1.5]])  # Mixing matrix
    V = np.dot(W, H.T)  # Generate observations
    return V


def test_example_V():
    ngenes = 10
    eg_V = example_V(ngenes)
    # print(eg_V.shape)
    # print(eg_V)
    assert eg_V.shape == (10, 5)
    assert np.all(eg_V >= 0)
    print("test_example_V() passed.")

def test_Factorizer(facto, atol):
    print(facto)
    V = example_V(10)
    
    facto.fit(V)
    
    W = facto.get_W()
    assert W.shape == (V.shape[0], facto.n_components)
    
    H = facto.get_H()
    assert H.shape == (facto.n_components, V.shape[1])
    
    V2 = facto.get_recovered_V()
    assert np.allclose(V, V2, atol=atol)
        
    print("test_Factorizer (%s) passed" % type(facto).__name__)


