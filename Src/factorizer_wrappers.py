from os import path
import numpy as np
from sklearn.decomposition import NMF, FastICA, PCA
import nimfa
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Make wrapper classes for FastICA and NMF, so we can interface to them identically
class ICA_Factorizer(FastICA):
    def __init__(self, n_components=None, max_iter=200, 
                 random_state=42, fun='logcosh', algorithm='parallel'):
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
    def __init__(self, n_components=None, max_iter=200, random_state=42, tol=0.01):
        NMF.__init__(self, n_components=n_components, max_iter=max_iter, 
                     random_state=random_state, tol=tol)
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

class Nimfa_NMF_Factorizer(object):
    """This version uses the nimfa package """
    def __init__(self, n_components=None, max_iter=1000, random_state=None):
    
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.V = None
        self.W = None
        self.H = None
        self.recovered_V = None
        
    def fit(self, V):
        self.V = V
        self.lsnmf = nimfa.Bmf(self.V, seed='random_vcol', max_iter=self.max_iter, rank=self.n_components)
        self.lsnmf_fit = self.lsnmf()
        self.W = self.lsnmf_fit.basis()
        self.H = self.lsnmf_fit.coef()
        return self
        
    def get_W(self):
        assert self.W is not None
        return self.W
    
    def get_H(self):
        assert self.H is not None
        return self.H
    
    def get_recovered_V(self):
        assert self.V is not None
        if self.recovered_V is None:
            W = self.get_W()
            H = self.get_H()
            self.recovered_V = np.dot(W, H)
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




