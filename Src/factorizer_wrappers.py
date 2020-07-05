import numpy as np
from sklearn.decomposition import NMF, FastICA, PCA
import nimfa
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# Make wrapper classes for FastICA and NMF, so we can interface to them identically
# noinspection PyPep8Naming,PyPep8Naming
class ICA_Factorizer(FastICA):
    def __init__(self, n_components=None, max_iter=5000, tol=0.000001, random_state=42,
                 fun='logcosh'):
        FastICA.__init__(self, n_components=n_components, max_iter=max_iter, tol=tol,
                         random_state=random_state, fun=fun)
        self.V = None
        self.W = None
        self.H = None
        self.recovered_V = None

    def fit(self, V, nu=None):
        assert nu is None
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
            # print(self.recovered_V)
        return self.recovered_V


# noinspection PyPep8Naming
class NMF_Factorizer(NMF):
    def __init__(self, n_components=None, max_iter=5000, tol=0.01, random_state=42):
        NMF.__init__(self, n_components=n_components, max_iter=max_iter,
                     random_state=random_state, tol=tol)
        self.V = None
        self.W = None
        self.H = None
        self.recovered_V = None

    def fit(self, V, nu1=None, nu2=None):
        assert nu1 is None and nu2 is None
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
            # print(self.recovered_V)
        return self.recovered_V


# noinspection PyPep8Naming
class Nimfa_NMF_Factorizer(object):
    """This version uses the nimfa package """

    # noinspection PyUnusedLocal
    def __init__(self, n_components=None, max_iter=200, tol=None, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state

        self.V = None
        self.W = None
        self.H = None
        self.lsnmf = None
        self.lsnmf_fit = None
        self.recovered_V = None

    def fit(self, V, nu=None):
        assert nu is None
        self.V = V
        self.lsnmf = nimfa.Bmf(self.V, seed='random_vcol', max_iter=self.max_iter,
                               rank=self.n_components)
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


# noinspection PyPep8Naming
class PCA_Factorizer(PCA):
    # noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
    def __init__(self, n_components=None, max_iter=None, tol=None, random_state=None):
        PCA.__init__(self, n_components=n_components)
        self.V = None
        self.W = None
        self.H = None
        self.recovered_V = None

    def fit(self, V, nu=None):
        assert nu is None
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
            self.recovered_V = np.dot(W, H) + self.mean_
        return self.recovered_V
