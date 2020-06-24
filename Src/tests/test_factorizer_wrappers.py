import unittest

import numpy as np

from factorizer_wrappers import NMF_Factorizer, ICA_Factorizer, PCA_Factorizer


def example_v(n_genes=100):
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
    H = np.array([[1, 1, 1, 1, 1], [0.5, 0 / 6, 1, 1.2, 1], [1.5, 1, 2, 1, 1.1],
                  [1, 0.4, 1, 1.1, 0.1], [1, 0.2, 0.8, 1, 1.5]])  # Mixing matrix
    V = np.dot(W, H.T)  # Generate observations
    return V


def factorizer_tst_helper(facto, atol):
    print(facto)
    V = example_v(10)

    facto.fit(V)

    W = facto.get_W()
    assert W.shape == (V.shape[0], facto.n_components)

    H = facto.get_H()
    assert H.shape == (facto.n_components, V.shape[1])

    V2 = facto.get_recovered_V()
    assert np.allclose(V, V2, atol=atol)

    print("test_Factorizer (%s) passed" % type(facto).__name__)


# noinspection PyMethodMayBeStatic,PyMethodMayBeStatic,PyMethodMayBeStatic,PyMethodMayBeStatic
class MyTestCase(unittest.TestCase):
    def test_example(self):
        ngenes = 10
        eg_V = example_v(ngenes)
        # print(eg_V.shape)
        # print(eg_V)
        assert eg_V.shape == (10, 5)
        assert np.all(eg_V >= 0)
        print("test_example_V() passed.")

    def test_NMF_Factorizer(self):
        factorizer_tst_helper(NMF_Factorizer(n_components=4), atol=0.5)
        factorizer_tst_helper(NMF_Factorizer(n_components=5), atol=0.1)

    def test_ICA_Factorizer(self):
        factorizer_tst_helper(ICA_Factorizer(n_components=4), atol=0.5)
        factorizer_tst_helper(ICA_Factorizer(n_components=5), atol=0.1)

    def test_PCA_Factorizer(self):
        factorizer_tst_helper(PCA_Factorizer(n_components=4), atol=0.5)
        factorizer_tst_helper(PCA_Factorizer(n_components=5), atol=0.1)


if __name__ == '__main__':
    unittest.main()
