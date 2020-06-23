import os
import unittest
from unittest import TestCase

import numpy as np

from factor_clustering import FactorClustering
from factorizer_wrappers import ICA_Factorizer, NMF_Factorizer


class TestFactorClustering(TestCase):
    def setUp(self):
        self._clustering = None

    def clustering(self):
        if self._clustering is None:
            self._clustering = FactorClustering()
            self._clustering.read_expression_matrix('../Data/Mini_Expression.csv')
        return self._clustering

    def test_read_expression_matrix(self):
        assert self.clustering().expression_df is not None
        assert self.clustering().expression_matrix is not None

    def test_l2_norm_diff(self):
        m1 = np.array([0, 0, 1])
        m2 = np.array([0, 0, 1])
        m3 = np.array([1, 1, 0])

        assert FactorClustering.l2_norm_diff(m1, m2) == 0
        assert np.isclose(FactorClustering.l2_norm_diff(m1, m3), 1)

    def test_calc_angle(self):
        m1 = np.array([0, 0, 1])
        m2 = np.array([0, 0, 1])
        m3 = np.array([0, 1, 0])
        assert np.isclose(FactorClustering.calc_angle(m1, m2), 0)
        assert np.isclose(FactorClustering.calc_angle(m1, m3), 90)

    def test_demonstrate_angles_in_high_dimensions(self):
        FactorClustering.demonstrate_angles_in_high_dimensions(1000, 100)

    def test_cached_factor_repeats_filename(self):
        pickle_fname = self.clustering().cached_factor_repeats_filename(NMF_Factorizer, 5, 10)
        print(pickle_fname)
        assert 'NMF' in pickle_fname

    def test_compute_and_cache_one_factor_repeats(self):
        pickle_fname = self.clustering().compute_and_cache_one_factor_repeats(
            self.clustering().expression_matrix, ICA_Factorizer, 4, 2)
        assert os.path.exists(pickle_fname)
        metagene_list = self.clustering().read_cached_factors(ICA_Factorizer, 4, 2)
        # Ensure there is randomness in the repeat results!
        assert not np.array_equal(metagene_list[0], metagene_list[1])

        metagene_list_2 = self.clustering().read_cached_factors(ICA_Factorizer, 4, 2)
        assert len(metagene_list_2) == 2

    def test_combined_factors_scatter(self):
        self.clustering().compute_and_cache_multiple_factor_repeats(4, 5, n_repeats=2)
        self.clustering().combined_factors_scatter(4, n_repeats=2)

    def test_investigate_cluster_statistics(self):
        self.clustering().compute_and_cache_one_factor_repeats(
            self.clustering().expression_matrix, NMF_Factorizer, 4, 2)
        result = self.clustering().investigate_cluster_statistics(NMF_Factorizer, 4, 2)
        print(result)


if __name__ == '__main__':
    unittest.main()
