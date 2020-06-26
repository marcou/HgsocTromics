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
            self._clustering = FactorClustering('Mini_Expression')
            self._clustering.read_expression_matrix()
        return self._clustering

    def test_read_expression_matrix(self):
        fc = self.clustering()
        assert fc.expression_df is not None
        assert fc.expression_matrix is not None

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
        fc = self.clustering()
        pickle_fname = fc.cached_factor_repeats_filename(NMF_Factorizer, 5, 10)
        print(pickle_fname)
        assert 'NMF' in pickle_fname

    def test_compute_and_cache_one_factor_repeats(self):
        fc = self.clustering()
        pickle_fname = fc.compute_and_cache_one_factor_repeats(
            fc.expression_matrix, ICA_Factorizer, 4, 2)
        assert os.path.exists(pickle_fname)
        metagene_list = fc.read_cached_factors(ICA_Factorizer, 4, 2)
        # Ensure there is randomness in the repeat results!
        assert not np.array_equal(metagene_list[0], metagene_list[1])

        metagene_list_2 = fc.read_cached_factors(ICA_Factorizer, 4, 2)
        assert len(metagene_list_2) == 2

    def test_single_factor_scatter(self):
        fc = self.clustering()
        n_repeats = 10
        n_components = 3
        facto = ICA_Factorizer
        fc.compute_and_cache_one_factor_repeats(
            fc.expression_matrix, facto, n_components, n_repeats, force=False)

        fc.single_factor_scatter(facto, n_components, n_repeats, show=False)

    def test_combined_factors_scatter(self):
        fc = self.clustering()
        n_repeats = 10
        fc.compute_and_cache_multiple_factor_repeats(4, 5, n_repeats=n_repeats, force=False)
        fc.combined_factors_scatter(4, n_repeats=n_repeats)

    def test_investigate_cluster_statistics(self):
        fc = self.clustering()
        fc.compute_and_cache_one_factor_repeats(
            fc.expression_matrix, NMF_Factorizer, 4, 2)
        result = fc.investigate_cluster_statistics(NMF_Factorizer, 4, 2)
        print(result)

    def test_compute_silhouette_score_and_median(self):
        n_repeats = 10
        n_components = 3
        facto_class = NMF_Factorizer
        fc = self.clustering()
        fc.compute_and_cache_one_factor_repeats(
            fc.expression_matrix, facto_class, n_components, n_repeats, force=False)
        score, median_metagenes = fc.compute_silhouette_score_and_median(
            NMF_Factorizer, n_components, n_repeats, doprint=False)
        print("Score = %8.6f" % score)
        assert 0 <= score <= 1.0
        assert median_metagenes.shape == (fc.n_genes, n_components)

    def test_save_multiple_median_metagenes_to_factors(self):
        fc = self.clustering()
        fc.save_multiple_median_metagenes_to_factors(NMF_Factorizer, start=2, end=5, n_repeats=10)

    def test_find_best_n_components(self):
        n_repeats = 10
        facto_class = NMF_Factorizer
        n_components_range = 2, 4
        fc = self.clustering()
        fc.compute_and_cache_multiple_factor_repeats(*n_components_range, n_repeats, force=False)
        fc.find_best_n_components(facto_class, *n_components_range, n_repeats,
                                  doprint=True, doshow=False)


if __name__ == '__main__':
    unittest.main()
