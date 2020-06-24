import os
import unittest
import numpy as np

from gene_enrichment import GeneEnrichment


class TestGeneEnrichment(unittest.TestCase):
    # noinspection PyTypeChecker
    def setUp(self):
        self.ge = GeneEnrichment('Mini_Expression')
        np.random.seed(42)
        self.random_metagene_matrix = np.random.randn(100, 3)

        os.makedirs('../Factors/%s' % self.ge.basename, exist_ok=True)
        filename = '../Factors/%s/%s' % (self.ge.basename, 'RandomTest_3.csv')
        np.savetxt(filename, self.random_metagene_matrix)

    def test_gene_symbols(self):
        symbols = self.ge.gene_symbols()
        assert len(symbols) == 100
        print(symbols[:10])

    def test_read_metagene_matrix(self):
        mgmat = self.ge.read_metagene_matrix('RandomTest_3.csv')
        assert mgmat.ndim == 2

    def test_investigate_rank_threshold(self):
        self.ge.investigate_rank_threshold(self.random_metagene_matrix)

    def test_select_influential_genes(self):
        self.ge.select_influential_genes(self.random_metagene_matrix[:, 0])

    def test_ranked_genes_by_component(self):
        rankings = self.ge.ranked_genes_by_component(self.random_metagene_matrix)
        assert len(list(rankings)) == 3
        print(rankings[0])

    def test_perform_gene_enrichment_analysis(self):
        # The Bonferroni method is faster for testing; use 'fdr' (default) in anger
        self.ge.perform_gene_enrichment_analysis(self.random_metagene_matrix, method='bonferroni')


if __name__ == '__main__':
    unittest.main()
