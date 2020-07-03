import os
import unittest
import numpy as np
import pandas as pd

from gene_enrichment import GeneEnrichment


class TestGeneEnrichment(unittest.TestCase):
    # noinspection PyTypeChecker
    def setUp(self):
        self.ge = GeneEnrichment('Mini_Test', 'DUMMY')
        np.random.seed(42)

        nc = 3
        self.random_metagene_matrix = np.random.randn(100, nc)

        os.makedirs('../Factors/%s' % self.ge.basename, exist_ok=True)
        filename = '../Factors/%s/%s' % (self.ge.basename, 'RandomTest_3.tsv')

        # We want to write a .tsv file with ENSG ids in the first column, then
        # columns for the nc components
        columns = ['IC%d' % c for c in range(nc)]
        factor_df = pd.DataFrame(data=self.random_metagene_matrix, columns=columns)
        expression_df = pd.read_csv('../Data/Mini_Test/Mini_Test_Expression.tsv',
                                    sep='\t', usecols=['GeneENSG'])
        factor_df['GeneENSG'] = expression_df.index
        factor_df.set_index('GeneENSG', inplace=True)
        assert len(factor_df.columns) == nc
        factor_df.to_csv(filename, sep='\t')

    def test_gene_symbols(self):
        symbols = self.ge.gene_symbols()
        assert len(symbols) == 100
        print(symbols[:10])

    def test_read_metagene_matrix(self):
        mgmat = self.ge.read_metagene_matrix('RandomTest_3.tsv')
        assert mgmat.ndim == 2
        assert mgmat.shape == (100, 3)

    def test_investigate_rank_threshold(self):
        self.ge.investigate_rank_threshold(self.random_metagene_matrix)

    def test_select_influential_genes(self):
        # NMF type example
        positive_metagene = np.random.randn(100) + 2.0
        positive_metagene[positive_metagene < 0] = 0
        positive_metagene[[0, 10]] = 123.0  # big number for two genes
        selection = self.ge.select_influential_genes(positive_metagene)
        assert len(selection) == 2
        assert selection[0] == self.ge.gene_symbols()[0]
        assert selection[1] == self.ge.gene_symbols()[10]

        # ICA or PCA type example
        mixed_metagene = np.random.randn(100)
        mixed_metagene[0] = 123.0
        mixed_metagene[10] = -123.0
        selection = self.ge.select_influential_genes(mixed_metagene)
        assert len(selection) == 2
        assert selection[0] == self.ge.gene_symbols()[0]
        assert selection[1] == self.ge.gene_symbols()[10]

    def test_ranked_genes_by_component(self):
        rankings = self.ge.ranked_genes_by_component(self.random_metagene_matrix)
        assert len(list(rankings)) == 3
        print(rankings[0])

    def test_perform_gene_enrichment_analysis(self):
        # The Bonferroni method is faster for testing; use 'fdr' (default) in anger
        self.ge.perform_gene_enrichment_analysis(self.random_metagene_matrix, method='bonferroni')


if __name__ == '__main__':
    unittest.main()
