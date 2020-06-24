# ## Gene enrichment analysis using GOATOOLS

from goatools import obo_parser
from goatools.go_enrichment import GOEnrichmentStudy
import Bio.UniProt.GOA as GOA
import gzip
import numpy as np
import pandas as pd
import pickle


# noinspection PyStringFormat
class GeneEnrichment:
    # Analyse standard deviation of components
    def __init__(self, basename):
        self.basename = basename
        self._gene_symbols = None
        self.cache_dir = '../Cache/%s/' % self.basename

    def gene_symbols(self):
        if self._gene_symbols is None:
            expression_filename = '../Data/%s.csv' % self.basename
            expression_df = pd.read_csv(expression_filename, sep='\t', usecols=['GeneENSG'])
            # expression_df.set_index('GeneENSG', inplace=True)
            ensgDictFile = '../Cache/ensgDict.pkl'
            with open(ensgDictFile, 'rb') as f:
                ensgDict = pickle.load(f)
            for (ensg, g) in ensgDict.items():
                if 'symbol' not in g.keys():
                    g['symbol'] = ensg  # ensure lookup always succeeds

            gene_ENSG_ids = expression_df['GeneENSG']
            self._gene_symbols = [ensgDict[ensg]['symbol'] for ensg in gene_ENSG_ids]
        return self._gene_symbols

    def read_metagene_matrix(self, factor_name):
        filename = '../Factors/%s/%s' % (self.basename, factor_name)
        metagene_matrix = np.loadtxt(filename)
        assert metagene_matrix.ndim == 2
        return metagene_matrix

    @staticmethod
    def investigate_rank_threshold(metagene_matrix):
        # Analyse standard deviation of components
        assert metagene_matrix.ndim == 2
        n_stddev = 3.0
        for ci in range(metagene_matrix.shape[1]):
            metagene = metagene_matrix[:, ci]
            stddev = np.std(metagene)
            threshold = n_stddev * stddev
            num_above_threshold = len(metagene[abs(metagene) > threshold])
            print("Component %d, SD=%4.2f, #genes outside %3.1f SDs=%d" % (
                ci, stddev, n_stddev, num_above_threshold))

    def select_influential_genes(self, metagene):
        assert metagene.ndim == 1
        n_stddev = 3.0
        influence = abs(metagene)
        stddev = np.std(metagene)
        threshold = n_stddev * stddev
        symbols = self.gene_symbols()
        assert len(symbols) == len(metagene)
        gixpairs = zip(symbols, influence)

        selection = [symbol for (symbol, v) in gixpairs if abs(v) > threshold]

        return selection

    def ranked_genes_by_component(self, metagene_matrix, oneperline=False):
        W = metagene_matrix
        ranked_genes_by_component = {}
        for ci in range(metagene_matrix.shape[1]):
            _genes = self.select_influential_genes(W[:, ci])
            ranked_genes_by_component[ci] = _genes
            if oneperline:
                print("Comp. %d: \n%s\n" % (ci, '\n'.join(_genes)))
            else:
                print("Comp. %d: %s" % (ci, ' '.join(_genes)))
        return ranked_genes_by_component

    def _perform_gene_enrichment_analysis_one_component(self, ci, gea_results_by_component, gea):
        tsv_name = self.cache_dir + 'goa_results_C%d.tsv' % ci
        if len(gea_results_by_component[ci]) > 0:
            with open(tsv_name, 'w') as f:
                gea.prt_tsv(f, gea_results_by_component[ci])
            ge_df = pd.read_csv(tsv_name, sep='\t')

            ge_df.rename(columns={'# GO': 'GO_ID'}, inplace=True)
            ge_df.set_index('GO_ID', inplace=True)
            ge_df.drop(columns=['NS', 'enrichment', 'p_uncorrected'], inplace=True)
            if 'p_fdr' in ge_df.columns:
                ge_df = ge_df[ge_df['p_fdr'] <= 0.05]
            else:
                ge_df = ge_df[ge_df['p_bonferroni'] <= 0.05]
            ge_df['Component'] = ci
            return ge_df
        else:
            return None

    def perform_gene_enrichment_analysis(self, metagene_matrix, method='fdr'):
        # Load the Gene Ontology
        n_comps = metagene_matrix.shape[1]
        gene_ontology = obo_parser.GODag('../DownloadedResources/go-basic.obo')

        # Load the human annotations
        c = 0
        with gzip.open('../DownloadedResources/goa_human.gaf.gz', 'rt') as gaf:
            funcs = {}
            for entry in GOA.gafiterator(gaf):
                c += 1
                uniprot_id = entry.pop('DB_Object_Symbol')
                funcs[uniprot_id] = entry

        # Our population is the set of genes we are analysing

        population = self.gene_symbols()
        print("We have %d genes in our population" % len(population))

        # Build associations from functional annotations we got from the gaf file
        associations = {}
        for x in funcs:
            if x not in associations:
                associations[x] = set()
            associations[x].add(str(funcs[x]['GO_ID']))

        gea = GOEnrichmentStudy(population, associations, gene_ontology,
                                propagate_counts=True,
                                alpha=0.05,
                                methods=[method])
        gea_results_by_component = {}
        rankings = self.ranked_genes_by_component(metagene_matrix)
        for ci in range(n_comps):
            study_genes = rankings[ci]
            gea_results_by_component[ci] = gea.run_study(study_genes)

        # Get results into a dataframe per component.  Easiest way is to use routine to
        # write a .tsv file, then read back and filter

        gea_results_df_by_component = []
        for ci in range(n_comps):
            ge_df = self._perform_gene_enrichment_analysis_one_component(
                ci, gea_results_by_component, gea)
            if ge_df is not None:
                gea_results_df_by_component += [ge_df]

        # Merge the per-component dataframes into a single one
        gea_all_sig_results_df = pd.DataFrame()
        gea_all_sig_results_df = gea_all_sig_results_df.append(gea_results_df_by_component)
        gea_all_sig_results_df.to_csv(self.cache_dir + 'gea_all_sig_results.tsv', sep='\t')


# noinspection PyUnreachableCode
def main():
    if False:
        ge = GeneEnrichment('Mini_Expression')
        metagenes = ge.read_metagene_matrix('RandomTest_3.csv')
    else:
        ge = GeneEnrichment('HGSOC_Protein_Expression')
        metagenes = ge.read_metagene_matrix('S_HGSOC_Protein_Expression_ica_numerical.txt_6.num')

    ge.perform_gene_enrichment_analysis(metagenes)


if __name__ == '__main__':
    main()
