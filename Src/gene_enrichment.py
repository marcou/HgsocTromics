# ## Gene enrichment analysis using GOATOOLS
import os

from goatools import obo_parser
from goatools.go_enrichment import GOEnrichmentStudy
import Bio.UniProt.GOA as GOA
import gzip
import numpy as np
import pandas as pd
import pickle
import wget
import mygene


# noinspection PyStringFormat,PyMethodMayBeStatic
class GeneEnrichment:
    # Analyse standard deviation of components
    def __init__(self, basename, prefix):
        self.basename = basename
        self.prefix = prefix  # prefix to results files
        self._gene_symbols = None
        self._all_ensg_ids = None
        self.cache_dir = '../Cache/%s/GeneEnrichment/' % self.basename
        os.makedirs(self.cache_dir, exist_ok=True)

    def all_ensg_ids(self):
        # We'll need to read the first row of the expression matrix file to get these
        if self._all_ensg_ids is None:
            expression_filename = '../Data/%s/%s_Expression.tsv' % (self.basename, self.basename)
            expression_df = pd.read_csv(expression_filename, sep='\t', usecols=['GeneENSG'])
            expression_df.set_index('GeneENSG', inplace=True)
            self._all_ensg_ids = expression_df.index.values.tolist()
        return self._all_ensg_ids

    def ensg_dictionary(self):
        ensgDictFile = self.cache_dir + 'ensgDict.pkl'
        if not os.path.exists(ensgDictFile):  # Run only if dictionary file does not already exist

            ensgIDs = self.all_ensg_ids()  # All the gene IDs in this study
            mg = mygene.MyGeneInfo()
            ginfo = mg.querymany(ensgIDs, scopes='ensembl.gene')

            ensgDict = {}
            for g in ginfo:
                ensg = g['query']
                del g['query']
                ensgDict[ensg] = g

            print("Writing dictionary to %s..." % ensgDictFile)
            with open(ensgDictFile, 'wb') as f:
                pickle.dump(ensgDict, f)
            print("Done.")

        with open(ensgDictFile, 'rb') as f:
            ensgDict = pickle.load(f)

        return ensgDict

    def gene_symbols(self):
        if self._gene_symbols is None:
            self.all_ensg_ids()

            ensgDict = self.ensg_dictionary()

            for (ensg, g) in ensgDict.items():
                if 'symbol' not in g.keys():
                    g['symbol'] = ensg  # ensure lookup always succeeds

            gene_ENSG_ids = self.all_ensg_ids()
            self._gene_symbols = [ensgDict[ensg]['symbol'] if ensg in ensgDict else ensg
                                  for ensg in gene_ENSG_ids]
        return self._gene_symbols

    def read_metagene_matrix(self, factor_name):
        filename = '../Factors/%s/%s' % (self.basename, factor_name)
        metagene_df = pd.read_csv(filename, sep='\t')
        metagene_df.set_index('GeneENSG', inplace=True)
        metagene_matrix = np.asarray(metagene_df)
        assert metagene_matrix.ndim == 2
        return metagene_matrix

    def investigate_rank_threshold(self, metagene_matrix):
        # Analyse standard deviation of components
        assert metagene_matrix.ndim == 2
        n_stddev = 3.0
        for ci in range(metagene_matrix.shape[1]):
            metagene = metagene_matrix[:, ci]
            selection = self.select_influential_genes(metagene)
            stddev = np.std(metagene)
            print("Component %d, SD=%4.2f, #genes outside %3.1f SDs=%d" % (
                ci, stddev, n_stddev, len(selection)))

    def select_influential_genes(self, metagene):
        assert metagene.ndim == 1
        n_stddev = 3.0
        influence = abs(metagene)
        stddev = np.std(metagene)
        mean = np.mean(metagene)
        min_ = np.min(metagene)
        threshold = n_stddev * stddev
        symbols = self.gene_symbols()
        assert len(symbols) == len(metagene)
        gixpairs = zip(symbols, influence)

        if min_ >= 0:
            # Looks like its MNF...
            selection = [symbol for (symbol, v) in gixpairs if v - mean > threshold]
        else:
            selection = [symbol for (symbol, v) in gixpairs if abs(v - mean) > threshold]

        return selection

    def ranked_genes_by_component(self, metagene_matrix):
        W = metagene_matrix
        ranked_genes_by_component = {}
        for ci in range(metagene_matrix.shape[1]):
            _genes = self.select_influential_genes(W[:, ci])
            ranked_genes_by_component[ci] = _genes
        return ranked_genes_by_component

    def _perform_gene_enrichment_analysis_one_component(self, ci, gea_results_by_component, gea):
        tsv_name = self.cache_dir + '%s_gea_C%d.tsv' % (self.prefix, ci)
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

    def cache_downloaded_resources(self):
        download_directory = '../DownloadedResources/'
        os.makedirs(download_directory, exist_ok=True)

        url_list = [
            'http://geneontology.org/gene-associations/goa_human.gaf.gz',
            'http://purl.obolibrary.org/obo/go/go-basic.obo']

        for url in url_list:
            print("Downloading resource from %s ..." % url)
            if not os.path.exists(os.path.join(download_directory, os.path.basename(url))):
                wget.download(url, out=download_directory)

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
            print('Comp. %d: %s...' % (ci, str(study_genes[:10])))
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

        gea_all_sig_results_df.to_csv(self.cache_dir + '%s_gea_all.tsv' % self.prefix, sep='\t')


# noinspection PyUnreachableCode
def main():
    ge = GeneEnrichment('TCGA_OV_VST', 'NMF_3')
    metagenes = ge.read_metagene_matrix('NMF_median_factor_3.tsv')
    ge.perform_gene_enrichment_analysis(metagenes, method='bonferroni')

    ge = GeneEnrichment('TCGA_OV_VST', 'ICA_3')
    metagenes = ge.read_metagene_matrix('ICA_median_factor_3.tsv')
    ge.perform_gene_enrichment_analysis(metagenes, method='bonferroni')

    ge = GeneEnrichment('TCGA_OV_VST', 'PCA_3')
    metagenes = ge.read_metagene_matrix('PCA_median_factor_3.tsv')
    ge.perform_gene_enrichment_analysis(metagenes, method='bonferroni')


if __name__ == '__main__':
    main()
