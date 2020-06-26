# coding: utf-8

# # Compare factors delivered by the three methods
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ## Factorizer classes
# These have now been hived off into ``factorizer_wrappers.py``.  Import and test them.

from factorizer_wrappers import ICA_Factorizer, NMF_Factorizer, PCA_Factorizer


# noinspection PyStringFormat,PyMethodMayBeStatic
class FactorClustering:
    # ## Read the expression matrix
    # This is repeated code, should be factored out...
    def __init__(self):
        self.expression_df = None
        self.expression_matrix = None
        self.expression_filename = None
        self.n_genes = None
        self.n_patients = None

    def read_expression_matrix(self, expression_filename='../Data/HGSOC_Protein_Expression.csv'):
        # Read in expression spreadsheet which has been processed (see end of notebook)
        # to inlcude only protein coding genes

        self.expression_filename = expression_filename
        self.expression_df = pd.read_csv(expression_filename, sep='\t')
        self.expression_df.set_index('GeneENSG', inplace=True)
        # assert len(self.expression_df) == 19730  # Only
        # assert len(self.expression_df.columns) == 80
        # assert self.expression_df.columns[-1] == 'AOCS_171'
        # TODO: is this right?
        self.expression_matrix = normalize(np.asarray(self.expression_df))
        self.n_genes, self.n_patients = self.expression_matrix.shape
        print(self.n_genes, "genes")
        print(self.n_patients, "patients")

    @staticmethod
    def l2_norm_diff(m1, m2):
        return np.sqrt(np.mean((m1 - m2) ** 2))

    # Angle calculation
    @staticmethod
    def calc_angle(v1, v2, show=False):
        dotp = np.dot(v1, v2)
        v1_mag = np.sqrt(np.sum(v1 * v1))
        v2_mag = np.sqrt(np.sum(v2 * v2))
        costheta = dotp / (v1_mag * v2_mag)

        angleRad = np.arccos(min(costheta, 1.0))
        angleDeg = angleRad * (180 / np.pi)

        if show:
            print("v1:\n")
            print(v1)
            print("\nv2:")
            print(v2)
            print("\nv1 Mag.:%6.4f" % v1_mag)
            print("\nv2 Mag.:%6.4f" % v2_mag)
            print("v1 . v2 = %6.4f" % dotp)
            print(dotp / (v1_mag * v2_mag))
            print("Angle between v1 and v2 = %5.1f degrees." % angleDeg)
        return angleDeg

    # ## Angle of vectors in a high dimensioned space
    @staticmethod
    def demonstrate_angles_in_high_dimensions(dims=50000, n=1000):
        # Demonstrating that in a 20,000 dimensioned space, any two random vectors will be at
        # very close to 90 degrees!
        alist = []
        rvs = np.random.randn(n, dims)
        for i in range(n - 1):
            v1 = rvs[i, :]
            v2 = rvs[i + 1, :]
            a = FactorClustering.calc_angle(v1, v2)
            alist += [a]

        plt.hist(alist, bins=int(np.sqrt(n)))
        plt.title("Mean=%6.2f, SD=%6.2f degrees" % (np.mean(alist), np.std(alist)))
        # plt.show()

    def cached_factor_repeats_filename(self, facto_class, n_components, n_repeats):
        # ## Multiple cached runs of NMF and ICA
        # Run NMF and ICA for a range of components, with repeats and save into .pkl fles
        # for later use.

        basename = Path(self.expression_filename).stem
        pickle_fname = "../Cache/%s/FactorClustering/%s_%d_%d.pkl" % \
                       (basename, facto_class.__name__, n_components, n_repeats)
        return pickle_fname

    def read_cached_factors(self, facto_class, n_components, n_repeats):
        pickle_fname = self.cached_factor_repeats_filename(
            facto_class, n_components, n_repeats)
        with open(pickle_fname, 'rb') as f:
            metagene_list = pickle.load(f)
        return metagene_list

    def compute_and_cache_one_factor_repeats(self, V, facto_class, n_components, n_repeats,
                                             force=True):
        pickle_fname = self.cached_factor_repeats_filename(
            facto_class, n_components, n_repeats)
        p = Path(pickle_fname)
        os.makedirs(p.parent, exist_ok=True)
        if force or not os.path.exists(pickle_fname):
            print(pickle_fname)
            metagene_list = []
            for i in range(n_repeats):
                facto = facto_class(n_components=n_components, max_iter=5000, tol=0.01,
                                    random_state=np.random.randint(10000))
                facto.fit(V)
                metagene_list += [facto.get_W()]
                print('\r%d/%d' % (i + 1, n_repeats), end='')
            print()
            with open(pickle_fname, 'wb') as f:
                pickle.dump(metagene_list, f)
        return pickle_fname

    def compute_and_cache_multiple_factor_repeats(self, start, end, n_repeats=50, force=True):
        # This will take several hours, if enabled!
        if True:
            V = self.expression_matrix
            for nc in range(start, end):
                self.compute_and_cache_one_factor_repeats(V, NMF_Factorizer, nc, n_repeats, force)
                self.compute_and_cache_one_factor_repeats(V, ICA_Factorizer, nc, n_repeats, force)

            print("All Done.")

    def single_factor_scatter(self, facto_class, n_components, n_repeats,
                              show=True, pca_reduced_dims=20):
        # t-SNE plot for on factorizer, on component

        metagene_list = self.read_cached_factors(facto_class, n_components, n_repeats)
        score, median_metagenes = self.compute_silhouette_score_and_median(
            facto_class, n_components, n_repeats, pca_reduced_dims=pca_reduced_dims)

        stacked_metagenes = np.hstack(metagene_list + [median_metagenes]).T

        flipped_metagenes = self.flip_metagenes(stacked_metagenes)

        # Reduce to a managable number of dimensions before passing to t-SNE
        pca = PCA(n_components=min(pca_reduced_dims, len(flipped_metagenes)))
        tsne = TSNE(n_components=2, init='pca', n_jobs=7)
        Y = tsne.fit_transform(pca.fit_transform(flipped_metagenes))

        # Plot the t-SNE projections
        assert Y.shape[0] == n_components * n_repeats + n_components

        # First plot the scattered points
        plt.scatter(Y[:n_components * n_repeats, 0], Y[:n_components * n_repeats, 1],
                    s=3, label='one sample')

        # Then plot the medians
        plt.scatter(Y[n_components * n_repeats:, 0], Y[n_components * n_repeats:, 1],
                    s=50, marker='+', label='cluster median')

        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.legend()
        plt.title("%s; nc=%d; silhouette s.=%6.4f; pca_dims=%d" %
                  (facto_class.__name__[:3], n_components, score, pca_reduced_dims))
        if show:
            plt.show()

    def flip_metagenes(self, stacked_metagenes):
        """ For ICA there is a problem that a metagenes can be oriented 180 from each other
        but are effectively the same.  Here we crudely try to normalise to one orientation -
        more thought it needed"""
        return [g if g[0] >= 0 else -g for g in stacked_metagenes[:]]

    def combined_factors_scatter(self, n_components, n_repeats, pca_reduced_dims=20):
        # ## t-SNE plots of NMF, ICA and PCA components
        # Interesting to see the components generated by the three methods ploted in the same space.
        # However, we must beware of over-interpeting t-SNE plots...

        # Read back the pickle files containing multiple runs. One file for each n_components
        # for each of NMF and ICA

        nmf_metagene_list = self.read_cached_factors(NMF_Factorizer, n_components, n_repeats)
        ica_metagene_list = self.read_cached_factors(ICA_Factorizer, n_components, n_repeats)

        # Add result of PCA analysis for same number of components
        pca_facto = PCA_Factorizer(n_components=n_components)
        pca_facto.fit(self.expression_matrix)
        pca_metagenes = pca_facto.get_W()

        stacked_metagenes = np.hstack(nmf_metagene_list + ica_metagene_list + [pca_metagenes]).T

        # For ICA at least, we see double the expected number of components, due to the arbitrary
        # direction of the vector.  So flip them into the same overall direction
        flipped_metagenes = self.flip_metagenes(stacked_metagenes)

        # Reduce to a managable number of dimensions before passing to t-SNE
        pca = PCA(n_components=min(pca_reduced_dims, len(flipped_metagenes)))
        tsne = TSNE(n_components=2, init='pca', n_jobs=7)
        Y = tsne.fit_transform(pca.fit_transform(flipped_metagenes))

        # Plot the t-SNE projections in two halves so that NMF and ICA show in different colours
        assert Y.shape[0] == 2 * n_components * n_repeats + n_components
        # Start indices of the components in Y
        nmf_Y = Y[0:n_components * n_repeats, :]
        ica_Y = Y[n_components * n_repeats: 2 * n_components * n_repeats, :]
        pca_Y = Y[2 * n_components * n_repeats:, :]

        plt.scatter(nmf_Y[:, 0], nmf_Y[:, 1], s=3, label='NMF')
        plt.scatter(ica_Y[:, 0], ica_Y[:, 1], s=3, label='ICA')
        plt.scatter(pca_Y[:, 0], pca_Y[:, 1], s=50, marker='+', label='PCA')

        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.legend()
        plt.title("Components: %d" % n_components)

    def plot_multiple_combined_factors_scatter(self, start_comp, end_comp, n_repeats=50):
        plt.figure(figsize=(20, 20))
        for nc in range(start_comp, end_comp):
            print('.', end='')
            plt.subplot(4, 3, nc - start_comp + 1)
            self.combined_factors_scatter(nc, n_repeats)
        plt.suptitle("t-SNE clustering for %d repeats of NMF and ICA", size=14)
        plt.show()

    def plot_multiple_single_factors_scatter(self, facto_class, start_comp, end_comp, n_repeats=50):
        plt.figure(figsize=(16, 20))
        for nc in range(start_comp, end_comp):
            print('.', end='')
            plt.subplot(4, 3, nc - start_comp + 1)
            self.single_factor_scatter(facto_class, nc, n_repeats, show=False)
        plt.suptitle("t-SNE clustering for %d repeats of %s" %
                     (n_repeats, facto_class.__name__[:3]), size=14)
        plt.show()

    def investigate_cluster_statistics(self, facto_class, n_components, n_repeats,
                                       pca_reduced_dims=20, doprint=False):
        # The given facto is not actually executed, just used to select the appropriate cached
        # .pkl files which were computed above.
        metagene_list = self.read_cached_factors(facto_class, n_components, n_repeats)
        stacked_metagenes = np.hstack(metagene_list).T
        flipped_metagenes = self.flip_metagenes(stacked_metagenes)

        pca = PCA(n_components=min(pca_reduced_dims, len(metagene_list)))
        kmeans = KMeans(n_clusters=n_components, random_state=0).fit(
            pca.fit_transform(flipped_metagenes))
        cluster_table = np.reshape(kmeans.labels_, (n_repeats, n_components))
        clusters_are_aligned = np.all(
            [cluster_table[r, :] == cluster_table[0, :] for r in range(n_repeats)])
        if doprint:
            for r in range(n_repeats):
                print(r, cluster_table[r, :])
            print()
        return clusters_are_aligned

    def investigate_multiple_cluster_statistics(self, start, end, n_repeats=50):
        # Lets see if clusters are assined consistently for NMF and ICA across a
        # range of n_components
        print("%6s %10s %10s" % ('', NMF_Factorizer.__name__, ICA_Factorizer.__name__))
        for nc in range(start, end):
            nmf_consistent = self.investigate_cluster_statistics(NMF_Factorizer, nc, n_repeats)
            ica_consistent = self.investigate_cluster_statistics(ICA_Factorizer, nc, n_repeats)
            print("%6d%10s %10s" % (nc, nmf_consistent, ica_consistent))

    def compute_silhouette_score_and_median(self, facto_class, n_components, n_repeats,
                                            pca_reduced_dims=10, doprint=False):

        # Get repeated metagenes for this n_components into a matrix of shape
        # (n_components*n_repeats, genes)
        metagene_list = self.read_cached_factors(facto_class, n_components, n_repeats)
        assert metagene_list[0].shape[0] == self.n_genes
        stacked_metagenes = np.hstack(metagene_list).T
        flipped_metagenes = np.array(self.flip_metagenes(stacked_metagenes))
        assert flipped_metagenes.shape[0] == n_components * n_repeats

        # Run k-means clustering, but reduce to a sensible number of dimensions first
        pca = PCA(n_components=min(pca_reduced_dims, len(metagene_list)))
        kmeans = KMeans(n_clusters=n_components, random_state=0).fit(
            pca.fit_transform(flipped_metagenes))
        cluster_labels = kmeans.fit_predict(flipped_metagenes)
        cluster_label_table = np.reshape(cluster_labels, (n_repeats, n_components))
        silhouette_avg = silhouette_score(flipped_metagenes, cluster_labels)

        if doprint:
            for r in range(n_repeats):
                print(r, cluster_label_table[r, :])

            print("For n_clusters =", n_components,
                  "The average silhouette_score is :", silhouette_avg)

        # Now let's collect the samples for each cluster and find the median centroid

        median_metagenes = []
        for nc in range(n_components):
            ixs = np.where(cluster_labels == nc)[0]
            metagenes_for_this_comp = flipped_metagenes[ixs, :]
            median_metagene = np.median(metagenes_for_this_comp, axis=0)
            median_metagenes += [median_metagene]

        # Put the metagenes in form of a matrix in standard (genes, components) orientation
        median_metagenes_matrix = np.vstack(median_metagenes).T
        assert median_metagenes_matrix.shape == (self.n_genes, n_components)

        return silhouette_avg, median_metagenes_matrix

    def find_best_n_components(self, facto_class, start, end, n_repeats, doprint=False,
                               doshow=False):
        scores = {}
        for n_components in range(start, end):
            scores[n_components], _ = self.compute_silhouette_score_and_median(
                facto_class, n_components, n_repeats, doprint=False)
            if doprint:
                print("%d: %8.6f" % (n_components, scores[n_components]))
        if doshow:
            plt.plot(scores.keys(), scores.values(), '-o')
            plt.xlabel('n_components')
            plt.ylabel('Silhouette score')
            plt.xticks(np.arange(start, end, step=2))
            plt.title("Silhouette plot for %s" % facto_class.__name__[:3])
            plt.show()


# noinspection PyUnusedLocal,PyUnreachableCode
def main():
    # expression_file = '../Data/Mini_Expression.csv'
    expression_file = '../Data/HGSOC_Protein_Expression.csv'
    # expression_file = '../Data/TCGA_OV_VST.csv'

    if False:
        FactorClustering.demonstrate_angles_in_high_dimensions()
        plt.show()

    fc = FactorClustering()

    fc.read_expression_matrix(expression_file)

    n_repeats = 50
    if False:
        # Beware - this will take hours (for the full size dataset)!
        #
        # clustering.compute_and_cache_multiple_factor_repeats(2, 15, n_repeats)
        # clustering.compute_and_cache_multiple_factor_repeats(15, 27, n_repeats)
        fc.compute_and_cache_multiple_factor_repeats(10, 11, n_repeats)

    if False:
        # clustering.plot_multiple_combined_factors_scatter(2, 14, n_repeats)
        # clustering.plot_multiple_combined_factors_scatter(15, 27, n_repeats)
        fc.plot_multiple_combined_factors_scatter(10, 11, n_repeats)

    if False:
        fc.investigate_multiple_cluster_statistics(2, 27, n_repeats)

    if False:
        # fc.find_best_n_components(NMF_Factorizer, 2, 21, 50, doprint=True, doshow=True)
        fc.find_best_n_components(ICA_Factorizer, 2, 21, 50, doprint=True, doshow=True)

    if False:
        fc.single_factor_scatter(NMF_Factorizer, 8, 50)

    if True:
        fc.plot_multiple_single_factors_scatter(NMF_Factorizer, 2, 14, 50)


if __name__ == '__main__':
    main()
