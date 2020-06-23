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

# ## Factorizer classes
# These have now been hived off into ``factorizer_wrappers.py``.  Import and test them.

from factorizer_wrappers import ICA_Factorizer, NMF_Factorizer, PCA_Factorizer


# noinspection PyStringFormat
class FactorClustering:
    # ## Read the expression matrix
    # This is repeated code, should be factored out...
    def __init__(self):
        self.expression_df = None
        self.expression_matrix = None
        self.expression_filename = None

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

        print(self.expression_matrix.shape[0], "genes")
        print(self.expression_matrix.shape[1], "patients")

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
    def demonstrate_angles_in_high_dimensions(dims=10000, n=5000):
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

    def compute_and_cache_one_factor_repeats(self, V, facto_class, n_components, n_repeats):
        pickle_fname = self.cached_factor_repeats_filename(
            facto_class, n_components, n_repeats)
        p = Path(pickle_fname)
        os.makedirs(p.parent, exist_ok=True)

        print(pickle_fname)
        metagene_list = []
        for i in range(n_repeats):
            facto = facto_class(n_components=n_components, max_iter=5000,
                                random_state=np.random.randint(10000))
            facto.fit(V)
            metagene_list += [facto.get_W()]
            print('\r%d/%d' % (i + 1, n_repeats), end='')
        print()
        with open(pickle_fname, 'wb') as f:
            pickle.dump(metagene_list, f)
        return pickle_fname

    def read_cached_factors(self, facto_class, n_components, n_repeats):
        pickle_fname = self.cached_factor_repeats_filename(
            facto_class, n_components, n_repeats)
        with open(pickle_fname, 'rb') as f:
            metagene_list = pickle.load(f)
        return metagene_list

    def compute_and_cache_multiple_factor_repeats(self, start, end, n_repeats=50):
        # This will take several hours, if enabled!
        if True:
            V = self.expression_matrix
            for nc in range(start, end):
                self.compute_and_cache_one_factor_repeats(V, NMF_Factorizer, nc, n_repeats)
                self.compute_and_cache_one_factor_repeats(V, ICA_Factorizer, nc, n_repeats)
                print()

            print("All Done.")

    def combined_factors_scatter(self, n_components, n_repeats):
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
        flipped_metagenes = [g if sum(g[:10]) > 0 else -g for g in stacked_metagenes[:]]

        # Reduce to a managable number of dimensions before passing to t-SNE
        pca = PCA(n_components=min(50, len(flipped_metagenes)))
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
        plt.figure(figsize=(16, 20))
        for nc in range(start_comp, end_comp):
            print('.', end='')
            plt.subplot(4, 3, nc - start_comp + 1)
            self.combined_factors_scatter(nc, n_repeats)
        plt.suptitle("t-SNE clustering for %d repeats of NMF and ICA", size=14)
        plt.show()

    # plot_multiple_combined_factors_scatter(3, 15)
    # plot_multiple_combined_factors_scatter(16, 28)

    # ## Pick out the clusters with k-means
    # Although NMF seems to produce components in a repeatable order - so that centroids can be
    # calculated directly, this seems not to be the case for ICA.  So use k-means to sort them out

    def investigate_cluster_statistics(self, facto_class, n_components, n_repeats,  doprint=False):
        # The given facto is not actually executed, just used to select the appropriate cached
        # .pkl files which were computed above.
        metagene_list = self.read_cached_factors(facto_class, n_components, n_repeats)
        stacked_metagenes = np.hstack(metagene_list).T
        flipped_metagenes = [g if sum(g[:10]) > 0 else -g for g in stacked_metagenes[:]]

        pca = PCA(n_components=min(10, len(metagene_list)))
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

# THIS IS BROKEN!
# Calculate angles between components
# a = pca.inverse_transform(kmeans.cluster_centers_)
# n = a.shape[0]
# angle_matrix = np.zeros((n, n))
# for i1 in range(n):
#     for i2 in range(n):
#         v1, v2 = a[i1, :], a[i2, :]
#         angle_matrix[i1, i2] = calc_angle(v1, v2)
#
# for i1 in range(n):
#     print("%2d: " % i1, end="")
#     for i2 in range(n):
#         if i2 <= i1:
#             print("%6.0f°" % angle_matrix[i1, i2], end="")
#     print()


# noinspection PyUnusedLocal,PyUnreachableCode
def main():
    # expression_file = '../Data/Mini_Expression.csv'
    expression_file = '../Data/HGSOC_Protein_Expression.csv'

    if False:
        FactorClustering.demonstrate_angles_in_high_dimensions()
        plt.show()

    clustering = FactorClustering()

    clustering.read_expression_matrix(expression_file)

    n_repeats = 50
    if False:
        # Beware - this will take hours (for the full size dataset)!
        clustering.compute_and_cache_multiple_factor_repeats(2, 31, n_repeats)

    if False:
        clustering.plot_multiple_combined_factors_scatter(2, 14, n_repeats)
        clustering.plot_multiple_combined_factors_scatter(15, 27, n_repeats)

    if False:
        clustering.investigate_multiple_cluster_statistics(2, 15, n_repeats)


if __name__ == '__main__':
    main()
