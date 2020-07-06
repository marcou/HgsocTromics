# NOTE: THIS FILE IS NOT CURRENTLY USED OR TESTED
# KEEPING  TO PICK BITS OF CODE AS NEEDED

# coding: utf-8

# # Applying ICA / NMF to AOCS Ovarian Cancer gene expression

import pickle
import warnings
from os import path

import matplotlib.pyplot as plt
import mygene
import numpy as np
import pandas as pd
import qgrid
import seaborn as sns
import statsmodels.formula.api as sm
from sklearn.manifold import TSNE
from factorizer_wrappers import ICA_Factorizer, NMF_Factorizer, PCA_Factorizer

warnings.simplefilter(action='ignore', category=FutureWarning)


# ### Matrix plotting utility

def l2_norm_diff(m1, m2):
    #   return np.mean(np.sqrt((m1 - m2)**2))
    return np.sqrt(np.mean((m1 - m2) ** 2))


def show_W_H_WH_V(W, H, V, rec_V, n_genes_to_pick=None):
    """ Show factorization matrices in visually pleasing form"""

    if n_genes_to_pick is None:
        gene_ixs = range(V.shape[0])
        title = "Matrix decomposition, showing all geges"
    else:
        gene_ixs = sorted(np.random.randint(0, V.shape[0], n_genes_to_pick))
        title = "Matrix decomposition, randomly selecting %d genes for visibility" % n_genes_to_pick
    fig, axs = plt.subplots(1, 4, figsize=(17, 6))
    fig.suptitle(title, size=16)
    axs[0].imshow(W[gene_ixs, :], aspect='auto')
    axs[0].set_title('W')
    axs[0].set_ylabel('genes', size=14)
    axs[0].set_xlabel('factors', size=14)

    axs[0].set_xticklabels('')

    axs[1].imshow(H, aspect='auto')
    axs[1].set_title('H')
    axs[1].set_ylabel('factors', size=14)
    axs[1].set_xlabel('patients', size=14)
    axs[1].set_yticklabels('')

    rms_err = l2_norm_diff(rec_V, V)
    axs[2].imshow(rec_V[gene_ixs, :], aspect='auto')
    axs[2].set_title('W H (RMS err=%6.2f)' % rms_err)

    axs[3].imshow(V[gene_ixs, :], aspect='auto')
    axs[3].set_title('V')
    axs[3].set_ylabel('genes', size=14)
    axs[3].set_xlabel('patients', size=14)

    plt.show()


# ## Read and explore the expression matrix

# Read in expression spreadsheet which has been processed (see end of notebook) to inlcude
# only protein coding genes
expression_df = pd.read_csv('../Data/HGSOC_Protein_Expression.csv', sep='\t')
expression_df.set_index('GeneENSG', inplace=True)
assert len(expression_df) == 19730  # Only
assert len(expression_df.columns) == 80
assert expression_df.columns[-1] == 'AOCS_171'

expression_matrix = np.asarray(expression_df)

print(expression_matrix.shape[0], "genes")
print(expression_matrix.shape[1], "patients")

plt.figure(figsize=(8, 12))
plt.imshow(expression_matrix, aspect='auto')
plt.colorbar()
plt.xlabel("Patients")
plt.ylabel("Genes")
plt.title("Expression matrix")
plt.show()

# ## Construct a dictionary to map Ensembl ENSG ids to symbols

# This is run-once code to query for all the Ensemble gene IDs we're using, construct
# a dictionary and write it to file.

ensgDictFile = '../Cache/ensgDict.pkl'
if not path.exists(ensgDictFile):  # Run only if dictionary file does not already exist
    mg = mygene.MyGeneInfo()
    ensgIDs = expression_df.index.values.tolist()  # All the gene IDs in this study
    ginfo = mg.querymany(ensgIDs, scopes='ensembl.gene')

    ensgDict = {}
    for g in ginfo:
        ensg = g['query']
        del g['query']
        ensgDict[ensg] = g

    print("Writing to %s..." % ensgDictFile)
    with open(ensgDictFile, 'wb') as f:
        pickle.dump(ensgDict, f)
    print("Done.")

# Read the gene dictionary file
with open(ensgDictFile, 'rb') as f:
    ensgDict = pickle.load(f)

for (ensg, g) in ensgDict.items():
    if 'symbol' not in g.keys():
        g['symbol'] = ensg  # ensure lookup always succeeds


# Example use:
def example_ensgDict_use():
    gid = 'ENSG00000000938'
    # All ENSG ids used in this study should be in the dictionary
    ginfo = ensgDict[gid]
    print(ginfo)


example_ensgDict_use()

# ## Note prior normalisation of the expression array
# Normalisation was applied by Ailith's script, using the method of a varaince stabalising
# transform.  See below, all patients have a minimum of aproximately 3.5, maximum approaximately
# 23.

expression_df.describe()


# ## Plot distributions of expression data
# ... for a quick visual check.

def show_expression_distributions(V):
    def labeled_figure():
        plt.figure(figsize=(14, 4))
        plt.xlabel('Expression level')
        plt.ylabel('Frequency')

    if True:
        labeled_figure()
        _ = plt.hist(V.ravel(), bins=40)
        plt.title("Distribution of all normalised expression levels")
        plt.show()

    if True:
        labeled_figure()
        _ = plt.hist(V, bins=40)
        plt.title("Distribution of normalised expression levels, broken out by patient")
        plt.show()

    if True:
        labeled_figure()
        n_genes_to_pick = 100
        random_gene_ixs = sorted(np.random.randint(0, V.shape[0], n_genes_to_pick))
        _ = plt.hist(V[random_gene_ixs, :].T, bins=10)
        plt.title(
            "Distribution of normalised expression levels, by gene, for random %d genes" %
            n_genes_to_pick)
        plt.show()


show_expression_distributions(expression_matrix)

# Read metadata (which we scraped from the Patch etal paper!)
metadata_df = pd.read_csv('../Data/AOCS_metadata.csv', index_col='AOCS_ID')
assert metadata_df.columns[0] == "Age"
assert metadata_df.columns[1] == "Response"
# Make sure the IDs match-up between the two dataframes
assert (all(metadata_df.index == expression_df.columns))
metadata_df['Response'].value_counts()

qgrid.show_grid(metadata_df)


# ## Use ICA, NMF and PCA factorization and plot stuff...

def fit_and_plot_model(V, meta_df, facto, plot=True):
    facto.fit(V)
    W = facto.get_W()
    H = facto.get_H()

    # Show the factored matrices and compare the reconstruction with the original
    if plot:
        show_W_H_WH_V(W, H, V, facto.get_recovered_V(), n_genes_to_pick=200)

    plot_df = meta_df.copy().drop('Age', axis=1)

    factors = ['Factor_%d' % i for i in range(facto.n_components)]
    for i in range(len(factors)):
        plot_df[factors[i]] = H[i, :]

    # Boxplots of H factors by Response
    if plot:
        plot_df.boxplot(column=factors, by='Response', fontsize=10, figsize=(14, 4),
                        layout=(1, facto.n_components))
        plt.show()

        # Scatter plots of metagenes matrix - W - using Seaborne
    if plot:
        sns.pairplot(plot_df, hue='Response')
        plt.show()

    # Make a t-SNE plot
    if plot:
        tsne = TSNE(n_components=2, init='pca', random_state=42, n_jobs=7)
        Y = tsne.fit_transform(W)
        sns.scatterplot(Y[:, 0], Y[:, 1])
        plt.show()

    # Put together a dictionary or results

    results_dict = {}

    # Find factor which best explains response

    ols_results = [sm.ols(fact + '~ C(Response)', data=plot_df).fit() for fact in factors]
    rsqs = [res.rsquared for res in ols_results]
    results_dict['best_rsq'] = np.max(rsqs)
    results_dict['best_factor'] = np.argmax(rsqs)
    results_dict['rms_err'] = l2_norm_diff(V, facto.get_recovered_V())

    return results_dict


print("================== ICA ======================")
fit_and_plot_model(expression_matrix, metadata_df,
                   ICA_Factorizer(n_components=14),
                   plot=False)

print("\n================== NMF ======================")
fit_and_plot_model(expression_matrix, metadata_df,
                   NMF_Factorizer(n_components=14),
                   plot=False)

print("\n================== PCA ======================")
fit_and_plot_model(expression_matrix, metadata_df,
                   PCA_Factorizer(n_components=14),
                   plot=False)

if False:
    facto = ICA_Factorizer(n_components=14)
    V = expression_matrix
    facto.fit(V)
    W = facto.get_W()
    H = facto.get_H()

    # Make a t-SNE plot
    tsne = TSNE(n_components=2, init='pca', random_state=42, n_jobs=7)
    Y = tsne.fit_transform(W)

    # response_dict = {'Sensitive':0, 'Refractory':1, 'Resistant':2}
    # colour = np.array([response_dict[r] for r in plot_df['Response']]).T
    sns.scatterplot(Y[:, 0], Y[:, 1])
    plt.show()


def retreive_or_generate_results1_df():
    # Explore results for ICA, NMP and PCA, generating a list of dictionaries
    resultsFile = '../Cache/results1.csv'
    if not path.exists(resultsFile):
        Factos = [ICA_Factorizer, NMF_Factorizer, PCA_Factorizer]
        results1 = []
        for nc in range(2, 40, 2):
            for random_state in [42, 345, 13, 235, 583]:
                for Facto in Factos:
                    params = {'n_components': nc, 'random_state': random_state}

                    facto = Facto(**params)
                    params['which'] = type(facto).__name__
                    print(params)
                    res = fit_and_plot_model(expression_matrix, metadata_df, facto, plot=False)
                    print(res)
                    results1.append({**params, **res})

        print("Writing results1.csv")
        results1_df = pd.DataFrame(results1)
        results1_df.to_csv('results1.csv')
        print("Done.")

    print("Reading", resultsFile)
    results1_df = pd.read_csv(resultsFile)
    results1_df = results1_df.drop(columns=['Unnamed: 0'])
    return results1_df


results1_df = retreive_or_generate_results1_df()
qgrid.show_grid(results1_df)
results1_df.columns

# Plot rms_err vs components for each method
results1_df = retreive_or_generate_results1_df()
for which in ["ICA_Factorizer", "NMF_Factorizer", "PCA_Factorizer"]:
    which_df = results1_df[results1_df['which'] == which]
    x, y = which_df['n_components'], which_df['rms_err']
    plt.plot(x, y, label=which[:3])
plt.legend()
plt.xlabel("n_components")
plt.ylabel("rms_err")
plt.show()

# Plot best_rsq fit to response vs components for each method
results1_df = retreive_or_generate_results1_df()
for which in ["ICA_Factorizer", "NMF_Factorizer", "PCA_Factorizer"]:
    which_df = results1_df[results1_df['which'] == which]
    which_df = which_df.groupby('n_components').mean()
    x, y = which_df.index, which_df['best_rsq']
    plt.plot(x, y, label=which[:3])
plt.legend()
plt.xlabel("n_components")
plt.ylabel("best_rsq")
plt.show()

which_df = results1_df[results1_df['which'] == which]
which_df.groupby('n_components').mean()


def retreive_or_generate_results2_df():
    # Explore FastICA with 14 components, for various parameters
    resultsFile = '../Cache/results2.csv'
    if not path.exists(resultsFile):
        results2 = []
        nc = 14
        for random_state in [42, 13, 56]:
            for max_iter in range(1, 100, 5):
                for fun in ['logcosh']:  # , 'exp', 'cube':
                    params = {'n_components': nc, 'random_state': random_state,
                              'fun': fun, 'max_iter': max_iter}
                    print(params)
                    facto = ICA_Factorizer(**params)
                    res = fit_and_plot_model(expression_matrix, metadata_df, facto, plot=False)
                    print(res)
                    results2.append({**params, **res})

        print("Writing results2.csv")
        results2_df = pd.DataFrame(results2)
        results2_df.to_csv('results2.csv')
        print("Done.")

    print("Reading", resultsFile)
    results2_df = pd.read_csv(resultsFile)
    return results2_df

qgrid.show_grid(retreive_or_generate_results2_df())


# ## Exploring distribution of weights in W and H matrices

def plot_matrix_weight_distributions(facto):
    facto.fit(expression_matrix)
    W = facto.get_W()
    H = facto.get_H()

    plt.figure(figsize=(12, 4))
    plt.suptitle("Distribution of W and H matrix weights for %s" % type(facto).__name__, size=16)
    plt.subplot(1, 2, 1)
    plt.hist(W.ravel(), bins=50, log=True)
    plt.xlabel("W matrix weights")
    plt.ylabel("Frequency")
    plt.subplot(1, 2, 2)
    plt.hist(H.ravel(), bins=20, log=True)
    plt.ylabel("Frequency")
    plt.xlabel("H matrix weights")
    plt.show()


facto = ICA_Factorizer(n_components=14)
plot_matrix_weight_distributions(facto)

facto = NMF_Factorizer(n_components=14)
plot_matrix_weight_distributions(facto)

facto = PCA_Factorizer(n_components=14)
plot_matrix_weight_distributions(facto)

# ## Let's find some influential genes!

# Read in the k=14 metagenes matrix found by BIODICA

biodica_matrix_file = "../Factors/S_HGSOC_Protein_Expression_ica_numerical.txt_6.num"
biod_ica = np.loadtxt(biodica_matrix_file)
biod_ica.shape

all_genes = expression_matrix.shape[0]
n_genes = all_genes  # trim for speed while we develop
n_comps = 6
expression_matrix[:n_genes, :]

gene_ENSG_ids = expression_df.index.values[:n_genes]
gene_symbols = [ensgDict[ensg]['symbol'] for ensg in gene_ENSG_ids]
nmf_facto = NMF_Factorizer(n_components=n_comps, max_iter=1000)
ica_facto = ICA_Factorizer(n_components=n_comps)
pca_facto = PCA_Factorizer(n_components=n_comps)

V = expression_matrix
nmf_facto.fit(V)
print("NMF fit done")
ica_facto.fit(V)
print("ICA fit done")
pca_facto.fit(V)
print("PCA fit done")


def show_component_ranked_plots(metagene_matrix):
    assert metagene_matrix.shape == (n_genes, n_comps)
    fig = plt.figure(figsize=(14, 14))

    for ci in range(n_comps):
        plt.subplot(4, 4, ci + 1)
        metagene = metagene_matrix[:, ci]
        # influence = abs(metagene)
        influence = metagene
        gixpairs = zip(gene_symbols, influence)
        gixpairs = sorted(gixpairs, key=lambda p: -p[1])
        ranked_symbols, ranked_influence, = zip(*gixpairs)
        plt.plot(ranked_influence)
        plt.yscale('log')
        if ci == 0:
            plt.xlabel('Influence rank')
            plt.ylabel('Influence')
        fig.tight_layout()
        plt.title('Component %d' % ci)
    plt.show()


show_component_ranked_plots(abs(biod_ica))


def show_component_distributions(metagene_matrix):
    assert metagene_matrix.shape == (n_genes, n_comps)
    fig = plt.figure(figsize=(14, 14))

    for ci in range(n_comps):
        plt.subplot(4, 4, ci + 1)
        metagene = metagene_matrix[:, ci]
        # influence = abs(metagene)
        influence = metagene

        plt.hist(influence, bins=20)
        # plt.yscale('log')
        if ci == 0:
            plt.xlabel('Influence')
            plt.ylabel('Freq')
        fig.tight_layout()
        plt.title('Component %d' % ci)
    plt.show()


show_component_distributions(biod_ica)

