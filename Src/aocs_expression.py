
# coding: utf-8

# # Applying ICA / NMF to AOCS Ovarian Cancer gene expression

# In[ ]:


from os import path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF, FastICA, PCA
from sklearn.manifold import TSNE
import statsmodels.formula.api as sm
import pickle
import mygene
import qgrid
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


from factorizer_wrappers import ICA_Factorizer, NMF_Factorizer, PCA_Factorizer, Nimfa_NMF_Factorizer
from factorizer_wrappers import example_V, test_example_V, test_Factorizer


# In[ ]:


sns.set(style='white', context='notebook', rc={'figure.figsize':(10,6)})


# In[ ]:


test_example_V()

test_Factorizer(ICA_Factorizer(n_components=4), atol=0.5) 
test_Factorizer(ICA_Factorizer(n_components=5), atol=0.001) 

test_Factorizer(PCA_Factorizer(n_components=4), atol=0.5) 
test_Factorizer(PCA_Factorizer(n_components=5), atol=0.001) 

test_Factorizer(NMF_Factorizer(n_components=4), atol=0.5) 
test_Factorizer(NMF_Factorizer(n_components=5), atol=0.1) 


# ### Matrix plotting utility

# In[ ]:


def l2_norm_diff(m1, m2):
#   return np.mean(np.sqrt((m1 - m2)**2))
    return np.sqrt(np.mean((m1 - m2)**2))

def test_l2_norm_diff():
    V = example_V(10)
    rms = l2_norm_diff(V, V+0.5)
    assert np.isclose(rms,0.5)
    
test_l2_norm_diff() 


# In[ ]:


def show_W_H_WH_V(W, H, V, rec_V, n_genes_to_pick=None):
    """ Show factorization matrices in visually pleasing form"""
    
    if n_genes_to_pick is None:
        gene_ixs = range(V.shape[0])
        title = "Matrix decomposition, showing all geges"
    else:
        gene_ixs = sorted(np.random.randint(0, V.shape[0], n_genes_to_pick))
        title = "Matrix decomposition, randomly selecting %d genes for visibility" % n_genes_to_pick
    fig, axs = plt.subplots(1,4, figsize=(17,6))
    fig.suptitle(title, size=16)
    axs[0].imshow(W[gene_ixs,:], aspect='auto')
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
    axs[2].imshow(rec_V[gene_ixs,:], aspect='auto')
    axs[2].set_title('W H (RMS err=%6.2f)' % rms_err)
   
    
    axs[3].imshow(V[gene_ixs,:], aspect='auto')
    axs[3].set_title('V')
    axs[3].set_ylabel('genes', size=14)
    axs[3].set_xlabel('patients', size=14)

    plt.show()

def test_show_W_H_WH_V():
    
    V = example_V(10)
    print("v.shape", V.shape)
    facto = ICA_Factorizer(n_components=2)
    facto.fit(V)
    W = facto.get_W()
    H = facto.get_H()
    print("H.shape", H.shape)
    
    show_W_H_WH_V(W,H, V, facto.get_recovered_V())
    show_W_H_WH_V(W,H,V, facto.get_recovered_V(), 100)
test_show_W_H_WH_V()    


# ## Read and explore the expression matrix

# In[ ]:


# Read in expression spreadsheet which has been processed (see end of notebook) to inlcude only protein coding genes
expression_df = pd.read_csv('../Data/HGSOC_Protein_Expression.csv', sep='\t')
expression_df.set_index('GeneENSG', inplace=True)
assert len(expression_df) == 19730   # Only 
assert len(expression_df.columns) == 80
assert expression_df.columns[-1] == 'AOCS_171'

expression_matrix = np.asarray(expression_df)

print(expression_matrix.shape[0], "genes")
print(expression_matrix.shape[1], "patients")


# In[ ]:


plt.figure(figsize=(8, 12))
plt.imshow(expression_matrix, aspect='auto')
plt.colorbar()
plt.xlabel(("Patients"))
plt.ylabel(("Genes"))
plt.title("Expression matrix")
plt.show()


# ## Construct a dictionary to map Ensembl ENSG ids to symbols

# In[ ]:


# This is run-once code to query for all the Ensemble gene IDs we're using, construct a dictionary and write
# it to file.

ensgDictFile = '../Cache/ensgDict.pkl'
if not path.exists(ensgDictFile):  # Run only if dictionary file does not already exist
    mg = mygene.MyGeneInfo()
    ensgIDs = expression_df.index.values.tolist()    # All the gene IDs in this study
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


# In[ ]:


# Read the gene dictionary file
with open(ensgDictFile, 'rb') as f:
    ensgDict = pickle.load(f)
    
for (ensg, g) in ensgDict.items():
    if 'symbol' not in g.keys():
        g['symbol'] = ensg    # ensure lookup always succeeds
    
# Example use:
def example_ensgDict_use():
    gid = 'ENSG00000000938'
    # All ENSG ids used in this study should be in the dictionary
    ginfo = ensgDict[gid]
              
example_ensgDict_use()
    


# ## Note prior normalisation of the expression array
# Normalisation was applied by Ailith's script, using the method of a varaince stabalising transform.  See below, all patients have a minimum of aproximately 3.5, maximum approaximately 23.

# In[ ]:


expression_df.describe()


# ## Plot distributions of expression data
# ... for a quick visual check.

# In[ ]:


def show_expression_distributions(V):
    def labeled_figure():
        plt.figure(figsize=(14,4))
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
        _ = plt.hist(V[random_gene_ixs,:].T, bins=10)
        plt.title("Distribution of normalised expression levels, broken out by gene, for random %d genes" %
                  n_genes_to_pick)
        plt.show()
    
show_expression_distributions(expression_matrix)


# ## Read the patient metadata
# In particular we are interested in treatment "Resposnse", which we scraped from the Patch paper (code at end of notebool).

# In[ ]:


# Read metadata (which we scraped from the Patch etal paper!)
metadata_df = pd.read_csv('../Data/AOCS_metadata.csv', index_col='AOCS_ID')
assert metadata_df.columns[0] == "Age"
assert metadata_df.columns[1] == "Response"
# Make sure the IDs match-up between the two dataframes
assert (all(metadata_df.index == expression_df.columns))
metadata_df['Response'].value_counts()


# In[ ]:


qgrid.show_grid(metadata_df)


# ## Use ICA, NMF and PCA factorization and plot stuff...

# In[ ]:


def fit_and_plot_model(V, met_df, facto, plot=True):

    facto.fit(V)
    W = facto.get_W()
    H = facto.get_H()

    # Show the factored matrices and compare the reconstruction with the original
    if plot:
        show_W_H_WH_V(W, H, V, facto.get_recovered_V(), n_genes_to_pick=200)
    
    plot_df = metadata_df.copy().drop('Age', axis=1)
    
    factors = ['Factor_%d'%i for i in range(facto.n_components)]
    for i in range(len(factors)):
        plot_df[factors[i]] = H[i, :]
    
    # Boxplots of H factors by Response
    if plot:
        plot_df.boxplot(column=factors, by='Response', fontsize=10, figsize=(14,4), layout=(1, facto.n_components))
        plt.show()    

    # Scatter plots of metagenes matrix - W - using Seaborne
    if plot:
        sns.pairplot(plot_df, hue='Response')
        plt.show()
        
    # Make a t-SNE plot
    if plot:
        tsne = TSNE(n_components=2, init='pca', random_state=42, n_jobs=7)
        Y = tsne.fit_transform(W)
        sns.scatterplot(Y[:,0], Y[:,1])
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
result = fit_and_plot_model(expression_matrix, metadata_df,     
                            ICA_Factorizer(n_components=14),
                            plot=False)

print("\n================== NMF ======================")
result = fit_and_plot_model(expression_matrix, metadata_df,     
                            NMF_Factorizer(n_components=14),
                            plot=False)

print("\n================== PCA ======================")
result = fit_and_plot_model(expression_matrix, metadata_df,     
                            PCA_Factorizer(n_components=14),
                            plot=False)


# In[ ]:


if False:
    facto=ICA_Factorizer(n_components=14)
    V = expression_matrix
    facto.fit(V)
    W = facto.get_W()
    H = facto.get_H()

    # Make a t-SNE plot
    tsne = TSNE(n_components=2, init='pca', random_state=42, n_jobs=7)
    Y = tsne.fit_transform(W)
    
    #response_dict = {'Sensitive':0, 'Refractory':1, 'Resistant':2}
    #colour = np.array([response_dict[r] for r in plot_df['Response']]).T
    sns.scatterplot(Y[:,0], Y[:,1])
    plt.show()


# In[ ]:


def retreive_or_generate_results1_df():
    # Explore results for ICA, NMP and PCA, generating a list of dictionaries
    resultsFile = '../Cache/results1.csv'
    if not path.exists(resultsFile):
        Factos = [ICA_Factorizer, NMF_Factorizer, PCA_Factorizer]
        results1 = []
        for nc in range(2,40,2):
            for random_state in [42, 345, 13, 235, 583]:
                for Facto in Factos:
                    params = {'n_components':nc, 'random_state':random_state}

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


# In[ ]:


results1_df = retreive_or_generate_results1_df()
qgrid.show_grid(results1_df)
results1_df.columns


# In[ ]:


# Plot rms_err vs components for each method
results1_df = retreive_or_generate_results1_df()
for which in ["ICA_Factorizer", "NMF_Factorizer", "PCA_Factorizer"]:
    which_df = results1_df[results1_df['which']==which]
    x,y  = which_df['n_components'], which_df['rms_err']
    plt.plot(x,y, label=which[:3])
plt.legend()
plt.xlabel("n_components")
plt.ylabel("rms_err")
plt.show()


# In[ ]:


# Plot best_rsq fit to response vs components for each method
results1_df = retreive_or_generate_results1_df()
for which in ["ICA_Factorizer", "NMF_Factorizer", "PCA_Factorizer"]:
    which_df = results1_df[results1_df['which']==which]
    which_df = which_df.groupby('n_components').mean()
    x,y  = which_df.index, which_df['best_rsq']
    plt.plot(x,y, label=which[:3])
plt.legend()
plt.xlabel("n_components")
plt.ylabel("best_rsq")
plt.show()


# In[ ]:


which_df = results1_df[results1_df['which']==which]
which_df.groupby('n_components').mean()


# In[ ]:


def retreive_or_generate_results2_df():
# Explore FastICA with 14 components, for various parameters
    resultsFile = '../Cache/results2.csv'
    if not path.exists(resultsFile):
        results2 = []
        nc = 14
        for random_state in [42, 13, 56]:
            for max_iter in range(1, 100, 5):
                for fun in ['logcosh'] :# , 'exp', 'cube':
                    params = {'n_components':nc, 'random_state':random_state,
                              'fun':fun, 'max_iter':max_iter}
                    print(params)
                    facto = ICA_Factorizer(**params)
                    res = fit_and_plot_model(expression_matrix, metadata_df, facto, plot=False)
                    print(res)
                    results2.append({**params, **res})

        print("Writing results2.csv")
        results2_df = pd.DataFrame(results2)
        results2_df.to_csv('results2.csv')
        print("Done.")

    print("Reading",resultsFile)
    results2_df = pd.read_csv(resultsFile)
    return results2_df
    


# In[ ]:


qgrid.show_grid(retreive_or_generate_results2_df())


# ## Exploring distribution of weights in W and H matrices

# In[ ]:


def plot_matrix_weight_distributions(facto):
    facto.fit(expression_matrix)
    W = facto.get_W()
    H = facto.get_H()

    plt.figure(figsize=(12, 4))
    plt.suptitle("Distribution of W and H matrix weights for %s" %type(facto).__name__, size=16)
    plt.subplot(1,2,1)
    plt.hist(W.ravel(), bins=50, log=True)
    plt.xlabel("W matrix weights")
    plt.ylabel("Frequency")
    plt.subplot(1,2,2)
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

# In[ ]:


# Read in the k=14 metagenes matrix found by BIODICA

biodica_matrix_file = "../Factors/S_HGSOC_Protein_Expression_ica_numerical.txt_6.num"
biod_ica = np.loadtxt(biodica_matrix_file)
biod_ica.shape


# In[ ]:


from sklearn import preprocessing

all_genes = expression_matrix.shape[0]
n_genes = all_genes   # trim for speed while we develop
n_comps = 6
expression_matrix[:n_genes,:]

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


# In[ ]:


def show_component_ranked_plots(metagene_matrix):
    assert metagene_matrix.shape == (n_genes, n_comps)
    fig = plt.figure(figsize=(14,14))

    for ci in range(n_comps):
        plt.subplot(4,4,ci+1)
        metagene = metagene_matrix[:, ci]
        #influence = abs(metagene)
        influence = metagene
        gixpairs = zip(gene_symbols, influence)
        gixpairs = sorted(gixpairs, key=lambda p: -p[1])
        ranked_symbols, ranked_influence,  = zip(*gixpairs)
        plt.plot(ranked_influence)
        plt.yscale('log')
        if ci == 0:
            plt.xlabel('Influence rank')
            plt.ylabel('Influence')
        fig.tight_layout() 
        plt.title('Component %d' % ci)
    plt.show()    


show_component_ranked_plots(abs(biod_ica))


# In[ ]:


def show_component_distributions(metagene_matrix):
    assert metagene_matrix.shape == (n_genes, n_comps)
    fig = plt.figure(figsize=(14,14))

    for ci in range(n_comps):
        plt.subplot(4,4,ci+1)
        metagene = metagene_matrix[:, ci]
        #influence = abs(metagene)
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


# In[ ]:


# Analyse standard deviation of components
n_stddev = 3.0
for ci in range(biod_ica.shape[1]):
    metagene = biod_ica[:,ci]
    stddev = np.std(metagene)
    threshold = n_stddev * stddev
    num_above_threshold = len(metagene[abs(metagene) > threshold])
    print("Component %d, SD=%4.2f, #genes outside %3.1f SDs=%d" % (ci, stddev, n_stddev, num_above_threshold))


# In[ ]:


def select_influential_genes(metagene, top_n):
    influence = abs(metagene)
    stddev = np.std(metagene)
    mean = np.mean(metagene)
    assert abs(mean) < 0.01    # we'll need to do something else for +ve NMF
    threshold = n_stddev * stddev
    gixpairs = zip(gene_symbols, influence)
    selection = [symbol for (symbol, v) in gixpairs if abs(v) > threshold]
    
    return selection    

W = biod_ica
ranked_genes_by_component = {}
for ci in range(n_comps):
    _genes = select_influential_genes(W[:,ci], 100)
    ranked_genes_by_component[ci] = _genes
    if True:
        print("Comp. %d: %s" % (ci, ' '.join(_genes)))
    else:
        print("Comp. %d: \n%s\n" % (ci, '\n'.join(_genes)))
        # print("Comp. %d: \n%s\n" % (ci, genes))


# ## Gene enrichment analysis using GOATOOLS

# In[ ]:


from goatools import obo_parser
from goatools.go_enrichment import GOEnrichmentStudy
import Bio.UniProt.GOA as GOA
import gzip

# Load the Gene Ontology
gene_ontology = go = obo_parser.GODag('../DownloadedResources/go-basic.obo')

# Load the human annotations
c=0
with gzip.open('../DownloadedResources/goa_human.gaf.gz', 'rt') as gaf:
    funcs = {}
    for entry in GOA.gafiterator(gaf):
        c +=1 
        uniprot_id = entry.pop('DB_Object_Symbol')
        funcs[uniprot_id] = entry


# In[ ]:


c


# In[ ]:


# Our population is the set of genes we are analysing

population = [ensgDict[ensg]['symbol'] for ensg in expression_df.index.values]
print("We have %d genes in our population" % len(population))

# Build associations from functional annotations we got from the gaf file
associations = {}
for x in funcs:
    if x not in associations:
        associations[x] = set()
    associations[x].add(str(funcs[x]['GO_ID']))


# In[ ]:



gea = GOEnrichmentStudy(population, associations, gene_ontology,
                         propagate_counts=True,
                         alpha=0.05,
                         methods=['fdr'])
gea_results_by_component = {}
for ci in range(n_comps):
    study_genes = ranked_genes_by_component[ci]
    gea_results_by_component[ci] = gea.run_study(study_genes)


# In[ ]:


# Get results into a dataframe per component.  Easiest way is to use routine to write a .tsv file, 
# then read back and filter

gea_results_df_by_component = []
for ci in range(n_comps):
    tsv_name = '../Cache/goa_results_C%d.tsv' % ci
    with open(tsv_name, 'w') as f:
        gea.prt_tsv(f, gea_results_by_component[ci])
    ge_df = pd.read_csv(tsv_name, sep='\t')

    ge_df.rename(columns={'# GO':'GO_ID'}, inplace=True)
    ge_df.set_index('GO_ID', inplace=True)
    ge_df.drop(columns=['NS', 'enrichment', 'p_uncorrected'], inplace=True)
    ge_df = ge_df[ge_df['p_fdr'] <= 0.05]
    ge_df['Component'] = ci
    
    gea_results_df_by_component += [ge_df]


# In[ ]:


# Merge the per-component dataframes into a single one
gea_all_sig_results_df = pd.DataFrame()
gea_all_sig_results_df = gea_all_sig_results_df.append(gea_results_df_by_component)
gea_all_sig_results_df.to_csv('../Cache/gea_all_sig_results.tsv', sep='\t')


# In[ ]:


qgrid.show_grid(gea_all_sig_results_df)


# ## Run-once code following

# ### Following is run-once code for recovering metadata from the Patch et al paper...

# In[ ]:


# Extract age and response group from scraped text file
def extract_metadata_from_crazy_scraped_file(scrape_file):
    with open(scrape_file, 'r') as f:
        l1 = f.readline().strip()
        l2 = f.readline().strip()
    l1_words = l1.split(' ')
    aocs_ids = l1_words[::2]
    aocs_ids = [s.replace('-','_') for s in aocs_ids]
    
    ages = l1_words[1::2]
    response = l2.split(' ')
    assert len(aocs_ids) == len(ages) == len(response) == 80
    # Build a dataframe
    df = pd.DataFrame()
    df['AOCS_ID'] = aocs_ids
    df['Age'] = ages
    df['Response'] = response
    
    df = df.set_index('AOCS_ID')
    df = df.sort_index()
    
    return df


# In[ ]:


# Enable only if 'AOCS_metadata.csv' is to be re-created
if False:
    metadata_df = extract_metadata_from_crazy_scraped_file('../Data/aocs_raw_figure_e6.txt')
    metadata_df.to_csv('../Data/AOCS_metadata.csv')
    readback_metadata_df = pd.read_csv('../Data/AOCS_metadata.csv', index_col='AOCS_ID')
    readback_metadata_df
    assert len(readback_metadata_df) == 80
    readback_metadata_df


# ## Run-once code to convert create a protein coding gene only expression file
# 
# The original 'AOCS_TPM_VST.csv' contains 57,424 transcripts, many of which are non-codeing.   We wish to work with protein coding genes only.  We proceed as follows:
# 1. Read AOCS_TPM_VST.csv into a dataframe with ENSG identifiers as index
# 1. Write a text file listing all ENSG identifiers extracted from AOCS_TPM_VST.csv creating ensg_list.txt
# 1. Obtain an annotated gene table from [Biomart](https://m.ensembl.org/info/data/biomart/index.html):
#    1. Manually (should automate) upload ensg_list.txt
#    1. Select attributes of Gene stable ID, Gene name, Gene type, and Gene description; it's Gene type which is important
#    1. Export the generated table to 'DownloadedResources/mart_export.txt'
# 1. Read mart_export.txt into a dataframe with ENSG identifiers as index and filter on Gene type == 'protein_coding'
# 1. Merge the original full expression dataframe with the filtered dataframe
# 1. Write out a tab-seperated file 'HGSOC_Protein_Expression.csv' containing GeneENSG as first column with patient expression values in the following 80 columns
# 
# The generated HGSOC_Protein_Expression.csv is in a format suitable for direct input to BIODICA and can be used for all other analysis.
# 

# In[ ]:


if False:
    # Read in original full AOCS spreadsheet
    full_expression_df = pd.read_csv('../Data/AOCS_TPM_VST.csv')
    full_expression_df.set_index('GeneENSG', inplace=True)
    assert len(full_expression_df) == 57914
    assert len(full_expression_df.columns == 80 + 1)
    assert full_expression_df.columns[-1] == 'AOCS_171'
    ensglist = full_expression_df.index.values.tolist()
    with open('../Cache/ensg_list.txt', 'w') as f:
        f.write('\n'.jTrueoin(ensglist))


# In[ ]:


# This is where you have to do the manual Biomart stuff as described above... then run the following cell


# In[ ]:


if False:
    # Read in the Biomart created file
    mart_export_df = pd.read_csv('../DownloadedResources/mart_export.txt', sep='\t')
    mart_export_df.set_index('Gene stable ID', inplace=True)
    assert mart_export_df.loc['ENSG00000198804', 'Gene type'] == 'protein_coding'
    
    # Create a dataframe containing only protein coding genes
    mart_export_protein_coding_df = mart_export_df[mart_export_df['Gene type'] == 'protein_coding']

    # Merge with full expression dataframe (only those present in both will be kept)
    expression_protein_coding_df = pd.merge(
        left=full_expression_df, right=mart_export_protein_coding_df, 
        left_index=True, right_index=True)
    expression_protein_coding_df.drop(columns=['Gene name', 'Gene type', 'Gene description'], inplace=True)
    assert len(expression_protein_coding_df.columns) == 80

    # Write the filtered expression matrix to a .csv file
    expression_protein_coding_df.to_csv('../Data/HGSOC_Protein_Expression.csv', index=True, index_label='GeneENSG', sep='\t')
    
    # Read it back and check all in order
    del expression_protein_coding_df
    expression_protein_coding_df = pd.read_csv('../Data/HGSOC_Protein_Expression.csv', sep='\t')
    expression_protein_coding_df.set_index('GeneENSG', inplace=True)
    assert len(expression_protein_coding_df.columns) == 80
    assert len(expression_protein_coding_df) == 19730
    
    # Paranoia: the following specific expression value was manually extracted from the orginal AOCS_TPM_VST.csv,
    # and is compared here to check we haven't somehow scrambled the ordering anywhere!
    assert expression_protein_coding_df.loc['ENSG00000138772', 'AOCS_004'] == 12.6329098049671
    
    del full_expression_df
    


# In[ ]:


help(gea)

