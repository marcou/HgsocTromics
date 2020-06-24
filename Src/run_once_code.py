# ## Run-once code following

import pandas as pd


# ### Following is run-once code for recovering metadata from the Patch et al paper...

# Extract age and response group from scraped text file
def extract_metadata_from_crazy_scraped_file(scrape_file):
    with open(scrape_file, 'r') as f:
        l1 = f.readline().strip()
        l2 = f.readline().strip()
    l1_words = l1.split(' ')
    aocs_ids = l1_words[::2]
    aocs_ids = [s.replace('-', '_') for s in aocs_ids]

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

# This is where you have to do the manual Biomart stuff as described above... then run the following cell


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
    expression_protein_coding_df.drop(columns=['Gene name', 'Gene type', 'Gene description'],
                                      inplace=True)
    assert len(expression_protein_coding_df.columns) == 80

    # Write the filtered expression matrix to a .csv file
    expression_protein_coding_df.to_csv('../Data/HGSOC_Protein_Expression.csv', index=True,
                                        index_label='GeneENSG', sep='\t')

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
