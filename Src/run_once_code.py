# ## Run-once code following

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


# noinspection PyStatementEffect,PyUnreachableCode,PyPep8,PyPep8
def run_once_code():
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
    # The original 'AOCS_TPM_VST.csv' contains 57,424 transcripts, many of which are non-codeing.
    # We wish to work with protein coding genes only.  We proceed as follows:
    # 1. Read AOCS_TPM_VST.csv into a dataframe with ENSG identifiers as index
    # 1. Write a text file listing all ENSG identifiers extracted from AOCS_TPM_VST.csv
    #    creating ensg_list.txt
    # 1. Obtain an annotated gene table from
    #    [Biomart](https://m.ensembl.org/info/data/biomart/index.html):
    #    1. Manually (should automate) upload ensg_list.txt
    #    1. Select attributes of Gene stable ID, Gene name, Gene type, and Gene description;
    #       it's Gene type which is important
    #    1. Export the generated table to 'DownloadedResources/mart_export.txt'
    # 1. Read mart_export.txt into a dataframe with ENSG identifiers as index and filter on
    #    Gene type == 'protein_coding'
    # 1. Merge the original full expression dataframe with the filtered dataframe
    # 1. Write out a tab-seperated file 'HGSOC_Protein_Expression.csv' containing GeneENSG
    #    as first column with patient expression values in the following 80 columns
    #
    # The generated HGSOC_Protein_Expression.csv is in a format suitable for direct input
    # to BIODICA and can be used for all other analysis.
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
            f.write('\n'.join(ensglist))

    # This is where you have to do the manual Biomart stuff as described above...
    # then run the following cell

    if False:
        # Read in the Biomart created file
        mart_export_df = pd.read_csv('../DownloadedResources/mart_export.txt', sep='\t')
        mart_export_df.set_index('Gene stable ID', inplace=True)
        assert mart_export_df.loc['ENSG00000198804', 'Gene type'] == 'protein_coding'

        # Create a dataframe containing only protein coding genes
        mart_export_protein_coding_df = mart_export_df[
            mart_export_df['Gene type'] == 'protein_coding']

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

        # Paranoia: the following specific expression value was manually extracted from the
        # orginal AOCS_TPM_VST.csv,
        # and is compared here to check we haven't somehow scrambled the ordering anywhere!
        assert expression_protein_coding_df.loc['ENSG00000138772', 'AOCS_004'] == 12.6329098049671

        del full_expression_df


# The following is not currently used - keeping here for now
def l2_norm_diff(m1, m2):
    return np.sqrt(np.mean((m1 - m2) ** 2))


# Angle calculation
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
# noinspection PyStringFormat
def demonstrate_angles_in_high_dimensions(dims=50000, n=1000):
    # Demonstrating that in a 20,000 dimensioned space, any two random vectors will be at
    # very close to 90 degrees!
    alist = []
    rvs = np.random.randn(n, dims)
    for i in range(n - 1):
        v1 = rvs[i, :]
        v2 = rvs[i + 1, :]
        a = calc_angle(v1, v2)
        alist += [a]

    plt.hist(alist, bins=int(np.sqrt(n)))
    plt.title("Mean=%6.2f, SD=%6.2f degrees" % (np.mean(alist), np.std(alist)))
    # plt.show()


def test_l2_norm_diff():
    m1 = np.array([0, 0, 1])
    m2 = np.array([0, 0, 1])
    m3 = np.array([1, 1, 0])

    assert l2_norm_diff(m1, m2) == 0
    assert l2_norm_diff(m1, m3) == 1


def test_calc_angle():
    m1 = np.array([0, 0, 1])
    m2 = np.array([0, 0, 1])
    m3 = np.array([0, 1, 0])
    assert np.isclose(calc_angle(m1, m2), 0)
    assert np.isclose(calc_angle(m1, m3), 90)


def test_demonstrate_angles_in_high_dimensions():
    demonstrate_angles_in_high_dimensions(1000, 100)
