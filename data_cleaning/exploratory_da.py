import pandas as pd
from rdkit.Chem import rdFingerprintGenerator, MolFromSmiles
import umap
from rdkit import RDLogger
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Literal
from useful_colors import usyd_blue, usyd_orange, usyd_grey
import seaborn as sns
from scipy.stats import ttest_ind
import itertools
from scipy.stats import ks_2samp
import os
from matplotlib.backends.backend_pdf import PdfPages

RDLogger.DisableLog("rdApp.*")  # type: ignore
plt.rcParams.update({'font.size': 20, 'font.family': 'Times New Roman'})


DO_DATASET = False
if DO_DATASET:
    honma = pd.read_excel("/home/luke/ames_graphormer/data/raw/Honma_New.xlsx")
    honma_train = honma[:-1589].copy()
    honma_test = honma[-1589:].copy()


    combined = pd.read_excel("/home/luke/ames_graphormer/data/raw/Combined_2s_as_0s.xlsx")
    # assert False, combined.head()
    combined_train = combined[combined['split'] == 'Train/Validation'].copy()
    combined_test = combined[combined['split'] == 'Test'].copy()
    del combined  # Ensure no accidental test set use

    assert len(combined_test) == len(honma_test), f"Length of combined test: {len(combined_test)} != Length of Honma test {len(honma_test)}"

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

    def smiles_to_mol(smiles: str):
        mol = MolFromSmiles(smiles)
        if mol is None:
            print(f"None smiles: {smiles}")
            return None
        return mol

    def mol_to_fp(mol, fp_dtype: Literal['rdk', 'numpy'] = 'rdk'):
        if mol is None:
            print(f"None mol: {mol}")
            return None
        match fp_dtype:
            case 'rdk':
                return mfpgen.GetFingerprint(mol)
            case 'numpy':
                return mfpgen.GetFingerprintAsNumPy(mol)
            case _:
                assert False, f"Invalid fp_dtype: {fp_dtype}"

    # Apply fingerprint generation on each dataset
    honma_train['mol'] = honma_train['smiles'].apply(smiles_to_mol)
    combined_train['mol'] = combined_train['smiles'].apply(smiles_to_mol)
    honma_test['mol'] = honma_test['smiles'].apply(smiles_to_mol)

    # Drop rows with missing fingerprints (None values)
    honma_train = honma_train.dropna()
    combined_train = combined_train.dropna()
    honma_test = honma_test.dropna()

    honma_train['fp'] = honma_train['mol'].apply(mol_to_fp)
    combined_train['fp'] = combined_train['mol'].apply(mol_to_fp)
    honma_test['fp'] = honma_test['mol'].apply(mol_to_fp)

    assert len(combined_train['ames']) == len(combined_train['fp']), f"Length of combined_train['ames']: {len(combined_train['ames'])} != Length of combined_train['fp']: {len(combined_train['fp'])}"

    combined_train.to_pickle("pickled_dataset_for_analysis/combined_train.pkl")
    honma_test.to_pickle("pickled_dataset_for_analysis/honma_test.pkl")

if DO_DATASET is False:
    combined_train = pd.read_pickle("pickled_dataset_for_analysis/combined_train.pkl")
    honma_test = pd.read_pickle("pickled_dataset_for_analysis/honma_test.pkl")

combined_train_fp = combined_train['fp'].tolist()
combined_train_ames = combined_train['ames'].tolist()
honma_test_fp = honma_test['fp'].tolist()

combined_train_source = combined_train['source'].to_numpy()
assert len(combined_train_source) == len(combined_train_fp), f"Length of combined_train_source: {len(combined_train_source)} != Length of combined_train_fp: {len(combined_train_fp)}"

from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

DO_SUMMARY_STATS = True
if DO_SUMMARY_STATS:
    def calculate_properties(mol):
        try:
            logP = rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
            molwt = rdMolDescriptors.CalcExactMolWt(mol)
            hba = rdMolDescriptors.CalcNumLipinskiHBA(mol)
            hbd = rdMolDescriptors.CalcNumLipinskiHBD(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
            return pd.Series([logP, tpsa, molwt, hba, hbd, rot], index=['C-LogP', 'TPSA', 'Molecular Weight', 'H-Bond Acceptor', 'H-Bond Donor', 'Rotatable Bond'])
        except:
            return pd.Series([None, None, None, None, None, None], index=['C-LogP', 'TPSA', 'Molecular Weight', 'H-Bond Acceptor', 'H-Bond Donor', 'Rotatable Bond'])

    combined_train[['C-LogP', 'TPSA', 'Molecular Weight', 'H-Bond Acceptor', 'H-Bond Donor', 'Rotatable Bond']] = combined_train['mol'].apply(calculate_properties)
    honma_test[['C-LogP', 'TPSA', 'Molecular Weight', 'H-Bond Acceptor', 'H-Bond Donor', 'Rotatable Bond']] = honma_test['mol'].apply(calculate_properties)

    # Map 'ames' from 0/1 to 'Negative'/'Positive'
    combined_train['ames_label'] = combined_train['ames'].map({0: 'Negative', 1: 'Positive'})
    honma_test['ames_label'] = honma_test['ames'].map({0: 'Negative', 1: 'Positive'})

    # List of descriptors to plot
    descriptors = ['C-LogP', 'TPSA', 'Molecular Weight', 'H-Bond Acceptor', 'H-Bond Donor', 'Rotatable Bond']

    # Prepare the data
    honma_train = combined_train[combined_train['source'] == 'honma']
    combined_train = combined_train  # Renamed for clarity

    # Define group labels and their corresponding dataframes
    group_labels = ["Honma Train", "Combined Train", "Honma Test"]
    group_data = [honma_train, combined_train, honma_test]

    def plot_and_save_violin_plots(honma_train, combined_train, honma_test, descriptors):
        fig, axes = plt.subplots(2, 3, figsize=(20, 15))

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        for i, descriptor in enumerate(descriptors):
            data = []
            for df, label in zip(group_data, group_labels):
                temp_df = df[[descriptor, 'ames_label']].copy()
                temp_df['Dataset'] = label
                data.append(temp_df)

                plot_data = pd.concat(data)
                plot_data = plot_data[plot_data[descriptor] < np.quantile(plot_data[descriptor], 0.975)]

                # Set 'Dataset' as a categorical variable with a specific order
                plot_data['Dataset'] = pd.Categorical(plot_data['Dataset'], categories=group_labels, ordered=True)

                sns.violinplot(
                    x='Dataset',
                    y=descriptor,
                    hue='ames_label',
                    data=plot_data,
                    order=group_labels,  # Ensure consistent ordering of datasets
                    hue_order=['Negative', 'Positive'],  # Ensure consistent ordering of hues
                    split=True,
                    inner="quartile",
                    ax=axes[i],
                    palette={'Negative': usyd_grey, 'Positive': usyd_orange}
                )
                axes[i].set_title(f'{descriptor} Distribution')
                axes[i].set_xlabel('')

                y_axis_label = {
                    'C-LogP': 'C-LogP',
                    'TPSA': 'TPSA (Å²)',
                    'Molecular Weight': 'Molecular Weight (g/mol)',
                    'H-Bond Acceptor': 'H-Bond Acceptors',
                    'H-Bond Donor': 'H-Bond Donors',
                    'Rotatable Bond': 'Rotatable Bonds',
                }

                axes[i].set_ylabel(y_axis_label[descriptor])

                # Remove individual legends from each subplot
                if axes[i].get_legend() is not None:
                    axes[i].get_legend().remove()

                # Calculate y-axis range and adjust for annotations
                y_min, y_max = axes[i].get_ylim()
                y_range = y_max - y_min
                y_offset = y_range * 0.05  # Small offset above the top of the violins
                increment = y_range * 0.05  # Height increment for each bracket

                # Perform pairwise t-tests between the three groups
                pairs = list(itertools.combinations(group_data, 2))
                pair_labels = list(itertools.combinations(group_labels, 2))

                # For keeping track of the maximum line_height for adjusting y-limits later
                max_line_height = y_max + y_offset

                for n_bracket, ((df1, df2), (label1, label2)) in enumerate(zip(pairs, pair_labels)):
                    # Set line_height for this bracket
                    line_height = y_max + y_offset + n_bracket * increment * 2

                    # Update max_line_height if needed
                    if line_height + increment > max_line_height:
                        max_line_height = line_height + increment

                    # Extract descriptor values for each group
                    data1 = df1[descriptor].dropna()
                    data2 = df2[descriptor].dropna()

                    # Perform t-test
                    t_stat, p_val = ttest_ind(data1, data2, equal_var=False)  # Welch's t-test

                    # Determine significance level
                    if p_val < 0.000001:
                        significance = '***'
                    elif p_val < 0.001:
                        significance = '**'
                    elif p_val < 0.05:
                        significance = '*'
                    else:
                        significance = 'ns'  # Not significant

                    # **Adjust x positions slightly to avoid overlap**
                    x1 = group_labels.index(label1) + n_bracket * 0.02
                    x2 = group_labels.index(label2) - n_bracket * 0.02

                    # **Adjust vertical line length to prevent overlap with lower annotations**
                    v_offset = increment * 0.2  # Shorten vertical lines by 20% of increment

                    # Plot the bracket with adjusted vertical lines
                    axes[i].plot(
                        [x1, x1, x2, x2],
                        [line_height + v_offset, line_height + increment, line_height + increment, line_height + v_offset],
                        lw=1.5, c='k'
                    )

                    # Add the significance text
                    axes[i].text(
                        (x1 + x2) * 0.5,
                        line_height + increment,
                        significance,
                        ha='center',
                        va='bottom',
                        color='k'
                    )

                # **Add extra padding above the highest significance line**
                extra_padding = y_range * 0.075  # Adjust the multiplier as needed for more padding

                # Adjust the y-limits to accommodate the annotations and extra padding
                axes[i].set_ylim(y_min, max_line_height + extra_padding)

            # Add a single legend for the entire figure
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, title='Ames Test Result', loc='lower center', ncol=2)

            # Adjust layout to make space for the legend
            plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust as needed

            # Save the figure
            plt.savefig('chemical_descriptors_violin_plots.pdf')
            plt.close()

    def plot_and_save_overlapping_histogram(honma_train, combined_train_main, honma_test, descriptors):
        # Ensure 'ames_label' exists in all dataframes
        for df in [honma_train, combined_train_main, honma_test]:
            if 'ames_label' not in df.columns:
                if 'ames' in df.columns:
                    df['ames_label'] = df['ames'].map({0: 'Negative', 1: 'Positive'})
                else:
                    raise ValueError("Dataframe must contain 'ames' column to map 'ames_label'.")

        # Define group labels and their corresponding dataframes
        group_labels = ["Combined Train", "Honma Train", "Honma Test"]
        group_data = [combined_train_main, honma_train, honma_test]

        # Define colors for each dataset with transparency
        colors = [usyd_grey, usyd_blue, usyd_orange]
        alphas = [0.5, 0.5, 0.5]  # Adjust transparency as needed

        # Split descriptors into first 3 and last 3
        first_descriptors = descriptors[:3]
        last_descriptors = descriptors[3:]

        # Helper function to plot and save histograms
        def plot_histograms(descriptors, filename_suffix):
            fig, axes = plt.subplots(len(descriptors), 2, figsize=(15, 5 * len(descriptors)))
            axes = axes.flatten()

            for i, descriptor in enumerate(descriptors):
                # Collect all data for this descriptor across all datasets and ames_labels
                overall_data_list = []
                for df in group_data:
                    # Check if descriptor exists in dataframe
                    if descriptor not in df.columns:
                        raise ValueError(f"Descriptor '{descriptor}' not found in dataframe columns.")

                    data = df[descriptor].dropna()
                    overall_data_list.append(data)

                overall_all_data = pd.concat(overall_data_list)
                upper_limit = np.quantile(overall_all_data, 0.975)
                lower_limit = np.quantile(overall_all_data, 0.025)
                overall_all_data = overall_all_data[(overall_all_data >= lower_limit) & (overall_all_data <= upper_limit)]

                # Determine bins using the Freedman–Diaconis rule
                if not overall_all_data.empty:
                    iqr = np.subtract(*np.percentile(overall_all_data, [75, 25]))
                    bin_width = 2 * iqr * len(overall_all_data) ** (-1 / 3)
                    if bin_width > 0:
                        bins = np.arange(lower_limit, upper_limit + bin_width, bin_width)
                        if len(bins) < 2:  # Ensure at least two bins are present
                            bins = np.linspace(lower_limit, upper_limit, num=10)
                    else:
                        bins = np.linspace(lower_limit, upper_limit, num=10)  # Default number of bins if bin_width is zero or negative
                else:
                    bins = np.linspace(lower_limit, upper_limit, num=10)  # Default number of bins if data is empty

                for j, ames_label in enumerate(['Negative', 'Positive']):
                    ax_index = i * 2 + j  # Compute the correct index for the axes array

                    # Clear the axis for safety
                    axes[ax_index].cla()

                    for k, df in enumerate(group_data):
                        data = df[df['ames_label'] == ames_label][descriptor].dropna()
                        # Limit individual data sets accordingly
                        data = data[(data >= lower_limit) & (data <= upper_limit)]

                        # Plot the histogram for each dataset with transparency
                        axes[ax_index].hist(
                            data,
                            bins=bins,
                            label=group_labels[k],
                            color=colors[k],
                            alpha=alphas[k],
                            edgecolor='black',
                            align='mid',
                            histtype='stepfilled',
                        )

                    # Set titles and labels
                    axes[ax_index].set_title(f"{descriptor} - Ames {ames_label}")
                    axes[ax_index].set_xlabel(descriptor)
                    axes[ax_index].set_ylabel('Frequency')

                    # Ensure consistent X-axis limits
                    axes[ax_index].set_xlim(bins[0], bins[-1])

                    # Set integer ticks on the x-axis
                    axes[ax_index].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

                    # Remove individual legends
                    axes[ax_index].legend().set_visible(False)

            # Add a single legend for the entire figure
            handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k], alpha=alphas[k]) for k in range(len(group_labels))]
            labels = group_labels
            fig.legend(handles, labels, title='Dataset', loc='lower center', ncol=3)

            # Adjust layout to make space for the legend
            plt.tight_layout(rect=[0, 0.05, 1, 1])

            # Save the figure
            plt.savefig(f'chemical_descriptors_overlapping_histogram_{filename_suffix}.pdf')
            plt.close()

        # Plot and save histograms for first and last descriptors
        if first_descriptors:
            plot_histograms(first_descriptors, 'continuous')
        if last_descriptors:
            plot_histograms(last_descriptors, 'discrete')

    # plot_and_save_violin_plots(honma_train, combined_train_main, honma_test, descriptors)
    plot_and_save_overlapping_histogram(honma_train, combined_train, honma_test, descriptors)

    from scipy.stats import ks_2samp

    def plot_significance_heatmaps(descriptors, datasets, combined_train, honma_test, honma_train, output_image='significance_heatmaps.pdf'):
        """
        Generate heatmaps for each descriptor's significance matrix and save them to a single image.

        Args:
            descriptors (list): List of descriptor names.
            datasets (list): List of dataset names.
            combined_train (DataFrame): DataFrame containing combined training data.
            honma_test (DataFrame): DataFrame containing Honma test data.
            honma_train (DataFrame): DataFrame containing Honma training data.
            output_image (str): Path to the output image file.

        Returns:
            None
        """
        from matplotlib.colors import LinearSegmentedColormap

        # Define color stops
        colors = [usyd_grey, usyd_orange]
        cmap = LinearSegmentedColormap.from_list('usyd_cmap', colors, N=100)

        # Define the dataframes to compare
        dataframes = {
            "Combined Train": combined_train,
            "Honma Train": honma_train,
            "Honma Test": honma_test
        }

        # Create a 3x2 grid of subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 15))
        axes = axes.flatten()

        from statsmodels.stats.multitest import multipletests

        for idx, descriptor in enumerate(descriptors):
            # Initialize a DataFrame to store KS test p-values
            ks_matrix = pd.DataFrame(index=datasets, columns=datasets)
            p_values = []

            # Calculate KS test p-values for each pair of datasets
            for i, (label1, df1) in enumerate(dataframes.items()):
                for j, (label2, df2) in enumerate(dataframes.items()):
                    if i <= j:  # Include diagonal, hide subdiagonal
                        # Extract descriptor data
                        data1 = df1[descriptor].dropna()
                        data2 = df2[descriptor].dropna()

                        # Perform KS test
                        _, p_value = ks_2samp(data1, data2)
                        p_values.append(p_value)

            # Apply Bonferroni correction
            corrected_p_values, corrected_alpha = multipletests(p_values, method='holm')[1:3]  # Use the corrected p-values and alphas
            print(f"corrected alpha: {corrected_alpha}")

            # Store the corrected p-values in the matrix
            p_idx = 0  # Use a separate index for corrected p-values
            for i, (label1, df1) in enumerate(dataframes.items()):
                for j, (label2, df2) in enumerate(dataframes.items()):
                    if i <= j:
                        ks_matrix.loc[label1, label2] = corrected_p_values[p_idx]
                        ks_matrix.loc[label2, label1] = corrected_p_values[p_idx]
                        p_idx += 1

            # Create the heatmap
            sns.heatmap(
                ks_matrix.astype(float),
                annot=True,
                fmt=".3f",
                cmap=cmap,
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'p-value', 'shrink': 0.6},
                square=True,
                linewidths=.5,
                linecolor='white',
                ax=axes[idx]
            )

            # Customize the heatmap
            axes[idx].set_title(f'KS Significance Matrix for {descriptor}', pad=15)
            axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=45, fontsize=14)
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, fontsize=14)

        # Adjust layout to make space for the legend
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        # Save the figure as a single image
        plt.savefig(output_image, bbox_inches='tight', format='pdf')
        plt.close(fig)

    # Ensure descriptors and datasets are defined
    descriptors = ['C-LogP', 'TPSA', 'Molecular Weight', 'H-Bond Acceptor', 'H-Bond Donor', 'Rotatable Bond']
    datasets = ["Combined Train", "Honma Train", "Honma Test"]

    # Call the function with the existing data
    plot_significance_heatmaps(descriptors, datasets, combined_train, honma_test, honma_train)

DO_UMAP = False
if DO_UMAP:
    def plot_and_save_umap(X_train, X_test, y_train, xlim: Optional[tuple[int, int]] = (-5, 25), ylim: Optional[tuple[int, int]] = (-15, 20)):
        reducer = umap.UMAP(metric='jaccard', min_dist=0.6, random_state=42)  # Moved inside function to avoid overwriting global state

        # Fit UMAP on the training data
        reducer.fit(X_train, y=y_train)
        train_embedding = reducer.embedding_

        # Transform the test data using the trained UMAP model
        test_embedding = reducer.transform(X_test)

        # Identify indices for 'honma' and 'others'
        honma_indices = np.where(combined_train_source == 'honma')[0]
        others_indices = range(len(combined_train_source))

        # assert False, f"honma_indices: {honma_indices}, others_indices: {others_indices}"

        honma_fig = plt.figure(figsize=(10, 8))
        honma_ax = honma_fig.add_subplot(111)
        honma_ax.scatter(
            train_embedding[honma_indices, 0],
            train_embedding[honma_indices, 1],
            alpha=0.6,
            label='Honma',
            color=usyd_grey,
            edgecolor='k',
            s=50
        )
        honma_ax.scatter(
            test_embedding[:, 0],
            test_embedding[:, 1],
            alpha=0.6,
            label='Test Set',
            color=usyd_orange,
            marker='x',
            s=80
        )
        if xlim:
            honma_ax.set_xlim(xlim)
        if ylim:
            honma_ax.set_ylim(ylim)
        honma_ax.set_xlabel('Embedding X')
        honma_ax.set_ylabel('Embedding Y')
        honma_ax.legend()

        honma_fig.tight_layout()
        honma_fig.savefig('umap_results/honma_umap.png', format='png', dpi=300)
        plt.close(honma_fig)

        # Create the second figure for 'others' sources
        combined_fig = plt.figure(figsize=(10, 8))
        combined_ax = combined_fig.add_subplot(111)  # Add a single Axes to the figure
        combined_ax.scatter(
            train_embedding[others_indices, 0],
            train_embedding[others_indices, 1],
            alpha=0.6,
            label='Combined',
            color=usyd_grey,
            edgecolor='k',
            s=50
        )
        combined_ax.scatter(
            test_embedding[:, 0],
            test_embedding[:, 1],
            alpha=0.6,
            label='Test Set',
            color=usyd_orange,
            marker='x',
            s=80
        )
        if xlim:
            combined_ax.set_xlim(xlim)
        if ylim:
            combined_ax.set_ylim(ylim)
        combined_ax.set_xlabel('Embedding X')
        combined_ax.set_ylabel('Embedding Y')
        combined_ax.legend()

        combined_fig.tight_layout()
        combined_fig.savefig('umap_results/combined_umap.png', format='png', dpi=300)
        plt.close(combined_fig)

    # Plot and save the UMAP embeddings as PDF
    plot_and_save_umap(combined_train_fp, honma_test_fp, combined_train_ames)