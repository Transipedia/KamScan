import functools
import shutil
import time
import statistics
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import rankdata
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
import multiprocessing as mp
import argparse
import glob
import os
import logging
import math

# ......................................................
#
#   CONSTANTS ----
#
# ......................................................
N_ROUND_NORM = 3
N_ROUND_TEST = 1
# For ZIW only
CSTE_VARIANCE = 2 * math.sqrt(2 / math.pi)


# ......................................................
#
#   FUNCTIONS ----
#
# ......................................................


# Estimate total number of tags (lines)

def estimate_total_lines(file_path, sample_size=100):
    """ Estimate the total number of lines in a file using a sample of size sample_size """
    with open(file_path, 'r') as f:
        # Read a sample of sample_size lines
        sample_lines = [next(f) for _ in range(sample_size)]
    
    # Calculate the average line size in the sample
    average_line_size = sum(len(line) for line in sample_lines) / sample_size
    
    # Get the total file size in bytes
    file_size = os.path.getsize(file_path)
    
    # Estimate the total number of lines
    estimated_total_lines = int(file_size / average_line_size)
    
    return estimated_total_lines


# Perform Pi-test

def perform_pitest(tag, grp_a_data, grp_b_data):
    log2fold_change = np.log2(np.mean(grp_a_data.loc[tag].values + 1) / np.mean(grp_b_data.loc[tag].values + 1))
    Pttest = perform_ttest(tag, grp_a_data, grp_b_data)

    if Pttest is not None and log2fold_change is not None and log2fold_change != 0:
        pivalue = abs(log2fold_change * (-np.log10(Pttest[2])))
        if not np.isnan(pivalue):  # Check for NaN 
            return Pttest[2], np.round(pivalue, N_ROUND_TEST), np.round(log2fold_change, N_ROUND_TEST)

# Perform anova test with covariable

def perform_anova(tag, grp_a_data, grp_b_data, covariates_df):
    
    covariate_columns = list(covariates_df.columns)
    
    # 1. Extract expression values (log-transformed)
    grp_a_values = np.log(grp_a_data.loc[tag].values + 1)
    grp_b_values = np.log(grp_b_data.loc[tag].values + 1)

    # 2. Build sample order matching the expression data
    sample_order = list(grp_a_data.columns) + list(grp_b_data.columns)

    # 3. Align covariates safely
    cov_aligned = covariates_df.loc[sample_order, covariate_columns].reset_index(drop=True)

    # 4. Build analysis DataFrame
    df = pd.DataFrame({
        "y": np.concatenate([grp_a_values, grp_b_values]),
        "group": ["A"] * len(grp_a_values) + ["B"] * len(grp_b_values),
    })

    df[covariate_columns] = cov_aligned

    # 5. Fit regression
    formula = "y ~ group + " + " + ".join(covariate_columns)
    model = smf.ols(formula, data=df).fit()

    # Find the correct coefficient name for the group comparison
    group_coef = "group[T.B]" if "group[T.B]" in model.pvalues else "group"

    # 6. Log2FC
    log2fold_change = np.log2(np.mean(grp_a_data.loc[tag].values + 1) / np.mean(grp_b_data.loc[tag].values + 1))
    
    p_value = model.pvalues[group_coef]

    return str(tag), np.round(model.tvalues[group_coef], N_ROUND_TEST), np.round(log2fold_change, N_ROUND_TEST), p_value

# Perform T-test

def perform_ttest(tag, grp_a_data, grp_b_data):
    grp_a_values = np.log(grp_a_data.loc[tag].values + 1)
    grp_b_values = np.log(grp_b_data.loc[tag].values + 1)
    log2fold_change = np.log2(np.mean(grp_a_data.loc[tag].values + 1) / np.mean(grp_b_data.loc[tag].values + 1))  
    t_statistic, p_value = ttest_ind(grp_a_values, grp_b_values)
    if not np.isnan(t_statistic):  # Check for NaN 
        return str(tag), np.round(t_statistic, N_ROUND_TEST), np.round(log2fold_change, N_ROUND_TEST), p_value
       
# Perform Wilcoxon Test

def perform_wilcoxon_test(tag, grp_a_data, grp_b_data):
    grp_a_values = grp_a_data.loc[tag].values
    grp_b_values = grp_b_data.loc[tag].values
    log2fold_change = np.log2(np.mean(grp_a_values + 1) / np.mean(grp_b_values + 1))  
    # Perform the Wilcoxon test
    statistic, p_value = mannwhitneyu(grp_a_values, grp_b_values)
    return str(tag), np.round(statistic, N_ROUND_TEST), np.round(log2fold_change, N_ROUND_TEST), p_value

# Calculate variance of the modified Wilcoxon rank sum statistic

def calculate_variance_ziw(data_dict: dict, prop_mean: float, n: int) -> float:
    # Calculate variance of the modified Wilcoxon rank sum statistic
    # @prop_mean: a float, average proportion of zeros in both groups
    prop_mean_complement = 1 - prop_mean

    # Variance in the first group
    v1 = prop_mean * prop_mean_complement
    var1 =  data_dict["product_n_a_n_b"] * prop_mean * v1 * (n * prop_mean + 3 * prop_mean_complement / 2) + \
    data_dict["sum_square_n_a_n_b"] * v1**2 * (5/4) + CSTE_VARIANCE * prop_mean * (n * v1)**(1.5) \
    * data_dict["sqrt_product_n_a_n_b"]
    var1 = var1 / 4

    # Variance in the second group
    var2 = data_dict["product_n_a_n_b"] * prop_mean**2 * (n * prop_mean + 1)/12
    
    # Variance of the modified Wilcoxon rank sum statistic
    variance = var1 + var2

    return variance

# Perform ZIW

def perform_ziw(tag, grp_a, grp_b, data_dict):
    grp_a_counts = np.array(grp_a.loc[tag].values.flatten().tolist())
    grp_b_counts = np.array(grp_b.loc[tag].values.flatten().tolist())

    n_non_zero_grp_a = np.count_nonzero(grp_a_counts)
    n_non_zero_grp_b = np.count_nonzero(grp_b_counts)

    # Proportion of non-zeros by group
    prop_grp_a: float = n_non_zero_grp_a / data_dict["n_obs_grp_a"]
    prop_grp_b: float = n_non_zero_grp_b / data_dict["n_obs_grp_b"]

    prop_max: float = max(prop_grp_a, prop_grp_b)
    prop_mean: float = np.mean([prop_grp_a, prop_grp_b])

    # Number of observations to keep in truncated groups
    n_truncated_grp_a: int = round(prop_max * data_dict["n_obs_grp_a"])
    n_truncated_grp_b: int = round(prop_max * data_dict["n_obs_grp_b"])
    indices_seq: list = range(n_truncated_grp_a)
    n_ziw: float = (n_truncated_grp_a + n_truncated_grp_b + 1) / 2

    # Truncate arrays by removing certain zeros and concatenate arrays in a single one
    non_zero_array = grp_a_counts[np.nonzero(grp_a_counts)]
    zero_array = np.repeat([0], n_truncated_grp_a - n_non_zero_grp_a)
    truncated_counts = np.concatenate((zero_array, non_zero_array))

    non_zero_array = grp_b_counts[np.nonzero(grp_b_counts)]
    zero_array = np.repeat([0], n_truncated_grp_b - n_non_zero_grp_b)
    truncated_counts = np.concatenate((truncated_counts, zero_array, non_zero_array), dtype = float)
    
    # Mean of the modified Wilcoxon rank sum statistic
    n_trun = n_truncated_grp_a + n_truncated_grp_b + 1
    ranks = n_trun - rankdata(truncated_counts, method = "average")
    r: float = ranks[indices_seq].sum() #np.sum(ranks[indices_seq]) not fastest
    
    s: float = r - n_truncated_grp_a * n_trun/2    

    variance: float = calculate_variance_ziw(data_dict, prop_mean, data_dict["n_tot"])
    
    # Modified Wilcoxon rank sum statistic
    if variance == 0:
        w: float = 0
    else:
        w: float = s / math.sqrt(variance)

    return tag, np.round(w, N_ROUND_TEST), 0


# Perform variance and coefficient of variation (CV)
def calculate_variance_and_cv(tag, data_chunk):
    numeric_values = pd.to_numeric(data_chunk.loc[tag], errors='coerce')
    numeric_values = numeric_values.dropna()

    if not numeric_values.empty:
        mean_value = np.mean(numeric_values)
        std_dev_value = np.std(numeric_values)
        variance_value = np.var(numeric_values)

        if mean_value != 0:
            cv_value = std_dev_value / mean_value

            tag_values = ' '.join(
                f"{round(float(x), 2):g}" if isinstance(x, (int, float, np.number)) or str(x).replace('.', '', 1).isdigit() else str(x)
                for x in data_chunk.loc[tag].values
            )

            return variance_value, cv_value, tag_values

    return None


def calculate_pre_statistics_for_ziw(data_dict: dict) -> dict:
    # Calculate some statistics required to perform ZIW
    data_dict["n_obs_grp_a"] = sum(1 for sample in data_dict.values() if sample == "A")
    data_dict["n_obs_grp_b"] = sum(1 for sample in data_dict.values() if sample == "B")
    data_dict["n_tot"] = data_dict["n_obs_grp_a"] + data_dict["n_obs_grp_b"]
    data_dict["n_ziw"] = (data_dict["n_tot"] + 1) / 2
    data_dict["product_n_a_n_b"] = data_dict["n_obs_grp_a"] * data_dict["n_obs_grp_b"]
    data_dict["sum_square_n_a_n_b"] = data_dict["n_obs_grp_a"]**2 + data_dict["n_obs_grp_b"]**2
    data_dict["sqrt_product_n_a_n_b"] = math.sqrt(data_dict["n_obs_grp_a"] * data_dict["n_obs_grp_b"])
    
    return data_dict



# Function to create the data dictionary from the condition file
def create_data_dict(condition_file, test_type):
    data_dict = {}
    conditions = {}

    with open(condition_file, 'r') as file:
        for line in file:
            row = line.strip().split()
            sample_id = row[0]
            condition = row[1]
            conditions.setdefault(condition, []).append(sample_id)

    assigned_condition = 'A'
    for condition, samples in sorted(conditions.items()):
        for sample_id in samples:
            data_dict[sample_id] = assigned_condition

        assigned_condition = 'B' if assigned_condition == 'A' else 'A'

    if test_type == "ziw":
        data_dict = calculate_pre_statistics_for_ziw(data_dict)      

    return data_dict



# Normalize function
def normalize(chunk, design_kmer_nb_file, header_row, norm_factor):
    # Read design_kmer_nb_file as a DataFrame
    design_kmer_nb = pd.read_csv(design_kmer_nb_file, delimiter=' ')
    # Add header for the chunks
    chunk.columns = header_row   
    # If first line is equal to the header, remove the first line
    if all(chunk.iloc[0] == header_row):
        chunk = chunk.iloc[1:]
    # Convert design_kmer_nb to a dictionary
    kmer_nb_dict = dict(zip(design_kmer_nb.iloc[:, 0], design_kmer_nb.iloc[:, 1]))

    # Apply normalization factors to each column
    for column in chunk.columns:
        if column in kmer_nb_dict:
            normalization_factor = norm_factor / kmer_nb_dict[column]
            # Apply normalization factor to numeric values only
            chunk[column] = np.where(pd.notnull(chunk[column]), np.round(chunk[column] * normalization_factor, N_ROUND_NORM), chunk[column])

    return chunk



# Work function for the pool of processes
def work_for_parallel_processes(label_dict, data_chunk, cpm_normalization, header, test_type, covariates_df, norm_factor_c):
    if cpm_normalization:
        # Normalize the data chunk
        normalized_chunk = normalize(data_chunk, cpm_normalization, header, norm_factor_c) 
        grp_a_samples = [sample_id for sample_id, condition in label_dict.items() if condition == 'A']
        grp_b_samples = [sample_id for sample_id, condition in label_dict.items() if condition == 'B']

        grp_a_data = normalized_chunk[grp_a_samples]
        grp_b_data = normalized_chunk[grp_b_samples]
    else:
        grp_a_data = data_chunk[[sample_id for sample_id, condition in label_dict.items() if condition == 'A']]
        grp_b_data = data_chunk[[sample_id for sample_id, condition in label_dict.items() if condition == 'B']]
        normalized_chunk = data_chunk
    results = [] 
    
    if test_type == 'pitest':
        for tag in data_chunk.index:
            result = perform_pitest(tag, grp_a_data, grp_b_data)
            if result is not None:
                tag_values = ' '.join(
                    f"{round(float(x), 2):g}" if isinstance(x, (int, float, np.number)) or str(x).replace('.', '', 1).isdigit() else str(x)
                    for x in data_chunk.loc[tag].values
                )
                results.append((tag_values , abs(result[1]), result[2]))                

    elif test_type == 'anova':
        for tag in data_chunk.index:
            result = perform_anova(tag, grp_a_data, grp_b_data, covariates_df)
            if result is not None:
                tag_values = ' '.join(
                    f"{float(x):.2f}".rstrip('0').rstrip('.') if isinstance(x, (int, float, np.number)) or str(x).replace('.', '', 1).isdigit() else str(x)
                    for x in data_chunk.loc[tag].values
                )
                results.append((abs(result[1]), tag_values, result[2], result[3]))
    
    elif test_type == 'ttest':
        for tag in data_chunk.index:
            result = perform_ttest(tag, grp_a_data, grp_b_data)
            if result is not None:
                tag_values = ' '.join(
                    f"{float(x):.2f}".rstrip('0').rstrip('.') if isinstance(x, (int, float, np.number)) or str(x).replace('.', '', 1).isdigit() else str(x)
                    for x in data_chunk.loc[tag].values
                )
                results.append((abs(result[1]), tag_values, result[2], result[3]))

    elif test_type == 'wilcoxon':
        for tag in data_chunk.index:
            result= perform_wilcoxon_test(tag, grp_a_data,grp_b_data)
            if result is not None:
                tag_values = ' '.join(
                    f"{float(x):.2f}".rstrip('0').rstrip('.') if isinstance(x, (int, float, np.number)) or str(x).replace('.', '', 1).isdigit() else str(x)
                    for x in data_chunk.loc[tag].values
                )
                results.append((abs(result[1]), tag_values, result[2], result[3]))

    elif test_type == "ziw":
        for tag in data_chunk.index:
            result = perform_ziw(tag, grp_a_data, grp_b_data, label_dict)
            tag_values = ' '.join(
                f"{round(float(x), 2):g}" if isinstance(x, (int, float, np.number)) or str(x).replace('.', '', 1).isdigit() else str(x)
                for x in data_chunk.loc[tag].values
            )
            results.append((abs(result[1]), tag_values))
            # WIP
            #kmer_values = data_chunk.loc[tag][1:].astype(float).round(1)
            #tag_values = ' '.join(kmer_values.astype(str).tolist())
            #results.append((abs(result[1]), tag_values)



    elif test_type == "variance":  # "coefficient_variation"
        for tag in normalized_chunk.index:
            result = calculate_variance_and_cv(tag, normalized_chunk)
            if result is not None:
                variance_value, cv_value, tag_values = result
                results.append((variance_value, cv_value, tag_values))
            

        

    
    logging.info(f"Chunk processed: {len(data_chunk)} rows")
    return results
