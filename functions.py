import functools
import shutil
import time
import statistics
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import rankdata
from scipy.stats import mannwhitneyu
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
N_BASE_FACTOR = 10**10
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
    


# Perform T-test 

def perform_ttest(tag, grp_a_data, grp_b_data):
    grp_a_values = np.log(grp_a_data.loc[tag].values + 1)
    grp_b_values = np.log(grp_b_data.loc[tag].values + 1)  
    t_statistic, p_value = ttest_ind(grp_a_values, grp_b_values)
    if not np.isnan(t_statistic):  # Check for NaN 
        return tag, np.round(t_statistic, N_ROUND_TEST), p_value
        
       
# Perform Wilcoxon Test

def perform_wilcoxon_test(tag, grp_a_data, grp_b_data):
    grp_a_data = grp_a_data.loc[tag].values
    grp_b_data = grp_b_data.loc[tag].values
    # Perform the Wilcoxon test
    statistic, p_value = mannwhitneyu(grp_a_data, grp_b_data)
    return tag, np.round(statistic, 2), p_value

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
    numeric_values = numeric_values.dropna()  # Supprimer les valeurs non numériques

    if not numeric_values.empty:
        mean_value = np.mean(numeric_values.values.astype(float))
        std_dev_value = np.std(numeric_values.values.astype(float))

        if mean_value != 0:  # Avoid division by zero
            cv_value = std_dev_value / mean_value
            tag_values = ' '.join(data_chunk.loc[tag].astype(str))
            return cv_value, tag_values
    return None

def calculate_pre_statistics_for_ziw(data_dict: dict) -> dict:
    # Calculate some statistics required to perform ZIW
    data_dict["n_obs_grp_a"] = sum(1 for patient in data_dict.values() if patient == "A")
    data_dict["n_obs_grp_b"] = sum(1 for patient in data_dict.values() if patient == "B")
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
            patient_id = row[0]
            condition = row[1]
            conditions.setdefault(condition, []).append(patient_id)

    assigned_condition = 'A'
    for condition, patients in sorted(conditions.items()):
        for patient_id in patients:
            data_dict[patient_id] = assigned_condition

        assigned_condition = 'B' if assigned_condition == 'A' else 'A'

    if test_type == "ziw":
        data_dict = calculate_pre_statistics_for_ziw(data_dict)      

    return data_dict



# Normalize function
def normalize(chunk, design_kmer_nb_file, header_row):
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
            normalization_factor = np.round((N_BASE_FACTOR / kmer_nb_dict[column]), N_ROUND_NORM)
            # Apply normalization factor to numeric values only
            chunk[column] = np.where(pd.notnull(chunk[column]), chunk[column] * np.round(normalization_factor, N_ROUND_NORM), chunk[column])

    return chunk



# Work function for the pool of processes
def work_for_parallel_processes(label_dict, data_chunk, cpm_normalization, header, test_type):
    if cpm_normalization:
        # Normalize the data chunk
        normalized_chunk = normalize(data_chunk, cpm_normalization, header) 
        grp_a_patients = [patient_id for patient_id, condition in label_dict.items() if condition == 'A']
        grp_b_patients = [patient_id for patient_id, condition in label_dict.items() if condition == 'B']

        grp_a_data = normalized_chunk[grp_a_patients]
        grp_b_data = normalized_chunk[grp_b_patients]
    else:
        grp_a_data = data_chunk[[patient_id for patient_id, condition in label_dict.items() if condition == 'A']]
        grp_b_data = data_chunk[[patient_id for patient_id, condition in label_dict.items() if condition == 'B']]
        normalized_chunk = data_chunk
    results = [] 
    if test_type == 'pitest':
        for tag in data_chunk.index:
            result = perform_pitest(tag, grp_a_data, grp_b_data)
            if result is not None:
                tag_values = ' '.join(data_chunk.loc[tag].values.astype(str))
                results.append((tag_values,abs(result[1]),result[2]))

    elif test_type == 'ttest':
        for tag in data_chunk.index:
            result = perform_ttest(tag, grp_a_data, grp_b_data)
            if result is not None:
                tag_values = ' '.join(data_chunk.loc[tag].values.astype(str))
                results.append((abs(result[1]), tag_values))

    elif test_type == 'wilcoxon':
        for tag in data_chunk.index:
            result= perform_wilcoxon_test(tag,grp_a_data,grp_b_data)
            if result is not None:
                tag_values = ' '.join(data_chunk.loc[tag].values.astype(str))
                results.append((abs(result[1]), tag_values))

    elif test_type == "ziw":
        for tag in data_chunk.index:
            result = perform_ziw(tag, grp_a_data, grp_b_data, label_dict)
            tag_values = ' '.join(data_chunk.loc[tag].values.astype(str))
            results.append((abs(result[1]), tag_values))
            # WIP
            #kmer_values = data_chunk.loc[tag][1:].astype(float).round(1)
            #tag_values = ' '.join(kmer_values.astype(str).tolist())
            #results.append((abs(result[1]), tag_values)
    elif test_type == "variance":  # "coefficient_variation"
        for tag in normalized_chunk.index:
            result = calculate_variance_and_cv(tag, normalized_chunk)
            if result is not None:
                cv_value, tag_values = result
                results.append((cv_value, tag_values))
    
    logging.info(f"Chunk processed: {len(data_chunk)} rows")
    return results
