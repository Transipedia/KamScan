#!/usr/bin/env python3
"""
Author  : Hugues HERRMANN
Purpose : Perform the zero-inflated wilcoxon on every row of the file
Wang, Wanjie, Eric Z. Chen, et Hongzhe Li. « Truncated Rank-Based Tests for Two-Part Models with Excessive Zeros and Applications to Microbiome Data ». arXiv, 13 octobre 2021. http://arxiv.org/abs/2110.05368.
Date    : 18.08.2022
Usage   : python3 apply_zero_inflated_wilcoxon.py -d design.tsv -c counts.tsv.gz -o out.tsv
"""

import time
import math
import statistics
from scipy.stats import norm
from scipy.stats import rankdata
import gzip
#import cProfile
import numpy as np
import sys


# --------------------------------------
#
#   ARGUMENTS
#
# --------------------------------------
try:
    DESIGN = sys.argv[sys.argv.index("-d") + 1]
except:    
    print ("""ERROR: the design file is missing. It should look like and be gziped:
           sample1    A
           sample2    B""")
    sys.exit()

try:
    COUNTS = sys.argv[sys.argv.index("-c") + 1]
except:    
    print ("ERROR: the count file is missing. It should be gziped.")
    sys.exit()

try:
    OUT_RANK = sys.argv[sys.argv.index("-o") + 1]
except:
    print ("ERROR: output name argument missing.")
    sys.exit()


# --------------------------------------
#
#   VARIABLES
#
# --------------------------------------
"""
DESIGN = "/mnt/beegfs/scratch/h_herrmann/parcours_test/design.tsv"
COUNTS = "/mnt/beegfs/scratch/h_herrmann/parcours_test/test.tsv.gz"
OUT_RANK = "/mnt/beegfs/scratch/h_herrmann/parcours_test/out_test.tsv"
"""
SEPARATOR_COUNTS = "\t"
SEPARATOR_DESIGN = " "
# P-value or quantile thresholds for signifiance
P_THRESHOLD: float = 0.05
Q_NORM: float = 1.96

# --------------------------------------
#
#   FUNCTIONS
#
# --------------------------------------
def parse_design(design_file: str, separator: str) -> list:
    # Return 2 lists with sample name of each group
    # @design_file:  str, file to parse
    # @separator:    str, separator of the line (ex: "\t")
    sample_name_index: int = 0
    grp_name_index: int = 1
    
    grp_a: list = []
    grp_b: list = []
    grp_a_name: str = ""
        
    with open(design_file, mode = "r") as design:
        for line in design:
            line_list = line.strip().split(separator)
            
            grp_name = line_list[grp_name_index]
            if grp_a_name == "":
                grp_a_name = grp_name
                
            if grp_name == grp_a_name:
                grp_a.append(line_list[sample_name_index])
            else:
                grp_b.append(line_list[sample_name_index])

    return [grp_a, grp_b]
        
        
def parse_counts_row(row: str, separator: str) -> list:
    # Return a numpy array of counts, the feature name is removed
    # @row:       a string with the name of the feature (gene, k-mer, etc) and its counts
    # @separator: a string, separator of the line (ex: "\t")
    counts = row.split(separator)
    # Remove the k-mer sequence
    feature_name: str = counts[0]
    del counts[0]
    
    return [feature_name, np.array(counts, dtype = float)]


############## EXEMPLE FUNCTION ####################
def calculate_variance_ziw(prop_mean: float, n_obs_grp_a: int, n_obs_grp_b: int, n: int) -> float:
    # Calculate variance of the modified wilcoxon rank sum statistic
    # @prop_mean: a float, average proportion of zeros in both groups
    prop_mean_complement = 1 - prop_mean
    
    # Variance in the first group
    v1 = prop_mean * prop_mean_complement
    var1 =  n_obs_grp_a * n_obs_grp_b * prop_mean * v1 * (n * prop_mean + 3 * prop_mean_complement / 2) + \
    (n_obs_grp_a**2 + n_obs_grp_b**2) * v1**2 * (5/4) + 2 * math.sqrt(2 / math.pi) * prop_mean * (n * v1)**(1.5) \
    * math.sqrt(n_obs_grp_a * n_obs_grp_b)
    var1 = var1 / 4
    
    # Variance in the second group
    var2 = n_obs_grp_a * n_obs_grp_b * prop_mean**2 * (n * prop_mean + 1)/12
    
    # Variance of the modified wilcoxon rank sum statistic
    variance = var1 + var2

    return variance


def calculate_ziw_statistic(grp_a_counts: list, grp_b_counts: list, n_obs_grp_a: int, n_obs_grp_b, n: int) -> float:
    # @grp_a_counts:   numpy array
    # @grp_b_counts:  numpy array
    # Number of non-zero observations by group
    n_non_zero_grp_a = np.count_nonzero(grp_a_counts)
    n_non_zero_grp_b = np.count_nonzero(grp_b_counts)

    # Proportion of non-zeros by group
    prop_grp_a: float = n_non_zero_grp_a / n_obs_grp_a
    prop_grp_b: float = n_non_zero_grp_b / n_obs_grp_b

    prop_max: float = max(prop_grp_a, prop_grp_b)
    prop_mean: float = statistics.mean([prop_grp_a, prop_grp_b])
    
    # Number of observations to keep in truncated groups
    n_truncated_grp_a: int = round(prop_max * n_obs_grp_a)
    n_truncated_grp_b: int = round(prop_max * n_obs_grp_b)
    indices_seq: list = range(n_truncated_grp_a)
    n_ziw: float = (n_truncated_grp_a + n_truncated_grp_b + 1) / 2
    
    # Truncate arrays by reloving certain zeros and concatenate arrays in a single one
    non_zero_array = grp_a_counts[np.nonzero(grp_a_counts)]
    zero_array = np.repeat([0], n_truncated_grp_a - n_non_zero_grp_a)
    truncated_counts = np.concatenate((zero_array, non_zero_array))

    non_zero_array = grp_b_counts[np.nonzero(grp_b_counts)]
    zero_array = np.repeat([0], n_truncated_grp_b - n_non_zero_grp_b)
    truncated_counts = np.concatenate((truncated_counts, zero_array, non_zero_array), dtype = float)
    
    # Mean of the modified wilcoxon rank sum statistic
    ranks = n_truncated_grp_a + n_truncated_grp_b + 1 - rankdata(truncated_counts, method = "average")
    r: float = ranks[indices_seq].sum()
    s: float = r - n_truncated_grp_a * n_ziw
    
    variance: float = calculate_variance_ziw(prop_mean, n_obs_grp_a, n_obs_grp_b, n)
    
    # Modified wilcoxon rank sum statistic
    w: float = s / math.sqrt(variance)
        
    return w


def calculate_p_value(w: float) -> float:
    # w:  a float, modified wilcoxon rank sum statistic
    p: float = 2 * (1 - norm.cdf(abs(w)))
    
    return p
############## END EXEMPLE FUNCTION ####################

def main(design_file: str, counts_input: str, output: str):    
    with gzip.open(counts_input, mode = "rt") as counts:
        ### ---------- Process header line
        header_line: str = next(counts).strip()
        header: list = header_line.split("\t")
        del header[0]
        
        # Header of the output file
        selected_features = open(output, "w")
        selected_features.write(header_line + "\tw\n")

        design_parsed = parse_design(design_file, SEPARATOR_DESIGN)
        grp_a_name_list = design_parsed[0]
        grp_b_name_list = design_parsed[1]
        # Get indices of group A and group B
        grp_a_indices: list = []
        for a in grp_a_name_list:
            grp_a_indices.append(header.index(a))

        grp_b_indices: list = []
        for b in grp_b_name_list:
            grp_b_indices.append(header.index(b))
        
        # Variables for ZIW statistics
        n_obs_grp_a: int = len(grp_a_name_list)
        n_obs_grp_b: int = len(grp_b_name_list)
        n: int = n_obs_grp_a + n_obs_grp_b

        ############## EXEMPLE STAT TEST ####################
        ### ---------- Process count lines
        for line in counts:
            line = line.strip()
            parse_row: list = parse_counts_row(line, SEPARATOR_COUNTS)
            feature: str = parse_row[0]
            counts = parse_row[1]

            grp_a_counts = counts[grp_a_indices] #.astype(float)
            grp_b_counts = counts[grp_b_indices] #.astype(float)

            w: float = calculate_ziw_statistic(grp_a_counts, grp_b_counts, n_obs_grp_a, n_obs_grp_b, n)
            # Normale CDF is too slow
            #p: float = calculate_p_value(w)
            
            if abs(w) > Q_NORM:
                selected_features.write(line + "\t" + str(round(abs(w), 5)) + "\n")
        selected_features.close()
        ############## END EXEMPLE STAT TEST ################
        

# --------------------------------------
#
#   MAIN
#
# --------------------------------------
main(DESIGN, COUNTS, OUT_RANK)
#cProfile.run("main(COUNTS)")


