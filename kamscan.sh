#!/bin/bash

#PBS -l select=1:ncpus=6:mem=100gb
#PBS -l walltime=150:00:00
source /home/safa.maddouri/miniconda3/bin/activate kamscan

# ----- Parameters ----- #
SCRIPT=/store/EQUIPES/SSFA/MEMBERS/safa.maddouri/KAMSCAN/scripts/kamscan.py
INPUT=/store/EQUIPES/SSFA/MEMBERS/safa.maddouri/KAMSCAN/1000mono
OUT=/store/EQUIPES/SSFA/MEMBERS/safa.maddouri/KAMSCAN/RESULTS
DESIGN=/store/EQUIPES/SSFA/MEMBERS/safa.maddouri/KAMSCAN/3cond
DESIGN_CPM=/store/EQUIPES/SSFA/MEMBERS/safa.maddouri/KAMSCAN/design_kmers_nb_per_patient
#NORM_FACTORS=
CHUNK=10000
TOP=10
THREADS=12
TEST=ziw
# ----- Parameters ----- #


/usr/bin/time -v python3 $SCRIPT -i $INPUT -o $OUT -d $DESIGN -t $TOP -c $CHUNK -p $THREADS -m $DESIGN_CPM --test_type $TEST 

