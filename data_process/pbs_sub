#!/bin/bash -l
#PBS -N mPL_aug
#PBS -A UMIC0069
#PBS -l select=1:ncpus=1:mem=100GB:ngpus=0
#PBS -l walltime=30:00
#PBS -j oe
#PBS -q casper

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

### Load modules required to run the job
module load intel peak_memusage
# module load conda
# conda activate my-npl-tf
module load ncarenv python/3.7.5
ncar_pylib 20200417

### Run program
peak_memusage.exe python run_process.py &> process_mdT_augNoRH.out

### Deactivate NCAR PyLib
deactivate
