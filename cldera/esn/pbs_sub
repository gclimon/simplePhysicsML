#!/bin/bash -l
#PBS -N test
#PBS -A UMIC0069
#PBS -l select=1:ncpus=1:mem=200GB:ngpus=0
#PBS -l walltime=30:00
#PBS -j oe
#PBS -q casper

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

### Load modules required to run the job
module load intel peak_memusage
module load conda
conda activate my-npl-tf

### Run program
peak_memusage.exe python run.py &> out.txt

### Deactivate NCAR PyLib
conda deactivate
