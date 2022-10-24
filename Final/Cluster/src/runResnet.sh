#!/bin/bash
#SBATCH --job-name=R_kfold
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH -o /home-mscluster/jwacks/src/slurm.%N.%j.out
#SBATCH -e /home-mscluster/jwacks/src/slurm.%N.%j.err
#####################################
echo "------------------------------------------------------------------------"
echo "Job started on" `date`
echo "------------------------------------------------------------------------"
echo Running on $HOSTNAME...
echo Running on $HOSTNAME... >&2

source ~/.bashrc
cd ~/src/
conda activate trainingEnv
source ~/switch-cuda.sh 11.2
echo "We are doing a job now,resnet_kfold"
pwd
python resnet_kfold.py
echo "------------------------------------------------------------------------"
echo "Job ended on" `date`
echo "------------------------------------------------------------------------"