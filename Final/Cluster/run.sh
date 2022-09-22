#!/bin/bash
#SBATCH --job-name=seg_smb
#SBATCH --partition=biggpu
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
conda activate resnetEnv
echo "Hello, we are doing a job now, seg smb, wide"
pwd
python resnet.py
echo "------------------------------------------------------------------------"
echo "Job ended on" `date`
echo "------------------------------------------------------------------------"