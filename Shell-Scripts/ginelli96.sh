#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=8:mem=12gb
#PBS -N Test

# ---------------------------------------------------------------
# Shell Script for Running Ginelli Algoritmh with L96 on Cluster
# ---------------------------------------------------------------

# Where output of experiment will be saved. CARE IF USING ARRAY JOBS
experiment_name=clean-test
notebook_directory=/rds/general/user/cfn18/home/Lyapunov-Analysis/Lorenz-96-Two-Layer
output=$notebook_directory/$experiment_name
mkdir $output

# Running Ginelli algorithm
module load anaconda3/personal
source activate personalpy3
date
cd $PBS_O_WORKDIR/Ginelli-L96
python ginelli.py
cp -r ginelli $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
source deactivate

# Copying Data We want over to notebook area
mv ginelli/step5 ginelli/timings.p $output

# Clean up
#rm -r $PBS_O_WORKDIR/Ginelli-L96 # Deleting run version of the model
