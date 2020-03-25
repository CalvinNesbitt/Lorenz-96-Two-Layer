#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=10gb
#PBS -N Ginelli96_h=0.765625

# Running Ginelli algorithm

module load anaconda3/personal
source activate personalpy3
date
cd $PBS_O_WORKDIR/Ginelli-L96
python main.py
cp *.nc $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
source deactivate

# Copying Data over to plotting area
cp *.nc /rds/general/user/cfn18/home/Lyapunov-Analysis/Lorenz-96-Two-Layer/.

# Clean up
rm -r $PBS_O_WORKDIR/Ginelli-L96 # Deleting run version of the model
rm __pycache__
