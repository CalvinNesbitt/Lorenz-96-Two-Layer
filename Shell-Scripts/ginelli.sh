#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=4:mem=10gb
#PBS -N tau0.1-test

# Running Ginelli algorithm

module load anaconda3/personal
source activate py3
date
cd $PBS_O_WORKDIR/Ginelli-L63
python parameters.py
python main.py
cp -r  Data $PBS_O_WORKDIR
cp parameters $PBS_O_WORKDIR
cp Timing.txt $PBS_O_WORKDIR
cp time-dir-write.py $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
python time-dir-write.py
source deactivate

# Moving Files we need in to common directory
cp -r Data Timing.txt parameters tau*

# Copying Directory over to plotting area
cp -r tau* /rds/general/user/cfn18/home/Lyapunov-Analysis/Lorenz-63-Example/.

# Clean up
rm -r $PBS_O_WORKDIR/Ginelli-L63 # Deleting run version of the model
rm -r Data Timing.txt parameters
rm time-dir-write.py
rm parameters.py
rm __pycache__
