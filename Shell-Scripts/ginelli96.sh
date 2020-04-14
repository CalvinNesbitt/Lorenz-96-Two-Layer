#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=8:mem=90gb
#PBS -N h-moonshot
#PBS -J 1-5

# ---------------------------------------------------------------
# Shell Script for Running Ginelli Algoritmh with L96 on Cluster
# - Edit experiment name and whether doing array job or single job
# ---------------------------------------------------------------

job_type='array' # 'single' or 'array'. Add 'PBS -J 1-N' for jobs
experiment_name=h-moonshot

module load anaconda3/personal
source activate personalpy3
date

# Directory where results are copied too
notebook_directory=/rds/general/user/cfn18/home/Lyapunov-Analysis/Lorenz-96-Two-Layer
output_directory=$notebook_directory/$experiment_name

if [ $job_type == 'array' ]
then
    # Array Jobs Setup #PBS -J 1-N
    working_directory=$PBS_O_WORKDIR/$experiment_name
    mkdir $working_directory
    job_directory=$working_directory/$PBS_ARRAY_INDEX
    mkdir $job_directory
    pwd
    cp -r $PBS_O_WORKDIR/Ginelli-L96 $job_directory
    cd $job_directory/Ginelli-L96
    python ginelli.py $PBS_ARRAY_INDEX
    mkdir $output_directory
    mkdir $output_directory/$PBS_ARRAY_INDEX
    mv ginelli/step5 ginelli/timings.p $output_directory/$PBS_ARRAY_INDEX
elif [ $job_type == 'single' ]
then
    cd $PBS_O_WORKDIR/Ginelli-L96
    python ginelli.py $PBS_ARRAY_INDEX
    cp -r ginelli $PBS_O_WORKDIR
    cd $PBS_O_WORKDIR
    source deactivate

    # Copying Data We want over to notebook area
    mv ginelli/step5 ginelli/timings.p $output_directory
else
    echo 'Didnt specify job type correctly.'
fi

# Clean up
#rm -r $PBS_O_WORKDIR/Ginelli-L96 # Deleting run version of the model
