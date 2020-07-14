#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=10:mem=80gb
#PBS -N Julia-Verification

# Make directory to copy model in to, submit job from there
NOW=$(date +"%Y-%m-%d-%T")
run_directory=$EPHEMERAL/Ginelli-L96-Julia/$NOW
mkdir -p $run_directory
cp $HOME/Ginelli-Julia-Verification/Lorenz96multi.jl $run_directory
cd $run_directory

$HOME/julia-1.4.2/bin/julia Lorenz96multi.jl $run_directory/Lorenz96multi.jl
