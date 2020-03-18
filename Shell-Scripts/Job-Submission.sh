# Make directory to copy model in to, submit job from there
NOW=$(date +"%Y-%m-%d-%T")
run_directory=$EPHEMERAL/Ginelli-L63/$NOW
mkdir $run_directory
cp -r $HOME/Ginelli-L63 $run_directory
cd $run_directory
cp Ginelli-L63/parameters.py $run_directory
qsub Ginelli-L63/ginelli.sh
