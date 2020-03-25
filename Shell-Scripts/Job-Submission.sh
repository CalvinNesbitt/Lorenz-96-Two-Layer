# Make directory to copy model in to, submit job from there
NOW=$(date +"%Y-%m-%d-%T")
run_directory=$EPHEMERAL/Ginelli-L96/$NOW
mkdir $run_directory
cp -r $HOME/Ginelli-L96 $run_directory
cd $run_directory
qsub Ginelli-L96/ginelli96.sh
