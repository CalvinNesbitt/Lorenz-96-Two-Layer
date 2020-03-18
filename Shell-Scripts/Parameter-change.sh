# Shell script for copying parameters and submissions scripts to cx1

local=/Users/cfn18/Documents/PhD-Work/Dynamical-Systems/Lyapunov-Analysis/Example-Computations/Ginelli-L63
shell=$local/Shell-Scripts/

scp $local/*.py $shell/*.sh cfn18@login.hpc.ic.ac.uk:/rds/general/user/cfn18/home/Ginelli-L63
