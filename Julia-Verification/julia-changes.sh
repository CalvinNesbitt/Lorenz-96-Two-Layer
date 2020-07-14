# Shell script for copying parameters and submissions scripts to cx1 for Julia work

local=/Users/cfn18/Documents/PhD-Work/Dynamical-Systems/Lyapunov-Analysis/Example-Computations/Lorenz-96-Two-Layer/Julia-Verification

scp $local/* cfn18@login.hpc.ic.ac.uk:/rds/general/user/cfn18/home/Ginelli-Julia-Verification
