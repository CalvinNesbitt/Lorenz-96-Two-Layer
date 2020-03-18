# Short script for making a directory with tau and steps parameters in name
import os

# First read tau and steps
import pickle

# Loading parameters from dictionary
infile = open('parameters','rb')
param_dict = pickle.load(infile)
infile.close()

# Assigning parameters for use
tau = param_dict['tau'] # How many steps TLE pushes you forward
kA = param_dict['kA']
kB = param_dict['kB']
kC = param_dict['kC']

steps = kA + kB + kC

# Make a directory with tau and steps in name
dirName = f'tau{tau}-steps{steps:.1E}KUPTSOV'

if (os.path.exists(dirName)): # Checking if directory with same name exists
    print(f'There is already a directory called {dirName}. Not making another.')
    quit()

os.mkdir(dirName)
