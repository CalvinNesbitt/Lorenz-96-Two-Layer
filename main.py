# Implementation of the Ginelli algorithm for Two Layer L96
import numpy as np
import time as tm
import pickle
from tqdm import tqdm
import xarray as xr
import sys
from ginelli96 import Ginelli96
import os

# Choosing h
h = 0.765625
print(os.getcwd())

# Ginelli Algorithm
test = Ginelli96(h=h)
test._run_ginelli()
test.save_data()
