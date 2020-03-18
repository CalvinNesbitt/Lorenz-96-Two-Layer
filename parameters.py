# Dictionary of Parameters for L63 Code
# These will be loaded
import pickle

param_dict = {
    't0': 0,
    'tau': 0.1, # How long before QR
    'kA': int(5.e4), # BLV Convergence
    'kB': int(1.e5), # QR steps. Note if you want k samples, need to do L * k steps
    'kC': int(5.e4), # CLV Convergence
    'eps': 0.01,
    'a': 10.0,
    'b': 8/3,
    'c': 28.0,
    'where': [0.1, 0.1, 0.1]
}

# Saving dictionary as a binary file for later use in plotting work
outfile = open('parameters','wb')
pickle.dump(param_dict,outfile)
outfile.close()
