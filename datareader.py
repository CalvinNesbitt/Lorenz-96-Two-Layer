class DataRead:
    """Class for reading data from a L63 Lyapunov Analysis run
    Parameter, directory, string: Directory containing data files.
    """

    def __init__(self, dirName):
        self.directory = dirName
        self.FTBLE = np.load(dirName + '/Data/FTBLE.npy')
        self.FTCLE = np.load(dirName + '/Data/FTCLE.npy')
        self.CLE = np.load(dirName + '/Data/CLE.npy')
        self.BLE = np.load(dirName + '/Data/BLE.npy')
        self.CLVs = np.load(dirName + '/Data/CLVs.npy')
        self.BLVs = np.load(dirName + '/Data/BLVs.npy')
        self.solution = np.load(dirName + '/Data/solution.npy')
        infile = open(self.directory +'/parameters','rb')
        self.param_dict = pickle.load(infile)
        infile.close()

    def who(self):
        "Prints the parameters associated with the run."
        print(f'The parameters used to generate this data were:\n{self.param_dict}')
    
