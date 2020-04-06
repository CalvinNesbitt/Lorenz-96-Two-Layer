# Utilities
# Various useful Functions

# Processing Utilities

def gl96process(raw, save='None'):
    "Remove nans, adds spectra, adds max positive LE"
    processed = raw.dropna(dim ='time')
    processed.update({'BLE': processed.FTBLE.mean(dim ='time')})
    processed.update({'CLE': processed.FTCLE.mean(dim ='time')})
    
    # Max positive LE index
    kd = processed.CLE.where(processed.CLE > 0).dropna(dim = 'le_index').le_index.max().item()
    processed.attrs.update({'kd': kd})
    
    if (save != None):
        processed.to_netcdf(save)
        
    return processed

def LE_average(processed_data, L, save='None'):
    """Returns xarray with FTLEs being averaged over long times. DESIGNED FOR PROCESSED DATA"""
    data = processed_data
    FTBLE = data.FTBLE.rolling(time = L).mean()[L - 1::L]
    FTCLE = data.FTCLE.rolling(time = L).mean()[L - 1::L]
    avg_data = data.sel(time = FTBLE.time) # Picking out data and average time steps.
    avg_data.FTBLE.values = FTBLE.values # Updating LEs
    avg_data.FTCLE.values = FTCLE.values # Updating LEs

    # Adding Attributes
    avg_data.attrs.update({'L' : L})
    Ltau = L * avg_data.attrs['tau']
    avg_data.attrs.update({'Avg Time' : Ltau})
    
    if (save != None):
        avg_data.to_netcdf(save)
    
    return avg_data

# Plotting Utilities

def spectra(data_list, variable, ftle='C', save = 'None'):
    """Plot of all LE means for a changing variable.
    param, variable, variable being changed.
    param, save, string of save name"""
    
    for data in data_list:
        Ltau = data.attrs['tau'] * 1 # for now L = 1
        attr = data.attrs[variable]
        
        # FTCLE or FTBLE?
        if (ftle == 'C'):
            spectra = data.FTCLE.mean(dim='time', skipna=True)
            label = f'{variable}$={attr:.2f}$'
        else: # then FTBLE
            spectra = data.FTBLE.mean(dim='time', skipna=True)
            label = f'{variable}$={attr:.2f}$'
            
        plt.plot(spectra, label = label)
    
    plt.legend()
    plt.title(f'FT{ftle}LE Means, $L\\tau =' + f"{Ltau:.2f}" + '$')
    plt.xlabel('LE Index') 
    
    if (save == 'None'):
        plt.show()
    else:
        print('Saving')
        plt.savefig(save, dpi=1200)
        plt.show()

def density(data_list, le, ftle='C', save = 'None'):
    """Plots Density using KDE, designed to compare for different h"""
    for data in data_list:
        Ltau = data.attrs['tau'] * 1 # for now L = 1
        h = data.attrs[variable]
        
        # FTCLE or FTBLE?
        if (ftle == 'C'):
            FTLE = data.FTCLE.dropna(dim= 'time', how= 'all').sel(le_index = le).values
            label = f'$h={h:.2f}$'
        else: # then FTBLE
            FTLE = data.FTCLE.dropna(dim= 'time', how= 'all').sel(le_index = le).values
            label = f'$h={h:.2f}$'
            
        x_d = np.linspace(FTLE.min() - 1, FTLE.max() + 1, 100) # Grid we evaluate PDF on
        kde = gaussian_kde(FTLE) # KDE. Using Gaussian kernels with default bandwidth, don't know how it decided bw?
        pdf = kde.evaluate(x_d)
        plt.plot(x_d, pdf, label = label) 
        
    plt.legend()
    plt.xlabel(f'FT{ftle}LE')
    plt.ylabel('$\\rho$')
    plt.title(f'FT{ftle}LE {le} Density Comparison, $L\\tau =' + f"{Ltau:.2f}" + '$')
    
    if (save == 'None'):
        plt.show()

    else:
        print('Saving')
        plt.savefig(f'Effect-of-Coupling-Strength/FTCLE-Densities/FT{ftle}LE_{le}.png', dpi=1200)
        plt.show()
        




    