import os
from tqdm import tqdm
import numpy as np
from scipy import signal

Fs = 100
Cutoff = 0.3
# Apply FIR filter
def firFilter(x):
    if len(x) < 511 * 3:
        if len(x) < 255 * 3:
            if len(x) < 127 * 3:
                return x # If it is too short, return it as is.
            else:
                b = signal.firwin(numtaps=127, cutoff=Cutoff, fs=Fs, pass_zero=True)
        else:
            b = signal.firwin(numtaps=255, cutoff=Cutoff, fs=Fs, pass_zero=True)
    else:
        b = signal.firwin(numtaps=511, cutoff=Cutoff, fs=Fs, pass_zero=True)
    
    x = signal.filtfilt(b, 1, x, axis=0)
    return x

def extractGravity_LAcc(change_indexes, acc): 
    gra = np.zeros(acc.shape)

    gra[:change_indexes[0]+1, 1:] = firFilter(acc[:change_indexes[0]+1, 1:])
    for i in tqdm(range(1, len(change_indexes))):
        gra[change_indexes[i-1]+1:change_indexes[i]+1, 1:] = firFilter(acc[change_indexes[i-1]+1:change_indexes[i]+1, 1:])
    gra[change_indexes[-1]+1:, 1:] = firFilter(acc[change_indexes[-1]+1:, 1:])
    
    lacc = acc - gra

    # timestamp
    gra[:, 0] = acc[:, 0]
    lacc[:, 0] = acc[:, 0]
    return gra, lacc

def exec_convertAcc(config):
    RAWNPY_DIR = 'npy'
    
    training_path = os.path.join(config['data_dir'], RAWNPY_DIR, 'train/')
    val_path = os.path.join(config['data_dir'], RAWNPY_DIR, 'validate/')
    test_path = os.path.join(config['data_dir'], RAWNPY_DIR, 'test/')
    
    training_output_path = os.path.join(config['output_dir'], "train/")
    val_output_path = os.path.join(config['output_dir'], "validate/")
    test_output_path = os.path.join(config['output_dir'], "test/")

    CHANGE_INDEXES = 'Changeindexes.npy'

    print(training_path)
    Gra, LAcc = extractGravity_LAcc(np.load(training_output_path + CHANGE_INDEXES), np.load(training_path + 'Hand/Acc.npy'))
    np.save(training_path + 'Hand/Gra_ver2', Gra)
    np.save(training_path + 'Hand/LAcc_ver2', LAcc)

    print(val_path)
    Gra, LAcc = extractGravity_LAcc(np.load(val_output_path + CHANGE_INDEXES), np.load(val_path + 'Hand/Acc.npy'))
    np.save(val_path + 'Hand/Gra_ver2', Gra)
    np.save(val_path + 'Hand/LAcc_ver2', LAcc)

    print(test_path)
    Gra, LAcc = extractGravity_LAcc(np.load(test_output_path + CHANGE_INDEXES), np.load(test_path + 'Acc.npy'))
    np.save(test_path + 'Gra_ver2', Gra)
    np.save(test_path + 'LAcc_ver2', LAcc)    