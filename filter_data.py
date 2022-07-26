import os, os.path
from pathlib import Path
import numpy as np
import scipy.signal 

def bandpass(data, lo, hi, sample_rate):
    '''
    Designs and applies a bandpass filter to the signal.
    
    Parameters
    ----------
    trials : 3d-array (trials x channels x samples)
        The EEGsignal
    lo : float
        Lower frequency bound (in Hz)
    hi : float
        Upper frequency bound (in Hz)
    sample_rate : float
        Sample rate of the signal (in Hz)
    
    Returns
    -------
    trials_filt : 3d-array (trials x channels x samples)
        The bandpassed signal
    '''

    # The iirfilter() function takes the filter order: higher numbers mean a sharper frequency cutoff,
    # but the resulting signal might be shifted in time, lower numbers mean a soft frequency cutoff,
    # but the resulting signal less distorted in time. It also takes the lower and upper frequency bounds
    # to pass, divided by the niquist frequency, which is the sample rate divided by 2:
    a, b = scipy.signal.iirfilter(6, [lo/(sample_rate/2.0), hi/(sample_rate/2.0)])

    # Applying the filter to each trial
    data_filt = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
    for i in range(data.shape[0]):
        data_filt[i,:,:] = scipy.signal.filtfilt(a, b, data[i,:,:], axis=1)
    
    return data_filt

if __name__ == "__main__":
    subjectNumber = len([name for name in os.listdir("./data/aligned") if os.path.isfile(os.path.join("./data/aligned", name))])
    Path("./data/filtered").mkdir(parents=True, exist_ok=True) 
    for subject in range(1, subjectNumber+1):
        data = dict(np.load('./data/aligned/patient'+str(subject)+'.npz')) 
        print('Filtering subject {}'.format(subject))
        data['data'] = bandpass(data['data'], 4, 40, 250)
        np.savez('./data/filtered/patient'+str(subject), **data)
