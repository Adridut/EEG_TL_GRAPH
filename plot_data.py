import matplotlib.pyplot as plt
import os, os.path
from pathlib import Path
import numpy as np


def plotTrial(raw_data, aligned_data):
    '''
    Input: raw_data and aligned_data with shape (n_trials, channels, timepoints)
    Plot of raw and aligned data for one trial to observe the effect of the alignment
    '''
    _, channels, _ = np.shape(raw_data)
    _, axs = plt.subplots(channels, 1)
    print((raw_data[0][:][:]))
    
    for channel in range(channels):
        axs[channel].plot(raw_data[0,channel,:])
        axs[channel].plot(aligned_data[0,channel,:])

    axs[channels-1].set_xlabel('Time (ms)')
    plt.show()




if __name__ == "__main__":
    subject = 1
    raw_data = dict(np.load('./data/raw/patient'+str(subject)+'.npz'))
    aligned_data = dict(np.load('./data/aligned/patient'+str(subject)+'.npz'))  
    plotTrial(raw_data['data'], aligned_data['data'])