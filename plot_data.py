import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def plotTrial(raw_data, aligned_data):
    '''
    Input: raw_data and aligned_data with shape (n_trials, channels, timepoints)
    Plot of raw and aligned data for one trial to observe the effect of the alignment
    '''
    _, channels, _ = np.shape(raw_data)
    _, axs = plt.subplots(round(channels/2), 2, figsize=(10,10))

    col = 0 
    row = 0   

    for channel in range(channels):
        if channel == (channels)/2:
            col = 1
            row = 0

        axs[row, col].plot(raw_data[0,channel,:])
        axs[row, col].plot(aligned_data[0,channel,:])
        row += 1

    axs[round((channels-1)/2), 0].set_xlabel('Time (ms)')
    axs[round((channels-1)/4), 0].set_ylabel('Magnitude')
    plt.savefig('./figs/raw_vs_aligned_comparison.png')
    plt.show()




if __name__ == "__main__":
    Path("./figs").mkdir(parents=True, exist_ok=True) #create figs folder if they don't exist
    subject = 1
    raw_data = dict(np.load('./data/raw/patient'+str(subject)+'.npz'))
    aligned_data = dict(np.load('./data/aligned/patient'+str(subject)+'.npz'))  
    plotTrial(raw_data['data'], aligned_data['data'])