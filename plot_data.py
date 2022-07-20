import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
import os, os.path


def plotTrial(subject, trial):
    '''
    Input: subject and trial id
    Plot of raw and aligned data for one trial to observe the effect of the alignment
    '''

    subject = 1
    raw_data = dict(np.load('./data/raw/patient'+str(subject)+'.npz'))['data']
    aligned_data = dict(np.load('./data/aligned/patient'+str(subject)+'.npz'))['data']  

    _, channels, _ = np.shape(raw_data)
    _, axs = plt.subplots(round(channels/2), 2, figsize=(10,10))

    col = 0 
    row = 0   

    for channel in range(channels):
        if channel == (channels)/2:
            col = 1
            row = 0

        axs[row, col].plot(raw_data[trial,channel,:])
        axs[row, col].plot(aligned_data[trial,channel,:])
        row += 1

    axs[round((channels-1)/2), 0].set_xlabel('Time (ms)')
    axs[round((channels-1)/4), 0].set_ylabel('Magnitude')
    plt.savefig('./figs/subject' + str(subject) + 'trial' + str(trial) + '.png')
    plt.show()

def plotTsneTrials(target):

    '''
    Input: target subject
    Plot of raw data and aligned data for source subjects (blue) and target suject (red)
    '''

    _, axs = plt.subplots(1, 2)
    subjectNumber = len([name for name in os.listdir("./data/raw") if os.path.isfile(os.path.join("./data/raw", name))])

    for i in range(1,subjectNumber+1):
        raw_data = dict(np.load('./data/raw/patient'+str(i)+'.npz'))['data']
        aligned_data = dict(np.load('./data/aligned/patient'+str(i)+'.npz'))['data'] 

        sne = TSNE(n_iter=2000).fit_transform(raw_data[:,:,0])
        sne_ea = TSNE(n_iter=2000).fit_transform(aligned_data[:,:,0])
        if target == i:
            axs[0].scatter(sne_ea[:,0], sne_ea[:,1], c='r')
            axs[1].scatter(sne[:,0], sne[:,1], c='r')
        else:
            axs[0].scatter(sne_ea[:,0], sne_ea[:,1], c='b')
            axs[1].scatter(sne[:,0], sne[:,1], c='b')

    plt.savefig('./figs/target' + str(target) + 'tsne.png')
    plt.show()




if __name__ == "__main__":
    Path("./figs").mkdir(parents=True, exist_ok=True) #create figs folder if they don't exist
    plotTrial(1, 0)
    plotTsneTrials(2)