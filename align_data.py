import numpy as np
from scipy.linalg import sqrtm, inv
import os, os.path
from pathlib import Path

def euclidean_alignment(X):
    '''
    Input: X is EEG data for one subject (n_trials, channels, timepoints)
    Output: X_align is EEG aligned data for one subject (n_trials, channels, timepoints)
    '''
    n_trials, channels, _ = np.shape(X)

    #Compute the mean spatial covariance matrix of the EEG signals of each subject
    R_mean = np.zeros((n_trials,channels,channels))
    for j in range(n_trials):
        R_mean[j] = np.cov(X[j])
    R_mean = np.mean(R_mean,axis=0)

    #Re-center the spatial covariance matrix of each subject at the identity matrix
    X_align = np.zeros(np.shape(X))
    for j in range(n_trials):
        X_align[j] = sqrtm(inv(R_mean))@X[j]

    print('Euclidean alignment subject '+str(subject)+' done')

    return X_align

if __name__ == "__main__":
    subjectNumber = len([name for name in os.listdir("./data/raw") if os.path.isfile(os.path.join("./data/raw", name))])
    Path("./data/aligned").mkdir(parents=True, exist_ok=True) 
    for subject in range(1, subjectNumber+1):
        data = dict(np.load('./data/raw/patient'+str(subject)+'.npz')) 
        data['data'] = euclidean_alignment(data['data'])
        np.savez('./data/aligned/patient'+str(subject), **data)
