import numpy as np
import os, os.path
import random
from pathlib import Path

def sort_by_classes(data):
    data['label'] = sorted(data['label'])
    return data

def cov(data):
    ''' Calculate the covariance for each trial and return their average '''
    n_trials, channels, _ = np.shape(data)

    R_mean = np.zeros((n_trials,channels,channels))
    for j in range(n_trials):
        R_mean[j] = np.cov(data[j])
    R_mean = np.mean(R_mean,axis=0)
    return R_mean

def whitening(sigma):
    ''' Calculate a whitening matrix for covariance matrix sigma. '''
    U, l, _ = np.linalg.svd(sigma)
    return U.dot( np.diag(l ** -0.5) )

def csp(class_x, class_rest):
    '''
    Calculate the CSP transformation matrix W.
    arguments:
        class_x - Array (trials x channels x samples) containing trials of one class
        class_rest - Array (trials x channels x samples) containing the trials of the remaining classes
    returns:
        Mixing matrix W
    '''
    cov_x = cov(class_x)
    cov_rest = cov(class_rest)
    P = whitening(cov_x + cov_rest)
    B, _, _ = np.linalg.svd( P.T.dot(cov_x).dot(P) )
    W = P.dot(B)
    return W

def apply_mix(W, data):
    ''' Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)'''
    ntrials, nchannels, nsamples = np.shape(data)
    trials_csp = np.zeros((ntrials, nchannels, nsamples))
    for i in range(ntrials):
        trials_csp[i,:,:] = W.T.dot(data[i,:,:])
    return trials_csp
        
if __name__ == "__main__":
    Path("./data/csp").mkdir(parents=True, exist_ok=True) 
    subjectNumber = len([name for name in os.listdir("./data/filtered") if os.path.isfile(os.path.join("./data/filtered", name))])
    for subject in range(1, subjectNumber+1):
        print("##############################################################################")
        print("CSP for patient", subject)
        data = dict(np.load('./data/filtered/patient'+str(subject)+'.npz')) 
        data = sort_by_classes(data)
        first = True
        for c in set(data['label']):
            print("class", c)
            class_x = [data['data'][i] for i in range(len(data['label'])) if data['label'][i] == c]
            class_rest = [data['data'][i] for i in range(len(data['label'])) if data['label'][i] != c]
            W = csp(class_x, class_rest)
            if first:
                features = apply_mix(W, np.array(class_x))
                first = False
            else:
                features = np.concatenate([features, apply_mix(W, np.array(class_x))])
        
        data['data'] = features
        random.Random(subject).shuffle(data['data'])
        random.Random(subject).shuffle(data['label'])
        np.savez('./data/csp/patient'+str(subject), **data)



