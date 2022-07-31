import numpy as np
import os, os.path

def sort_by_classes(data):
    data['label'] = sorted(data['label'] , key=lambda x: x[0])
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
        
if __name__ == "__main__":
    subjectNumber = len([name for name in os.listdir("./data/filtered") if os.path.isfile(os.path.join("./data/aligned", name))])
    for subject in range(1, subjectNumber+1):
        data = dict(np.load('./data/filtered/patient'+str(subject)+'.npz')) 
        data = sort_by_classes(data)
        for c in set(data['label']):
            csp([data['data'][i] for i in range(len(data['label'])) if data['label'][i] == c],
             [data['data'][i] for i in range(len(data['label'])) if data['label'][i] != c])