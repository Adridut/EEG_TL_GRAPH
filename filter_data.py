from scipy.signal import butter, lfilter
from scipy.signal import freqz
import os, os.path
from pathlib import Path
import numpy as np


def butter_bandpass_filter(data, lowcut, highcut, step, fs, order=5):
    startbands = np.arange(lowcut, highcut, step = step)
    nyq = 0.5 * fs

    filtered_data = np.zeros([data.shape[0], data.shape[1], data.shape[2], len(startbands)])
    i = 0
    for startband in startbands:
        band = "{:02d}_{:02d}".format(startband, startband+step)
        print('Filtering through {} Hz band'.format(band))

        low = startband/nyq
        high = (startband+step)/nyq
        b,a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data, axis=-1)
        filtered_data[:,:,:,i] = y
        i += 1

    return filtered_data

if __name__ == "__main__":
    subjectNumber = len([name for name in os.listdir("./data/aligned") if os.path.isfile(os.path.join("./data/aligned", name))])
    Path("./data/filtered").mkdir(parents=True, exist_ok=True) 
    for subject in range(1, subjectNumber+1):
        data = dict(np.load('./data/aligned/patient'+str(subject)+'.npz')) 
        print('Filtering subject {}'.format(subject))
        data['data'] = butter_bandpass_filter(data['data'], lowcut=4, highcut=40, step=4, fs=250, order=5)
        print('###############################################################################')
        np.savez('./data/filtered/patient'+str(subject), **data)
