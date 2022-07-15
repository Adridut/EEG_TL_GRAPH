from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery
import numpy as np
from pathlib import Path


#Constants
DATASET = BNCI2014001() #from https://www.bbci.de/competition/iv/desc_2a.pdf
LABELS = DATASET.event_id
PARADIGM = MotorImagery(n_classes=len(LABELS)) 
SUBJECTS = DATASET.subject_list


def assign_labels(data):
    '''
    Get keys of labels and assign them to the data.
    Input: data with labels as strings
    Output: data with labels as ints
    ''' 
    for label in LABELS:
        data['label'][data['label'][:] == label] = LABELS[label]

    return data


if __name__ == "__main__":
    Path("./data/raw").mkdir(parents=True, exist_ok=True) #create data folder if it doesn't exist
    for subject in SUBJECTS:
        data = {}
        data['data'], data['label'], _ = PARADIGM.get_data(dataset=DATASET, subjects=[subject])
        data = assign_labels(data)

        np.savez('./data/raw/'+'raw_patient'+str(subject), **data) 

        print('subject '+str(subject)+' done')