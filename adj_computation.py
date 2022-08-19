import numpy as np
import os, os.path
from pathlib import Path


def Scompute_mulinfo(data,chans):
    adj_matrix = np.zeros((chans,chans))
    matrix = data
    for i in range(len(matrix)):
        for j in range(i+1,len(matrix)):
            size = matrix[i].shape[-1]
            px = np.histogram(matrix[i], 256, (0, 255))[0] / size
            py = np.histogram(matrix[j], 256, (0, 255))[0] / size
            hx = - np.sum(px * np.log(px + 1e-8))
            hy = - np.sum(py * np.log(py + 1e-8))

            hxy = np.histogram2d(matrix[i], matrix[j], 256, [[0, 255], [0, 255]])[0]
            hxy /= (1.0 * size)
            hxy = - np.sum(hxy * np.log(hxy + 1e-8))

            r = hx + hy - hxy
            adj_matrix[i][j] = r
            adj_matrix[j][i] = r

    return adj_matrix

if __name__ == "__main__":
    Path("./data/adj").mkdir(parents=True, exist_ok=True) 
    subjectNumber = len([name for name in os.listdir("./data/csp") if os.path.isfile(os.path.join("./data/csp", name))])
    for subject in range(1, subjectNumber+1):
        data = dict(np.load('./data/csp/patient'+str(subject)+'.npz'))
        X = data['data']
        adj = np.zeros((X.shape[0],X.shape[1],X.shape[1]))
        print("Loading adj...")
        for i in range(len(X)):
            print('#', sep=' ', end='', flush=True)
            adj[i] = Scompute_mulinfo(X[i],X.shape[1])

        np.savez('./data/adj/patient'+str(subject), adj=adj)
        print('\n')
        print("adj patient ", subject, "saved")


