import numpy as np
import torch
import os, os.path
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from einops import rearrange, reduce, repeat



if __name__ == "__main__":
    subjectNumber = len([name for name in os.listdir("./data/csp") if os.path.isfile(os.path.join("./data/csp", name))])
    targetSubject = 1
    isfirst = True
    for subject in range(1, subjectNumber+1):
        data = dict(np.load('./data/csp/patient'+str(subject)+'.npz'))
        adj = dict(np.load('./data/adj/patient'+str(subject)+'.npz'))
        if subject == targetSubject:
            # target subject is used for validation
            val_x = data['data']
            val_y = data['label']
        else:
            if isfirst:
                train_x = data['data'] 
                train_y = data['label']
                edges = adj['adj']
                isfirst = False
            else:
                train_x = np.concatenate([train_x,data['data'] ],axis=0)
                train_y = np.concatenate([train_y,data['label']],axis=0)
                edges = np.concatenate([edges,adj['adj']],axis=0)

    # Shuffle validation and training data 
    # index = [j for j in range(len(val_y))]
    # np.random.shuffle(index)
    # val_x = val_x[index]
    # val_y = val_y[index]

    # index = [j for j in range(len(train_x))]
    # np.random.shuffle(index)
    # train_x = train_x[index]
    # train_y = train_y[index]

    # model = GATv2Conv(train_x.shape, 4, heads=2)
    # model.train()
    # pred = model(torch.from_numpy(train_x).to(torch.float32), edges)
    # print(pred)

    #x = torch.randn((8, 12, 207, 16))
    #edge_index = torch.randint(high=206, size=(2, 4608))
    #x = einops.rearrange(x, 'b l n f -> (b l) n f') 
    
    # TODO create edge list via adj
    edge_index = [(0,1),(2,1)]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    train_x = torch.from_numpy(train_x).to(torch.float32)
    layer = GATv2Conv(in_channels=1001, out_channels=22)
    result = torch.stack([layer(graph, edge_index) for graph in train_x], dim=0)
    print(result)