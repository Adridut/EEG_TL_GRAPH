from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from tqdm import tqdm
import torch
import os
from torch_geometric.nn import global_add_pool, global_mean_pool
import copy
import pandas as pd
from re import S
import torch.nn.functional as F

# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv

import torch_geometric.transforms as T
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
import torch.nn as nn





class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        # TODO: Implement a function that initializes self.convs, 
        # self.bns, and self.softmax.

        super(GCN, self).__init__()

        # A list of GCNConv layers
        self.convs = None

        # A list of 1D batch normalization layers
        self.bns = None

        # The log softmax layer
        self.softmax = None

        self.num_layers = num_layers

        # post-message-passing
        heads = 1
        self.post_mp = nn.Sequential(
            nn.Linear(heads * hidden_dim, hidden_dim), nn.Dropout(dropout), 
            nn.Linear(hidden_dim, output_dim))

        ############# Your code here ############
        ## Note:
        ## 1. You should use torch.nn.ModuleList for self.convs and self.bns
        ## 2. self.convs has num_layers GCNConv layers
        ## 3. self.bns has num_layers - 1 BatchNorm1d layers
        ## 4. You should use torch.nn.LogSoftmax for self.softmax
        ## 5. The parameters you can set for GCNConv include 'in_channels' and 
        ## 'out_channels'. For more information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
        ## 6. The only parameter you need to set for BatchNorm1d is 'num_features'
        ## For more information please refer to the documentation: 
        ## https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        ## (~10 lines of code)
        # A list of GCNConv layers
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=hidden_dim)] +
            [GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)                             
                for i in range(num_layers-2)] + 
            [GCNConv(in_channels=hidden_dim, out_channels=output_dim)]    
        )

        # A list of 1D batch normalization layers
        self.bns = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=hidden_dim) 
                for i in range(num_layers-1)
        ])
        

        # The log softmax layer
        self.softmax = torch.nn.LogSoftmax()


        #########################################

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        # TODO: Implement a function that takes the feature tensor x and
        # edge_index tensor adj_t and returns the output tensor as
        # shown in the figure.

        out = None

        ############# Your code here ############
        ## Note:
        ## 1. Construct the network as shown in the figure
        ## 2. torch.nn.functional.relu and torch.nn.functional.dropout are useful
        ## For more information please refer to the documentation:
        ## https://pytorch.org/docs/stable/nn.functional.html
        ## 3. Don't forget to set F.dropout training to self.training
        ## 4. If return_embeds is True, then skip the last softmax layer
        ## (~7 lines of code)
        for conv, bn in zip(self.convs[:-1], self.bns):
            x1 = F.relu(bn(conv(x, adj_t)))
            if self.training:
                x1 = F.dropout(x1, p=self.dropout)
            x = x1
        x = self.convs[-1](x, adj_t)
        out = x if self.return_embeds else self.softmax(x)

        #########################################

        return out


### GCN to predict graph property
class GCN_Graph(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()

        # Load encoders for Atoms in molecule graphs

        # Node embedding model
        # Note that the input_dim and output_dim are set to hidden_dim
        self.gnn_node = GCN(hidden_dim, hidden_dim,
            hidden_dim, num_layers, dropout, return_embeds=True)

        self.pool = None

        ############# Your code here ############
        ## Note:
        ## 1. Initialize self.pool as a global mean pooling layer
        ## For more information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#global-pooling-layers
        self.pool = global_mean_pool

        #########################################

        # Output layer
        self.linear = torch.nn.Linear(hidden_dim, output_dim)


    def reset_parameters(self):
      self.gnn_node.reset_parameters()
      self.linear.reset_parameters()

    def forward(self, batched_data):
        # TODO: Implement a function that takes as input a 
        # mini-batch of graphs (torch_geometric.data.Batch) and 
        # returns the predicted graph property for each graph. 
        #
        # NOTE: Since we are predicting graph level properties,
        # your output will be a tensor with dimension equaling
        # the number of graphs in the mini-batch

    
        # Extract important attributes of our mini-batch
        x, edge_index, batch, edge_attr = batched_data.x, batched_data.edge_index, batched_data.batch, batched_data.edge_attr
        embed = x

        out = None

        ############# Your code here ############
        ## Note:
        ## 1. Construct node embeddings using existing GCN model
        ## 2. Use the global pooling layer to aggregate features for each individual graph
        ## For more information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#global-pooling-layers
        ## 3. Use a linear layer to predict each graph's property
        ## (~3 lines of code)
        embed = self.gnn_node(embed, edge_index)
        features = self.pool(embed, batch)
        out = self.linear(features)

        #########################################

        return out

def train(model, device, data_loader, optimizer, loss_fn):
    # TODO: Implement a function that trains your model by 
    # using the given optimizer and loss_fn.
    model.train()
    loss = 0

    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
      #batch = batch.to(device)


        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch.y == batch.y

        ############# Your code here ############
        ## Note:
        ## 1. Zero grad the optimizer
        ## 2. Feed the data into the model
        ## 3. Use `is_labeled` mask to filter output and labels
        ## 4. You may need to change the type of label to torch.float32
        ## 5. Feed the output and label to the loss_fn
        ## (~3 lines of code)

        optimizer.zero_grad()
        out = model(batch)
        #TODO out.shape increase over time which is weird and probably wrong
        # the following is only a work-around for the above issue
        loss = loss_fn(out[out.shape[0]-1], batch.y.float()[0])
        # loss = loss_fn(out, batch.y.float())

        #########################################
        
        loss.backward()
        optimizer.step()

    return loss.item()

from sklearn.metrics import accuracy_score

# The evaluation function
def eval(model, device, loader, evaluator, save_model_results=False, save_file=None):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        #batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            #TODO same work-around as in train
            pred = pred[pred.shape[0]-1]
            # pred = pred
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.stack(y_true)
    y_pred = torch.stack(y_pred)
    # y_true = torch.cat(y_true)
    # y_pred = torch.cat(y_pred, dim = 1)
    for i in range(y_pred.shape[0]):
        max = y_pred[i][0]
        maxIndex = 0
        for j in range(y_pred.shape[1]):
            if max < y_pred[i][j]:
                max = y_pred[i][j]
                maxIndex = j
            y_pred[i][j] = 0
        y_pred[i][maxIndex] = 1

    score = accuracy_score(y_true, y_pred)

    # y_true = np.expand_dims(y_true, axis=1)
    # y_pred = np.expand_dims(y_pred, axis=1)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    if save_model_results:
        print ("Saving Model Predictions")
        
        # Create a pandas dataframe with a two columns
        # y_pred | y_true
        #TODO change reshape to save pred correctly
        data = {}
        data['y_pred'] = y_pred.reshape(-1)
        data['y_true'] = y_true.reshape(-1)

        df = pd.DataFrame(data=data)
        # Save to csv
        df.to_csv('ogbg-molhiv_graph_' + save_file + '.csv', sep=',', index=False)

    return score


def getEdgeIndex(adj):
    # get edge index from adj matrix
    # adj is a numpy array
    # return edge_index
    indexI = []
    indexJ = []
    edge_attr = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] > 0.5:
                indexI.append(i)
                indexJ.append(j)
                edge_attr.append(adj[i][j])

    return [indexI, indexJ], edge_attr

class eeg_graph:
  def __init__(self, x, y, adj, counter):
    self.x = torch.from_numpy(x).float()
    #encode label into one-hot vector
    #TODO set number of label to be scalable
    task = []

    for i in range(1,5):

        if y == i:
            task.append(1)
        else:
            task.append(0)
    self.y = torch.tensor([task])
    edge_index, edge_attr = getEdgeIndex(adj)
    self.edge_index = torch.tensor(edge_index).long()
    self.edge_attr = torch.tensor(edge_attr).float()
    self.batch = torch.tensor([counter])
    # self.num_nodes = x.shape[0]
    # self.num_edges = self.edge_index.shape[1]
    # self.num_features = x.shape[1]
    # self.num_classes = 4



import numpy as np
from sklearn.model_selection import train_test_split

if 'IS_GRADESCOPE_ENV' not in os.environ:
  tqdm.pandas()

  # Load the dataset
  isfirst = True
  targetSubject = 1
  for i in range(1,10):
    data = dict(np.load('./data/csp/patient'+str(i)+'.npz'))
    adj = dict(np.load('./data/adj/patient'+str(i)+'.npz'))
    if i == targetSubject:
        # target subject is used for testing
        test_x = data['data']
        test_y = data['label']
        test_adj = adj['adj']
    else:
        if isfirst:
            X = data['data'] 
            y = data['label']
            all_adj = adj['adj']
            isfirst = False
        else:
            X = np.concatenate([X,data['data'] ],axis=0)
            y = np.concatenate([y,data['label']],axis=0)
            all_adj = np.concatenate([all_adj,adj['adj']],axis=0)

  lg = []
  valid = []
  test = []

  X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, shuffle=False)
  adj_train, adj_valid, adj_train2, adj_valid2 = train_test_split(all_adj, all_adj, test_size=0.33, shuffle=False)
  adj_train = np.concatenate([adj_train,adj_train2],axis=0)
  adj_valid = np.concatenate([adj_valid,adj_valid2],axis=0)

  test_value = 1

  for i in range(round(X_train.shape[0]/test_value)):    
    lg.append(eeg_graph(X_train[i][:][:], y_train[i], adj_train[i][:][:], i))

  for i in range(round(X_valid.shape[0]/test_value)):
    valid.append(eeg_graph(X_valid[i][:][:], y_valid[i], adj_valid[i][:][:], i))

  for i in range(round(test_x.shape[0]/test_value)):
    test.append(eeg_graph(test_x[i][:][:], test_y[i], test_adj[i][:][:], i))




  # Load the dataset 
  #dataset = PygGraphPropPredDataset(name='ogbg-molhiv')

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('Device: {}'.format(device))

  #new_loader = DataLoader(lg, batch_size=32, shuffle=True, num_workers=0)


#   split_idx = dataset.get_idx_split()


  # Check task type
#   print('Task type: {}'.format(dataset.task_type))

#   train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, num_workers=0)
#   valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, num_workers=0)
#   test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, num_workers=0)




  # Please do not change the args
  args = {
      'device': device,
      'num_layers': 5,
      'hidden_dim': 1001,
      'dropout': 0.5,
      'lr': 0.001,
      'epochs': 30,
  }
  args

  model = GCN_Graph(args['hidden_dim'],
              4, args['num_layers'],
              args['dropout']).to(device)
  #TODO change evaluator?
  evaluator = Evaluator(name='ogbg-molhiv')

  model.reset_parameters()

  optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
  loss_fn = torch.nn.CrossEntropyLoss()

  best_model = None
  best_valid_acc = 0

  for epoch in range(1, 1 + args["epochs"]):
    print('Training...')
    loss = train(model, device, lg, optimizer, loss_fn)

    print('Evaluating...')
    train_result = eval(model, device, lg, evaluator)
    val_result = eval(model, device, valid, evaluator)
    test_result = eval(model, device, test, evaluator)

    #TODO consider other metrics
    train_acc, valid_acc, test_acc = train_result, val_result, test_result
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_model = copy.deepcopy(model)
    print(f'Epoch: {epoch:02d}, '
          f'Loss: {loss:.4f}, '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')

    train_acc = eval(best_model, device, lg, evaluator)
    valid_acc = eval(best_model, device, valid, evaluator, save_model_results=True, save_file="valid")
    test_acc  = eval(best_model, device, test, evaluator, save_model_results=True, save_file="test")

    print(f'Best model: '
        f'Train: {100 * train_acc:.2f}%, '
        f'Valid: {100 * valid_acc:.2f}% '
        f'Test: {100 * test_acc:.2f}%')

