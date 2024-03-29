import argparse

import dgl
import dgl.nn as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import time
import os
import psutil

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(rank, size, g, features, labels, masks, model, bind):
    dist.init_process_group('gloo', rank=rank, world_size=size)
    
    model = DistributedDataParallel(model)
    p = psutil.Process()
    # core binding
    if bind == True:
        num_cores = 38
        cores = list(range(rank*num_cores, (rank+1)*num_cores))
        p.cpu_affinity(cores)
        # os.sched_setaffinity(0, cores)

    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(15):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed').",
    )
    parser.add_argument(
        "--binding",
        type=str,
        default='False'
    )
    args = parser.parse_args()
   

    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    
    bind = False
    if args.binding == "True":
        bind = True

    g = data[0]
    device = torch.device("cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    mp.set_start_method('spawn')
    size = 2
    num_cores = psutil.cpu_count(logical=False) // size
    master_addr = '127.0.0.1'
    master_port = '29500'

    # create GCN model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GCN(in_size, 128, out_size)
    
    start = time.time()
    processes = []
    
    for rank in range(size):
        
        p = mp.Process(target=train, args=(rank, size, g, features, labels, masks, model,bind))
        p.start()
        processes.append(p)

    # Wait for the processes to finish
    for p in processes:
        p.join()
    end = time.time()
    print(end - start, "sec")

