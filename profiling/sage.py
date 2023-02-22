import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from torch.utils.data import DataLoader
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from utils import load_data
from utils import evaluate as ev
from torch.profiler import profile, record_function, ProfilerActivity

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, x):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
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


def train(g, features, labels, masks, model, dataset,args, train_id):
    # define train/val samples, loss function and optimizer
    train_mask, val_mask = masks
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    if args.algo == 'mini':
        sampler = NeighborSampler([15, 10, 5],  # fanout for [layer-0, layer-1, layer-2]
                                prefetch_node_feats=['feat'],
                                prefetch_labels=['label'])
    else:
        sampler = MultiLayerFullNeighborSampler(3)
    
    train_dataloader = DataLoader(g, train_id, sampler, device=device,
                                  batch_size=1024, shuffle=False,
                                  drop_last=False, num_workers=4)

    # training loop
    for epoch in range(1):
        model.train()
        # total_loss = 0
        # with train_dataloader.enable_cpu_affinity():
        #     for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
        #         x = blocks[0].srcdata['feat']
        #         y = blocks[-1].dstdata['label']
        #         y_hat = model(blocks, x)
        #         loss = F.cross_entropy(y_hat, y)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         total_loss += loss.item()
        #         print('loss:', total_loss)
        # print("Epoch: ", epoch )
        
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if dataset in {"cora","citeseer","pubmed"}:
            acc = evaluate(g, features, labels, val_mask, model)
            print(
                "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                    epoch, loss.item(), acc
                )
            )
        else:
            multilabel_data = {"amazon","yelp"}
            multilabel = dataset in multilabel_data
            val_f1_mic, val_f1_mac = ev(model, g, labels, g.ndata["val_mask"], multilabel)
            print("Val F1-mic {:.4f}, Val F1-mac {:.4f}".format(val_f1_mic, val_f1_mac))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphSAGE")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed')",
    )
    parser.add_argument(
        "--profile",
        type=bool,
        default=False,
        help="True of False",
    )
    parser.add_argument("--algo", default='mini', choices=['mini', 'full'])

    args = parser.parse_args()
    print(f"Training with DGL built-in GraphSage module")

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
    elif args.dataset in {"flickr","reddit","amazon","yelp"}:
        multilabel_data = {"amazon","yelp"}
        multilabel = args.dataset in multilabel_data
        data = load_data(args,multilabel)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    
    # load and preprocess dataset
    if args.dataset == "cora" or args.dataset == "citeseer" or args.dataset == "pubmed":
        g = data[0]
    else:
        g = data.g

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"]
    train_id = data[2]
    # create GraphSAGE model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = SAGE(in_size, 128, out_size).to(device)

    # model training
    print("Training...")
    if args.profile == True:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("train"):
                train(g, features, labels, masks, model, args.dataset,args, train_id)
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    else:
        train(g, features, labels, masks, model, args.dataset,args, train_id)
    
    prof.export_chrome_trace("trace.json")
    # test the model
    '''
    if args.dataset == "cora" or args.dataset == "citeseer" or args.dataset == "pubmed":
        print("Testing...")
        acc = evaluate(g, features, labels, g.ndata["test_mask"], model)
        print("Test accuracy {:.4f}".format(acc))
    else:
        val_f1_mic, val_f1_mac = ev(model, g, labels, g.ndata["val_mask"], multilabel)
        print("Val F1-mic {:.4f}, Val F1-mac {:.4f}".format(val_f1_mic, val_f1_mac))
    '''