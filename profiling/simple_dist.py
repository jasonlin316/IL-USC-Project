import argparse

import dgl
import dgl.nn as dglnn
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF

import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from torch.profiler import profile, record_function, ProfilerActivity
from torchmetrics.classification import MulticlassAccuracy
import time
import os
import psutil

comp_core = []
load_core = []

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def bind_to_cores(rank, size, num_cores):
        cores = list(range(rank*num_cores, (rank+1)*num_cores))
        os.sched_setaffinity(0, cores)
        print(f"Rank {rank}: Bound to cores {cores}")

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, graph, dataloader):
    model.eval()
    ys = []
    y_hats = []
    with dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
        for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            with torch.no_grad():
                x = blocks[0].srcdata["feat"]
                ys.append(blocks[-1].dstdata["label"])
                y_hats.append(model(blocks, x))
                accuracy = MulticlassAccuracy(num_classes=47)
    return accuracy(torch.cat(y_hats), torch.cat(ys))


def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(pred, label)


def train(args, rank, size, device, g, dataset, model):
    # create sampler & dataloader
    dist.init_process_group('gloo', rank=rank, world_size=size)
    model = DistributedDataParallel(model)
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler(
        [15, 10, 5],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    if rank == 0:
        load_core = list(range(0,4))
        comp_core = list(range(4,38))
    else:
        load_core = list(range(38,42))
        comp_core = list(range(42,76))
    
    for epoch in range(1):
        start = time.time()
        model.train()
        total_loss = 0
        # model, opt= ipex.optimize(model, optimizer=opt)
        #with train_dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, g, val_dataloader)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, total_loss / (it + 1), acc.item()
            )
        )
        end = time.time()
        print(end - start, "sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="cpu",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")
    

    mp.set_start_method('spawn')
    size = 2
    num_cores = psutil.cpu_count(logical=False) // size
    master_addr = '127.0.0.1'
    master_port = '29500'

    # load and preprocess dataset
    print("Loading data")
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 128, out_size).to(device)
    

    processes = []
    for rank in range(size):
        p = mp.Process(target=train, args=(args,rank, size, device, g, dataset, model))
        p.start()
        processes.append(p)
    
    # Wait for the processes to finish
    for p in processes:
        p.join()

    # model training
    print("Training...")

    # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    # prof.export_chrome_trace("amd_one_socket.json")
    # acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
    # print("Test Accuracy {:.4f}".format(acc.item()))