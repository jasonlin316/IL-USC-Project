import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from torch.optim import SGD
import time
import os
import psutil
import subprocess
import torchvision.models as models

def bind_to_cores(rank, size, num_cores):
    cores = list(range(rank*num_cores, (rank+1)*num_cores))
    os.sched_setaffinity(0, cores)
    print(f"Rank {rank}: Bound to cores {cores}")
    
    # Bind memory to CPU cores using numactl
    mem_nodes = ",".join([str(core//psutil.cpu_count(logical=False)) for core in cores])
    subprocess.Popen(f"numactl --cpunodebind={rank} --membind={mem_nodes} -- python train.py --rank {rank} --world-size {size} --num-cores {num_cores}", shell=True)

def run(rank, size, num_cores):
    # Set up the distributed environment
    dist.init_process_group('gloo', rank=rank, world_size=size)
    
    # Bind the process to specific CPU cores and memory nodes
    bind_to_cores(rank, size, num_cores)
    
    # Load the data
    train_dataset = CIFAR10('./data', train=True, transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    
    # Define the model
    model = models.resnet50(pretrained=False, num_classes=10)
    
    # Wrap the model with DistributedDataParallel
    model = DistributedDataParallel(model)
    
    # Define the optimizer
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # Train the model
    for epoch in range(10):
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(images)
            loss = torch.nn.functional.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Rank {rank}: Epoch {epoch}, batch {i}, loss {loss.item()}")
        end_time = time.time()
        print(f"Rank {rank}: Epoch {epoch}, training time {end_time - start_time}")
    
    # Save the model
    if rank == 0:
        torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    # Set up the multiprocessing environment
    mp.set_start_method('spawn')
    
    # Specify the number of processes, the number of CPU cores per process, and the address of the master node
    size = 2
    num_cores = psutil.cpu_count(logical=False) // size
    master_addr = '127.0.0.1'
    master_port =
