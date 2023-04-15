import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10
import torchvision.models as models
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from torch.optim import SGD
import time

def run(rank, size):
    # Set up the distributed environment
    dist.init_process_group('gloo', rank=rank, world_size=size)
    
    # Load the data
    train_dataset = CIFAR10('./data', train=True, transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler)
    
     # Define the model
    model = models.resnet50(pretrained=False, num_classes=10)
    # Wrap the model with DistributedDataParallel
    model = DistributedDataParallel(model)
    
    # Define the optimizer
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # Train the model
    for epoch in range(1):
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
    
    # Specify the number of processes and the address of the master node
    size = 2
    master_addr = '127.0.0.1'
    master_port = '29500'
    
    # Spawn the processes
    processes = []
    for rank in range(size):
        p = mp.Process(target=run, args=(rank, size))
        p.start()
        processes.append(p)
    
    # Wait for the processes to finish
    for p in processes:
        p.join()
