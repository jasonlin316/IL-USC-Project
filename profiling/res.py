import os
import torch
from torchvision.models import resnet50
import psutil
import time
# os.sched_setaffinity(os.getpid(), {0})
# print('Set PID-{} to affinity: {}'.format(os.getpid(), os.sched_getaffinity(os.getpid())))

comp_core = list(range(0,38))
psutil.Process().cpu_affinity(comp_core)
torch.set_num_threads(38)

net = resnet50()
net.eval()

start = time.time()
for i in range(10):
    with torch.no_grad():
        net(torch.randn(1, 3, 224, 224))

end = time.time()
print(end - start, "sec")