import os
import time

from thop import profile
from thop import clever_format
import torch

from config import GlobalConfig
from model import TCP

config = GlobalConfig()
model = TCP(config)
print(model)
image = torch.randn(1, 3, 256, 900, device='cuda')
measure = torch.randn(1, 9, device='cuda')
target = torch.randn(1, 2, device='cuda')
model = model.cuda()

flops, params = profile(model, inputs=(image, measure, target))
flops, params = clever_format([flops, params], "%.3f")
print(flops, params)

# latency test
model.eval()
repeat = 1000
time_start = time.time()
for _ in range(repeat):
    model(image, measure, target)
    torch.cuda.synchronize()
time_end = time.time()
print('latency: ', (time_end-time_start)/repeat * 1000, 'ms')

# with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False, with_stack=True) as prof:
#     model(image, measure, target)
# resultList = prof.table().split("\n")
# prof.export_chrome_trace('track.json')
# prof.export_stacks('cpu_stack.json', metric="self_cpu_time_total")
# prof.export_stacks('gpu_stack.json', metric="self_cuda_time_total")