from __future__ import print_function

import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# os.environ['NO_TENSOR_OPS'] = '1'
# os.environ['CUBLAS_LT_DISABLE_TENSOR_CORES'] = '1'
# os.environ['CUBLAS_POLICY_ALLOW_TENSOR_CORES'] = '0'


import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import timeit
import numpy as np
import time as timee

import sys
import torch
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)

scaler = torch.amp.GradScaler('cuda')

# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark with single GPU',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16 training')
parser.add_argument('--fp16-pushpull', action='store_true', default=False,
                    help='use fp16 compression during byteps pushpull')

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=200000, # modify
                    help='number of benchmark iterations')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number of classes')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--profiler', action='store_true', default=False,
                    help='disables profiler')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# cudnn.benchmark = True

# Set up standard model.
model = getattr(models, args.model)(num_classes=args.num_classes)

if args.cuda:
    # Move model to GPU.
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01)

# Set up fake data
datasets = []
for _ in range(100):
    data = torch.rand(args.batch_size, 3, 224, 224)
    target = torch.LongTensor(args.batch_size).random_() % 1000
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    datasets.append(data)
data_index = 0

def benchmark_step():
    global data_index

    data = datasets[data_index%len(datasets)]
    data_index += 1
    optimizer.zero_grad()
    if args.fp16:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(data)
            loss = F.cross_entropy(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def log(s, nl=True):
    print(s, end='\n' if nl else '')
    sys.stdout.flush()


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, 1))

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
enable_profiling = args.profiler

with torch.autograd.profiler.profile(enabled=enable_profiling, use_cuda=True) as prof:
    for x in range(args.num_iters):
        time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
        img_sec = args.batch_size * args.num_batches_per_iter / time
        log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
        img_secs.append(img_sec)
        timee.sleep(1) # 添加了休眠


# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (1, device, 1 * img_sec_mean, 1 * img_sec_conf))



# $ run -- python3 benchmark.py --batch-size 216
# Model: resnet50
# Batch size: 216
# Number of GPUs: 1
# Iter #0: 365.8 img/sec per GPU
