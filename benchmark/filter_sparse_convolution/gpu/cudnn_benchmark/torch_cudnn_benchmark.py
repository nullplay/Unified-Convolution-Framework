import torch
import torch.nn as nn
import time
import timeit
import torch.utils.benchmark as benchmark
from torch.utils import mkldnn as mkldnn_utils
import torch.backends.mkldnn
import sys

f = open(sys.argv[1])
lines = f.read().splitlines()
print(*torch.__config__.show().split("\n"), sep="\n")

torch.backends.cudnn.benchmark=True
cuda = torch.device('cuda')
with torch.no_grad() :
  for line in lines:
      line = line.split()

      N   = int(line[1]) 
      H   = int(line[2])
      W   = int(line[3])
      R   = int(line[4])
      S   = int(line[5])
      STR = int(line[6])
      PAD = int(line[7])
      C   = int(line[8])
      M   = int(line[9])

      layer = nn.Conv2d(in_channels=C,out_channels=M,kernel_size=(R,S),stride=STR,padding=PAD,bias=False).to(cuda)
      x = torch.rand((N,C,H,W)).to(cuda)
     
      # warmup
      y=layer(x)
      y=layer(x)
      y=layer(x)

      sum=0
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      for i in range(100) :
        start.record()
        for r in range(10):
          y = layer(x)
        end.record()
        torch.cuda.synchronize()
        sum += start.elapsed_time(end)/10

      print(sum/100.0)
