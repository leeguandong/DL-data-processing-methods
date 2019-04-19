import torch
import numpy as np

# numpy 和 torch 之间的转换，torch 可以实现 numpy 的功能
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)

print(
    '\nnumpy', np_data,
    '\ntorch', torch_data
)
