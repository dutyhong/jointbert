"""
__project_ = 'PycharmProjects'
__file_name__ = 'crf_test'
__author__ = 'duty'
__time__ = '2021/1/21 4:17 PM'
__product_name = PyCharm
"""
import torch
from torchcrf import CRF
seq_length = 2
batch_size = 3
num_tags = 5

emissions = torch.randn(batch_size, seq_length, num_tags)
model = CRF(num_tags)
tags = torch.tensor([[0,1], [2,4], [3,1]], dtype=torch.long)
mask = torch.tensor([[1,1], [1,1], [1,1]], dtype=torch.uint8)
res = model(emissions, tags, mask)
print(res)

