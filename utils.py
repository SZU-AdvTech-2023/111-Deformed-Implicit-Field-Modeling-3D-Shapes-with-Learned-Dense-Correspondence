# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def self_euclidean_distance(a, eps=0.0):
    assert a.ndim == 2
    sub = a[None, ...] - a[:, None, ...]
    return torch.norm(sub, dim=-1)
