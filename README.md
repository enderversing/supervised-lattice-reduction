# Supervised Neural Lattice Reduction

I am inspired by [self-supervised neural lattice reduction](https://arxiv.org/abs/2311.08170).  I wonder if supervised neural lattice reduction would be better than LLL, or even as good as BKZ.

## Dataset Generation

I used Sage to generate 500 random 30x30 lattice bases and their BKZ-reduced counterparts in `gen_random.ipynb`. 

## Dataset 

`unreduced_30m_random.npy` contains 500 random 30x30 lattice bases. `reduced_30m_random.npy` contains each of the the random bases, BKZ-reduced.

## Dataset Loading 

I encoded the 500 random 30x30 lattice bases and their BKZ-reduced counterparts as graphs in `load.py`. 

## Message-Passing Graph Neural Network

This is where I got stuck. I tried to train a GNN on `unreduced_30m_random.npy` and `reduced_30m_random.npy` in `train.py` unsuccessfully. 

This is the error message:

```
Traceback (most recent call last):
  File "../train.py", line 36, in <module>
    output = model(batch)
  File "../.pyenv/versions/3.10.16/envs/project/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "../.pyenv/versions/3.10.16/envs/project/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "../train.py", line 21, in forward
    x = self.conv1(x, edge_index)
  File "../.pyenv/versions/3.10.16/envs/project/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "../.pyenv/versions/3.10.16/envs/project/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "../.pyenv/versions/3.10.16/envs/project/lib/python3.10/site-packages/torch_geometric/nn/conv/gcn_conv.py", line 242, in forward
    edge_index, edge_weight, x.size(self.node_dim),
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got -2)
```

