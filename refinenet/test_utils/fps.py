# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:12:41 2020

@author: cvhadessun
"""

import torch
import torch_cluster.fps_cpu

if torch.cuda.is_available():
    import torch_cluster.fps_cuda


def fps(x, batch=None, ratio=0.5, random_start=True):
    r""""A sampling algorithm from the `"PointNet++: Deep Hierarchical Feature
    Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper, which iteratively samples the
    most distant point with regard to the rest points.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        ratio (float, optional): Sampling ratio. (default: :obj:`0.5`)
        random_start (bool, optional): If set to :obj:`False`, use the first
            node in :math:`\mathbf{X}` as starting node. (default: obj:`True`)

    :rtype: :class:`LongTensor`

    .. testsetup::

        import torch
        from torch_cluster import fps

    .. testcode::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1])
        >>> batch = torch.tensor([0, 0, 0, 0])
        >>> index = fps(x, batch, ratio=0.5)
    """

    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x

    assert x.dim() == 2 and batch.dim() == 1
    assert x.size(0) == batch.size(0)
    assert ratio > 0 and ratio < 1

    if x.is_cuda:
        return torch_cluster.fps_cuda.fps(x, batch, ratio, random_start)
    else:
        return torch_cluster.fps_cpu.fps(x, batch, ratio, random_start)


# x = torch.Tensor([[-1, -1,0], [-1, 1,0], [1, -1,0], [1, 1,0],[2,2,0]])

# batch = torch.tensor([0, 0, 0, 0,0])

# index = fps(x, batch, ratio=0.4)


# print(index)


#
def fps_n(points,num_sample):
    #use the torch_cluster.fps to sampling the points
    #points:the coordinates of points
    #num_sample:the number of sampledd points
    #return the index of points in points

    x=torch.Tensor(points)

    num_points=x.size(0)

    batch = x.new_zeros(x.size(0), dtype=torch.long)

    ratio=num_sample/num_points

    index=fps(x,batch,ratio)

    return index

