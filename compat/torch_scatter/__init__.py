"""The torch_scatter functions G4SATBench uses, implemented with torch_geometric.utils.scatter."""
from torch_geometric.utils import scatter as _scatter

__all__ = ["scatter_sum", "scatter_mean", "scatter_add", "scatter_max", "scatter_min"]


def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
    assert out is None, "out= is not supported"
    return _scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")


scatter_add = scatter_sum


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    assert out is None, "out= is not supported"
    return _scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")


def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    assert out is None, "out= is not supported"
    return _scatter(src, index, dim=dim, dim_size=dim_size, reduce="max")


def scatter_min(src, index, dim=-1, out=None, dim_size=None):
    assert out is None, "out= is not supported"
    return _scatter(src, index, dim=dim, dim_size=dim_size, reduce="min")
