"""Lloyd's k-means with the kmeans_pytorch.kmeans signature that G4SATBench uses."""
import torch

__all__ = ["kmeans"]


def kmeans(X, num_clusters, distance="euclidean", tol=1e-4, iter_limit=100,
           tqdm_flag=0, device=None, seed=None):
    if distance != "euclidean":
        raise NotImplementedError("only euclidean distance is supported")
    if device is not None:
        X = X.to(device)
    X = X.detach()
    n = X.shape[0]
    if n < num_clusters:
        X = X.repeat((num_clusters + n - 1) // n, 1)
        n = X.shape[0]

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen)[:num_clusters].to(X.device)
    centers = X[perm].clone()

    labels = torch.zeros(n, dtype=torch.long, device=X.device)
    for _ in range(iter_limit):
        labels = torch.cdist(X, centers).argmin(dim=1)
        new_centers = centers.clone()
        for k in range(num_clusters):
            mask = labels == k
            if mask.any():
                new_centers[k] = X[mask].mean(dim=0)
        shift = (new_centers - centers).pow(2).sum(dim=1).sqrt().sum()
        centers = new_centers
        if shift < tol:
            break
    return labels, centers
