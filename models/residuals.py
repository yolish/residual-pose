import torch


def add_residuals(cls_log_distr, centroids, redisuals, gt_indices=None):
    batch_size = cls_log_distr.shape[0]
    _, max_indices = cls_log_distr.max(dim=1)
    # Take the global latents by zeroing other scene's predictions and summing up
    w = centroids * 0
    if gt_indices is not None:
        max_indices = gt_indices
    w[range(batch_size), max_indices] = 1
    selected_centroids = torch.sum(w * centroids, dim=1)
    return selected_centroids + redisuals

def select_centroids(cls_log_distr, centroids):
    batch_size = cls_log_distr.shape[0]
    _, max_indices = cls_log_distr.max(dim=1)
    # Take the global latents by zeroing other scene's predictions and summing up
    w = centroids * 0
    w[range(batch_size), max_indices] = 1
    selected_centroids = torch.sum(w * centroids, dim=1)
    return selected_centroids
