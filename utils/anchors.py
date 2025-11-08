# utils/anchors.py
import torch
from sklearn.cluster import KMeans

def kmeans_anchors(boxes, n_anchors=3):
    # boxes: [N,2] w,h normalized
    kmeans = KMeans(n_clusters=n_anchors)
    kmeans.fit(boxes)
    return kmeans.cluster_centers_

def assign_anchors(targets, anchors):
    # targets [N,5], anchors [num_anchors,2]
    N = targets.shape[0]
    anchor_idx = torch.randint(0,len(anchors),(N,))
    return anchor_idx
