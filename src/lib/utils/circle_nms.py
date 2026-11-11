"""Circle suppression for [cx, cy, radius, score, class] detections.

The legacy Cython soft_nms expects boxes and a score in column four;
passing circles to it changes both the geometry and the wrong score column.
"""
import numpy as np


def circle_iou(circle, others):
    others = np.asarray(others)
    radius = np.maximum(others[:, 2], 0)
    r = max(float(circle[2]), 0)
    distance = np.linalg.norm(others[:, :2] - circle[:2], axis=1)
    intersection = np.zeros(len(others), dtype=np.float64)
    contained = distance <= np.abs(radius - r)
    intersection[contained] = np.pi * np.minimum(radius[contained], r) ** 2
    crossing = (~contained) & (distance < radius + r) & (radius > 0) & (r > 0)
    d, rr = distance[crossing], radius[crossing]
    a = np.arccos(np.clip((d*d + r*r - rr*rr) / (2*d*r), -1, 1))
    b = np.arccos(np.clip((d*d + rr*rr - r*r) / (2*d*rr), -1, 1))
    area = (-d+r+rr) * (d+r-rr) * (d-r+rr) * (d+r+rr)
    intersection[crossing] = r*r*a + rr*rr*b - .5*np.sqrt(np.maximum(area, 0))
    union = np.pi * (r*r + radius*radius) - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def soft_nms_circles(detections, sigma=.5, min_score=.001):
    """Gaussian soft-NMS; return scored circles, preserving class IDs."""
    remaining = np.asarray(detections).copy()
    remaining = remaining[(remaining[:, 2] > 0) & (remaining[:, 3] >= min_score)]
    selected = []
    while len(remaining):
        best = int(np.argmax(remaining[:, 3]))
        circle = remaining[best].copy()
        selected.append(circle)
        remaining = np.delete(remaining, best, axis=0)
        overlap = circle_iou(circle, remaining)
        remaining[:, 3] *= np.exp(-(overlap * overlap) / sigma)
        remaining = remaining[remaining[:, 3] >= min_score]
    return np.asarray(selected, dtype=detections.dtype).reshape(-1, detections.shape[1])
