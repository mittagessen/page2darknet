#!/usr/bin/env python3
import numpy as np
import glob

def iou(x, centroids):
    """
    Calculates IOU of a (w, h) rectangle and n centroids.
    """
    inter = np.prod(np.where(x < centroids, x, centroids), axis=1)
    return inter/(np.prod(x) + np.prod(centroids, axis=1) - inter)

def kmeans(data='out/labels/*.txt', k=5, w_feat=30., h_feat=30., eps=0.005):
    """
    Calculates clusters taking darknet bbox data from test files matching a glob expression.

    Args:
        data (str): glob pattern matching darknet bbox files
        k (int): number of centroids to compute
        w_feat (float): width dimension size of output feature map (used for
                        scaling, default architecture  pixel width/32)
        h_feat (float): height dimension size of output feature map
        eps (float): absolute tolerance for abort condition equivalence check

    Returns:
        (np.array, float) Scaled centroids and mean IOU of boxes and assigned centroids.
    """
    bboxes = []

    for f in glob.iglob(data):
        with open(f, 'r') as fp:
            for box in fp.read().splitlines():
                w, h = box.split()[3:]
                bboxes.append([float(w), float(h)])
    bboxes = np.array(bboxes)

    centroids = np.random.permutation(bboxes)[:k]
    n = len(bboxes)
    idx = 0
    passign = np.full(n, -1)
    pdists = np.zeros((n, k))

    while True:
        idx += 1
        dists = np.apply_along_axis(lambda x: 1 - iou(x, centroids), 1, bboxes)
        assign = np.argmin(dists, axis=1)

        if np.allclose(assign, passign, atol=eps):
            # reduction factor of yolo is 32
            centroids *= np.array([w_feat, h_feat])
            centroids.sort(0)
            return centroids, 1-np.mean(dists.min(axis=1))
            break

        for i in range(k):
            a = bboxes[assign == i]
            centroids[i] = np.mean(a[:,0]), np.mean(a[:,1])
        passign = assign.copy()
        pdists = dists.copy()

if __name__=="__main__":
    for x in range(1, 11):
        centroids, miou = kmeans(k=x)
        print(', '.join('{:.5f}'.format(x) for x in centroids.flatten()))
        print('mean iou: {}'.format(miou))
