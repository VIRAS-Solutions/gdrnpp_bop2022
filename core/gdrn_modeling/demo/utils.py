import numpy as np
import json

def calc_add_metric(pts_est, pts_gt):
    # pts_est: projected points of the estimation pose
    # pts_gt: projected points of the ground truth pose
    print("######################")
    print(pts_est.shape)
    print(pts_est[0])
    distances = np.linalg.norm(pts_est-pts_gt,axis=0)
    distances_avg = np.mean(distances)
    return distances_avg

def read_json(path):
    data = {}
    with open(path) as f:
        data = json.load(f)
    return data