import torch
from Loader.mmPoseLoader import mmPoseLoader as mpLoader
from PointCloud.PointCloud import PointCloud
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

if __name__ == '__main__':

    dataset = mpLoader(r"./dataset/mmPrivPose3D/pose_estimation/P3.csv", max_points=32, crop_to_human=True)
    device = 0
    loader = DataLoader(dataset, batch_size=1000, shuffle=True, num_workers=4)

    for pc, kps in loader:
        # pc:  (B, 512, 3)
        # kps: (B, K, 3)
        pc = pc.to(device)
        kps = kps.to(device)
        point_cloud = PointCloud(pc, kps)
        point_cloud.show_frames(draw_skeleton=True, fps=1, axis=(1, 2, 0))



