from PointCloud.PointCloud import PointCloud
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.transformer import Point_transformer
import torch

B, N, C = 4, 128, 64
model = Point_transformer(in_dim=C, out_points=19)

pos = torch.randn(B, N, 3)
feat = torch.randn(B, N, C)

out = model(pos, feat)
print(out.shape)   # 期待: torch.Size([4, 19, 3])
