import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from Loader.mmPoseLoader import mmPoseLoader as mpLoader
from model.transformer import Point_transformer
from train import evaluate, build_dataloader   # 如果在同文件，就直接 import；否则复制 evaluate 函数即可
from option.opt import *

def main():
    # 要评测的 csv 列表
    test_csvs = [
        r"./dataset/mmPrivPose3D/pose_estimation/P10.csv",
        r"./dataset/mmPrivPose3D/pose_estimation/P11.csv",
        r"./dataset/mmPrivPose3D/pose_estimation/P12.csv",
        r"./dataset/mmPrivPose3D/pose_estimation/P13.csv",
        r"./dataset/mmPrivPose3D/pose_estimation/P14.csv",
        r"./dataset/mmPrivPose3D/pose_estimation/P15.csv",
        # r"..."
    ]

   

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 和训练时保持一致的模型结构 & pos_encoder
    model = Point_transformer(in_dim=in_dim, out_points=out_points).to(device)
    pos_encoder = nn.Sequential(
        nn.Linear(3, in_dim),
        nn.ReLU(inplace=True),
        nn.Linear(in_dim, in_dim)
    ).to(device)

    # 加载训练好的权重
    ckpt = torch.load("point_transformer_best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    pos_encoder.load_state_dict(ckpt["pos_encoder_state_dict"])

    # 构建测试集 DataLoader
    test_loader = build_dataloader(
        test_csvs,
        max_points=max_points,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    criterion = nn.L1Loss()

    test_loss, test_mpjpe = evaluate(
        model=model,
        pos_encoder=pos_encoder,
        dataloader=test_loader,
        device=device,
        criterion=criterion
    )

    print(f"[TEST] loss: {test_loss:.6f}, MPJPE: {test_mpjpe:.6f}")


if __name__ == "__main__":
    main()
