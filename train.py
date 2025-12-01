import os
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

from Loader.mmPoseLoader import mmPoseLoader as mpLoader
from model.transformer import Point_transformer
from option.opt import *

# === 新增：MPJPE loss 和权重 ===
def mpjpe_loss(pred, target, reduction: str = "mean"):
    """
    pred:   [B, J, 3]
    target: [B, J, 3]
    return: standard MPJPE
    """
    diff = pred - target               # [B, J, 3]
    per_joint = torch.norm(diff, dim=-1)  # [B, J]
    if reduction == "mean":
        return per_joint.mean()
    elif reduction == "none":
        return per_joint
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def build_dataloader(csv_paths, max_points, batch_size, num_workers, shuffle):
    """
    根据一组 csv 路径构建 DataLoader。
    """
    datasets = []
    for p in csv_paths:
        if not os.path.exists(p):
            print(f"[WARN] CSV not found: {p}")
            continue
        ds = mpLoader(p, max_points=max_points, crop_to_human=True)
        datasets.append(ds)

    if len(datasets) == 0:
        raise RuntimeError("No valid CSV dataset paths!")

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


def evaluate(model, pos_encoder, dataloader, device, coord_criterion, mpjpe_weight: float):
    """
    评测器：在给定 dataloader 上跑一遍，返回平均组合 loss 和 MPJPE。

    组合 loss = L1Loss + mpjpe_weight * MPJPE
    MPJPE: Mean Per Joint Position Error
    假设 kps 和 pred_kps 的单位一致。
    """
    model.eval()
    pos_encoder.eval()

    total_loss = 0.0
    total_mpjpe = 0.0
    total_samples = 0

    with torch.no_grad():
        for pc, kps in dataloader:
            pc = pc.to(device).float()   # [B, N, 3]
            kps = kps.to(device).float() # [B, 19, 3]

            B = pc.size(0)
            c_features = pos_encoder(pc)          # [B, N, in_dim]
            pred_kps = model(pc, c_features)      # [B, 19, 3]

            # 坐标分量 L1
            coord_loss = coord_criterion(pred_kps, kps)
            # MPJPE
            mpjpe = mpjpe_loss(pred_kps, kps)
            # 组合 loss
            loss = coord_loss + mpjpe_weight * mpjpe

            total_loss += loss.item() * B
            total_mpjpe += mpjpe.item() * B
            total_samples += B

    avg_loss = total_loss / total_samples
    avg_mpjpe = total_mpjpe / total_samples

    return avg_loss, avg_mpjpe


def main():
    # 1. 配置区：你可以在这里把多个 csv 填进来
    train_csvs = [
        r"./dataset/mmPrivPose3D/pose_estimation/P0.csv",
        r"./dataset/mmPrivPose3D/pose_estimation/P1.csv",
        r"./dataset/mmPrivPose3D/pose_estimation/P2.csv",
        r"./dataset/mmPrivPose3D/pose_estimation/P3.csv",
        r"./dataset/mmPrivPose3D/pose_estimation/P4.csv",
        r"./dataset/mmPrivPose3D/pose_estimation/P5.csv",
        r"./dataset/mmPrivPose3D/pose_estimation/P6.csv",
        r"./dataset/mmPrivPose3D/pose_estimation/P7.csv",
    ]

    val_csvs = [
        r"./dataset/mmPrivPose3D/pose_estimation/P8.csv",
        r"./dataset/mmPrivPose3D/pose_estimation/P9.csv",
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. 构建 train / val DataLoader
    train_loader = build_dataloader(
        train_csvs,
        max_points=max_points,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    val_loader = build_dataloader(
        val_csvs,
        max_points=max_points,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    # 3. 模型 & pos encoder
    model = Point_transformer(in_dim=in_dim, out_points=out_points).to(device)
    pos_encoder = nn.Sequential(
        nn.Linear(3, in_dim),
        nn.ReLU(inplace=True),
        nn.Linear(in_dim, in_dim)
    ).to(device)

    # 4. 损失函数 & 优化器
    coord_criterion = nn.L1Loss()   # 原来的 L1 损失，专门管坐标分量
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(pos_encoder.parameters()),
        lr=lr
    )

    best_val_loss = float("inf")
    best_ckpt_path = "point_transformer_best.pth"

    # 5. 训练 + 实时 eval
    for epoch in range(num_epochs):
        model.train()
        pos_encoder.train()

        running_loss = 0.0
        running_coord_loss = 0.0
        running_MPJPE_loss = 0.0
        num_train_samples = 0

        for pc, kps in train_loader:
            pc = pc.to(device).float()   # [B, N, 3]
            kps = kps.to(device).float() # [B, 19, 3]

            B = pc.size(0)

            # 前向
            c_features = pos_encoder(pc)              # [B, N, in_dim]
            pred_kps = model(pc, c_features)          # [B, 19, 3]

            # === 新的组合 loss ===
            coord_loss = coord_criterion(pred_kps, kps)   # L1
            mpjpe = mpjpe_loss(pred_kps, kps)             # MPJPE
            loss = coord_loss + MPJPE_WEIGHT * mpjpe

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_coord_loss += coord_loss.item() * B
            running_MPJPE_loss += mpjpe.item() * B
            running_loss += loss.item() * B
            num_train_samples += B

        train_loss = running_loss / num_train_samples
        coord_loss = running_coord_loss / num_train_samples
        MPJPE_loss = running_MPJPE_loss / num_train_samples


        # === 实时 eval：每个 epoch 结束后在 val 集上评估 ===
        val_loss, val_mpjpe = evaluate(
            model=model,
            pos_encoder=pos_encoder,
            dataloader=val_loader,
            device=device,
            coord_criterion=coord_criterion,
            mpjpe_weight=MPJPE_WEIGHT
        )

        # 如果你的坐标单位是米，这里也可以 *1000 打印成 mm
        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"train_loss: {train_loss:.6f}  "
            f"train_normal_loss: {coord_loss:.6f}  "
            f"train_MPJPE_loss: {MPJPE_loss:.6f}   "
            f"val_loss: {val_loss:.6f}  "
            f"val_MPJPE: {val_mpjpe:.6f}"

        )

        # 如果 val_loss 更好，就保存最优 checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "pos_encoder_state_dict": pos_encoder.state_dict(),
                "best_val_loss": best_val_loss,
            }, best_ckpt_path)
            print(f"  -> New best model saved to {best_ckpt_path} (val_loss={best_val_loss:.6f})")

    print("Training finished.")
    print(f"Best val_loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
