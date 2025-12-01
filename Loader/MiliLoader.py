# milipoint_loader.py
import pickle
import numpy as np
import torch


class MiliPointSequenceLoader:
    """
    用于从 MiliPoint 处理后的 pkl 文件中读取点云序列。

    文件格式:
        obj = pickle.load(f)
        obj: list[ dict{'x': (N_i,3), 'y': (18,3)} ]

    功能:
        - load_sequence(path): 返回 [T, M, 3] 的点云 Tensor
          T = 帧数, M = 统一后的点数
    """

    def __init__(self, num_points: int):
        """
        :param num_points: 希望统一到的点数。
            - 如果为 None，则使用该文件中出现的最大点数进行 padding。
            - 如果为 int，则每帧都采样/补零到 num_points。
        """
        self.num_points = num_points

    def _fix_num_points(self, pc: np.ndarray, M: int) -> np.ndarray:
        """
        将一帧的点云统一为 M 个点：
        - N > M: 随机下采样
        - N < M: 补 0
        """
        N = pc.shape[0]

        if N > M:
            idxs = np.random.choice(N, M, replace=False)
            pc = pc[idxs]
        elif N < M:
            pad = np.zeros((M - N, pc.shape[1]), dtype=pc.dtype)
            pc = np.concatenate([pc, pad], axis=0)

        return pc

    def load_sequence(self, pkl_path: str) -> torch.Tensor:
        """
        给定 pkl 路径，返回形状为 [帧数, 点数, 3] 的 torch.Tensor。
        """
        # 1. 读取 pkl
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

        assert isinstance(obj, list), "Expect list of dicts in the pkl file."
        assert len(obj) > 0, "Empty sequence."

        # 2. 先把所有帧的点云拿出来
        pcs = []
        max_N = 0
        for s in obj:
            pc = np.asarray(s["x"], dtype=np.float32)  # (N_i, 3)
            pcs.append(pc)
            if pc.shape[0] > max_N:
                max_N = pc.shape[0]

        # 3. 决定统一的点数 M
        if self.num_points is None:
            M = max_N
        else:
            M = self.num_points

        # 4. 对每一帧做采样/补零，然后堆成 [T, M, 3]
        fixed_pcs = []
        for pc in pcs:
            pc = self._fix_num_points(pc, M)
            fixed_pcs.append(pc)

        seq = np.stack(fixed_pcs, axis=0)  # (T, M, 3)

        # 5. 转成 torch.Tensor
        return torch.from_numpy(seq)  # float32, [T, M, 3]
    
    def load_sequence_with_skeleton(self, pkl_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        给定 pkl 路径，返回:
            pointclouds: [T, M, 3]  float32
            skeletons:   [T, K, 3]  float32 (通常 K=18)
        """
        # 1. 读取 pkl
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

        assert isinstance(obj, list), "Expect list of dicts in the pkl file."
        assert len(obj) > 0, "Empty sequence."

        pcs = []
        skels = []
        max_N = 0

        # 2. 把所有帧的点云 & 骨架拿出来
        for s in obj:
            pc = np.asarray(s["x"], dtype=np.float32)   # (N_i, 3)
            kp = np.asarray(s["y"], dtype=np.float32)   # (K, 3)，你的例子是 (18,3)

            pcs.append(pc)
            skels.append(kp)

            if pc.shape[0] > max_N:
                max_N = pc.shape[0]

        # 3. 决定统一的点数 M
        if self.num_points is None:
            M = max_N
        else:
            M = self.num_points

        # 4. 对每一帧点云做采样/补零
        fixed_pcs = []
        for pc in pcs:
            pc_fixed = self._fix_num_points(pc, M)
            fixed_pcs.append(pc_fixed)

        # 5. 堆成 numpy 数组
        seq_pc = np.stack(fixed_pcs, axis=0)  # (T, M, 3)
        seq_kp = np.stack(skels,    axis=0)   # (T, K, 3)

        # 6. 转成 torch.Tensor
        pointclouds = torch.from_numpy(seq_pc)  # float32, [T, M, 3]
        skeletons   = torch.from_numpy(seq_kp)  # float32, [T, K, 3]

        return pointclouds, skeletons
