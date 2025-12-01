import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class mmPoseLoader(Dataset):
    """
    用法示例:
        dataset = mmPoseLoader("P1.csv", max_points=512,
                               crop_to_human=True, margin=0.3)
        pc, kps = dataset[0]   # pc: (512,3), kps:(K,3)

        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for pc, kps in loader:    # pc: (B,512,3)
            ...

    返回:
      __getitem__(i) -> (point_cloud_i, keypoints_i)
        point_cloud_i : torch.FloatTensor, shape = (max_points, 3)
        keypoints_i   : torch.FloatTensor, shape = (K, 3)
    """

    _float_pattern = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')

    def __init__(self, csv_path: str, max_points: int,
             crop_to_human: bool = False,
             margin: float = 0.3):
        super().__init__()
        self.csv_path = csv_path
        self.max_points = max_points
        self.crop_to_human = crop_to_human
        self.margin = margin

        df = pd.read_csv(csv_path)

        pc_list = []   # 每帧点云 (max_points,3)
        kps_list = []  # 每帧关键点 (K,3)

        for row in df.itertuples(index=False):
            # 这里列名要和 CSV 保持一致
            pc_str = getattr(row, "SortedPointCloud")
            kps_str = getattr(row, "Keypoints3D")

            # 先解析关键点 (K,3)
            kps = self._parse_vec3_string(kps_str)     # (K,3)

            # 再解析点云 (N,3)
            pc_raw = self._parse_vec3_string(pc_str)  # (N,3)

            # ---- 人体区域提取：用关键点包围盒裁剪点云 ----
            if self.crop_to_human and pc_raw.shape[0] > 0 and kps.shape[0] > 0:
                pc_raw = self._crop_to_human_region(pc_raw, kps, self.margin)

            # 再做 FPS + 补 0，得到固定长度
            pc_fixed = self._pad_or_fps(pc_raw, max_points)  # (max_points,3)

            pc_list.append(pc_fixed)
            kps_list.append(kps)

        # ---------- 关键点数量一致性检查 + 过滤异常帧 ----------
        # 先统计每种 keypoint 个数出现的频率，取“多数派”作为标准（通常是 19）
        kp_counts = [kps.shape[0] for kps in kps_list]
        # 简单众数：用字典数一下
        count_freq = {}
        for c in kp_counts:
            count_freq[c] = count_freq.get(c, 0) + 1
        # 找出现次数最多的 keypoint 个数
        target_k = max(count_freq.items(), key=lambda x: x[1])[0]

        valid_pc_list = []
        valid_kps_list = []

        for i, (pc_frame, kps_frame) in enumerate(zip(pc_list, kps_list)):
            if kps_frame.shape[0] != target_k:
                # 打个 log，方便你知道哪些帧被丢掉了
                print(f"[mmPoseLoader] skip frame {i}: {kps_frame.shape[0]} keypoints (expected {target_k})")
                continue
            valid_pc_list.append(pc_frame)
            valid_kps_list.append(kps_frame)

        if len(valid_kps_list) == 0:
            raise ValueError(
                f"No valid frames in {csv_path}: all frames have wrong keypoint count. "
                f"Keypoint count distribution = {count_freq}"
            )

        # 最终采用的关键点个数
        k = target_k

        # 全部转成 torch.Tensor
        self.point_clouds = torch.from_numpy(np.stack(valid_pc_list, axis=0)).float()  # (F_valid,max_points,3)
        self.keypoints    = torch.from_numpy(np.stack(valid_kps_list, axis=0)).float() # (F_valid,K,3)

        # 也可以存一下 K，方便后面 debug
        self.num_keypoints = k


    # ----------- 工具函数 ----------- #

    @staticmethod
    def _parse_vec3_string(text: str) -> np.ndarray:
        """
        把一帧里的字符串解析成 (N, 3) 的 numpy 数组。
        text 格式可以是 "[x y z ...]" / "[x y z],[...]" / 任意包含数字的字符串。
        """
        if not isinstance(text, str):
            text = str(text)

        nums = [float(x) for x in mmPoseLoader._float_pattern.findall(text)]
        if len(nums) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        if len(nums) % 3 != 0:
            raise ValueError(
                f"Cannot reshape to (N,3): got {len(nums)} floats in: {text[:80]}..."
            )
        return np.array(nums, dtype=np.float32).reshape(-1, 3)

    @staticmethod
    def _farthest_point_sampling(points: np.ndarray, m: int) -> np.ndarray:
        """
        Farthest Point Sampling
        points: (N, 3)
        返回采样点索引: (m,)
        """
        n = points.shape[0]
        if m >= n:
            return np.arange(n, dtype=np.int64)

        centroids = np.zeros(m, dtype=np.int64)
        distances = np.ones(n, dtype=np.float32) * 1e10

        farthest = 0  # 起点随便选一个
        for i in range(m):
            centroids[i] = farthest
            centroid = points[farthest:farthest+1, :]      # (1,3)
            dist = np.sum((points - centroid) ** 2, axis=1)
            distances = np.minimum(distances, dist)
            farthest = int(np.argmax(distances))

        return centroids

    @staticmethod
    def _pad_or_fps(points: np.ndarray, max_points: int) -> np.ndarray:
        """
        对一帧点云:
          - N > max_points : FPS 下采样到 max_points
          - N < max_points : 尾部补 0
          - N = max_points : 不变
        返回 (max_points,3)
        """
        n = points.shape[0]

        if n == 0:
            return np.zeros((max_points, 3), dtype=np.float32)

        if n > max_points:
            idx = mmPoseLoader._farthest_point_sampling(points, max_points)
            return points[idx]

        if n < max_points:
            pad = np.zeros((max_points - n, 3), dtype=np.float32)
            return np.concatenate([points, pad], axis=0)

        return points

    @staticmethod
    def _crop_to_human_region(points: np.ndarray,
                              keypoints: np.ndarray,
                              margin: float) -> np.ndarray:
        """
        用关键点的 3D 包围盒裁剪点云，只保留人体附近的点。

        points:    (N,3)
        keypoints: (K,3)
        margin:    在包围盒基础上向外扩展的距离（单位同坐标，一般是米）

        返回: (N',3)，如果由于太严格导致 N' == 0，则退回原始 points。
        """
        if points.shape[0] == 0 or keypoints.shape[0] == 0:
            return points

        min_xyz = keypoints.min(axis=0) - margin
        max_xyz = keypoints.max(axis=0) + margin

        # (N,3) 的 bool，按 xyz 同时在范围内
        mask = (points >= min_xyz) & (points <= max_xyz)
        mask = np.all(mask, axis=1)

        cropped = points[mask]
        if cropped.shape[0] == 0:
            # 如果裁剪完一个点都没有，说明 margin 太小 / 点云太稀疏，直接退回全局点云
            return points

        return cropped

    # ----------- PyTorch Dataset 接口 ----------- #

    def __len__(self):
        return self.point_clouds.shape[0]

    def __getitem__(self, idx):
        """
        返回:
          point_cloud: (max_points,3) float32
          keypoints:   (K,3)         float32
        """
        return (
            self.point_clouds[idx],
            self.keypoints[idx],
        )
