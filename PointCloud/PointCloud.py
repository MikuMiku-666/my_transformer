# pointcloud_printer.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 触发 3D 支持

class PointCloud:
    """
    用于可视化点云 + 可选 skeleton 的简易打印器。

    参数
    ----
    point_clouds:
        - torch.Tensor 或 np.ndarray
        - 形状 [T, N, 3] 或 [N, 3]

    skeletons: (可选)
        - torch.Tensor 或 np.ndarray
        - 形状 [T, K, 3] 或 [K, 3]
        - 如果不传，则只绘制点云

    bones: (可选)
        - list[tuple[int, int]]
        - 每个元素 (i, j) 表示 skeleton 中第 i 个关节与第 j 个关节之间画一条线
    """

    def __init__(self, point_clouds,
                 skeletons=None,
                 bones=None):
        # 统一点云为 numpy [T, N, 3]
        self.point_clouds = self._to_numpy_3d(point_clouds, name="point_clouds")

        # skeleton 可选
        if skeletons is not None:
            self.skeletons = self._to_numpy_3d(skeletons, name="skeletons")
            # 帧数要匹配或 skeleton 只有 1 帧（全局共享）
            if self.skeletons.shape[0] not in (1, self.point_clouds.shape[0]):
                raise ValueError(
                    f"skeletons T={self.skeletons.shape[0]} "
                    f"与 point_clouds T={self.point_clouds.shape[0]} 不匹配"
                )
        else:
            self.skeletons = None

        # bones: list[(i, j)]
        self.bones = bones

    def _to_numpy_3d(self, arr, name="array"):
        """把 [T,N,3] / [N,3] / torch.Tensor 统一成 numpy [T,N,3]。"""
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        else:
            arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim == 2:
            # [N,3] -> [1,N,3]
            arr = arr[None, ...]
        if not (arr.ndim == 3 and arr.shape[2] == 3):
            raise ValueError(f"{name} 期望形状 [T,N,3] 或 [N,3]，实际为 {arr.shape}")
        return arr

    def _set_equal_axes(self, ax, points):
        """让三个轴比例一致，看起来不变形。"""
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        max_range = np.array([x.max() - x.min(),
                              y.max() - y.min(),
                              z.max() - z.min()]).max() / 2.0

        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    def show_frame(self,
                   frame_idx=0,
                   title=None,
                   figsize=(6, 6),
                   s=8,
                   draw_skeleton=False,
                   skeleton=None,
                   axis=(0, 1, 2)):
        """
        可视化某一帧点云（可选画 skeleton）。

        参数
        ----
        frame_idx : int
        title : str | None
        figsize : tuple
        s : float
        draw_skeleton : bool
        skeleton :
        axis : tuple[int, int, int]
            指定把 point_cloud 的哪三个维度映射到 (X, Y, Z)。
            例如:
              - (0,1,2): 原始 (x,y,z)
              - (0,2,1): 把原来的 z 画在 Y 轴上
              - (2,0,1): 把原来的 z 当作 X 轴
        """
        T = self.point_clouds.shape[0]
        if not (0 <= frame_idx < T):
            raise IndexError(f"frame_idx {frame_idx} out of range [0, {T-1}]")

        ix, iy, iz = axis
        pts = self.point_clouds[frame_idx]  # (N,3)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # 画点云（根据 axis 选取坐标）
        ax.scatter(pts[:, ix], pts[:, iy], pts[:, iz], s=s)

        # 是否画 skeleton
        if draw_skeleton:
            skel_pts = None

            # 优先使用参数传入的 skeleton
            if skeleton is not None:
                if isinstance(skeleton, torch.Tensor):
                    skel_pts = skeleton.detach().cpu().numpy().astype(np.float32)
                else:
                    skel_pts = np.asarray(skeleton, dtype=np.float32)
            elif self.skeletons is not None:
                # skeletons 只有 1 帧：视为对所有帧共用
                if self.skeletons.shape[0] == 1:
                    skel_pts = self.skeletons[0]
                else:
                    skel_pts = self.skeletons[frame_idx]

            if skel_pts is not None:
                # 画关节点
                ax.scatter(skel_pts[:, ix], skel_pts[:, iy], skel_pts[:, iz],
                           s=s * 2, marker='o')

                # 如果提供了 bones，就画骨架线
                if self.bones is not None:
                    for i, j in self.bones:
                        p1 = skel_pts[i]
                        p2 = skel_pts[j]
                        ax.plot([p1[ix], p2[ix]],
                                [p1[iy], p2[iy]],
                                [p1[iz], p2[iz]])

        ax.set_xlabel(f"axis[{ix}]")
        ax.set_ylabel(f"axis[{iy}]")
        ax.set_zlabel(f"axis[{iz}]")

        if title is None:
            title = f"Frame {frame_idx}"
        ax.set_title(title)

        # 这里把点云重排后丢给 _set_equal_axes，这样 xyz 比例是按当前 axis 的
        self._set_equal_axes(ax, pts[:, [ix, iy, iz]])
        plt.tight_layout()
        plt.show()



    def show_frames(self,
                    start_frames=0,
                    end_frames=None,
                    title=None,
                    figsize=(20, 20),
                    s=8,
                    draw_skeleton=False,
                    skeleton=None,
                    fps=5,
                    axis=(0, 1, 2)):
        """
        连续播放多帧点云（可选绘制骨架）。

        额外参数
        ----
        axis : tuple[int, int, int]
            同 show_frame，用来指定 (X,Y,Z) 对应的维度索引。
        """

        def _worker():
            import numpy as np
            plt.ion()  # 打开交互模式

            ix, iy, iz = axis

            T = self.point_clouds.shape[0]
            start = max(0, int(start_frames))
            if end_frames is None:
                end = T - 1
            else:
                end = min(T - 1, int(end_frames))

            if start > end:
                print(f"[PointCloud] Invalid frame range: {start} > {end}")
                return

            # 第一次创建 figure / axes / scatter
            pts0 = self.point_clouds[start]  # (N,3)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

            # ---- 按键控制：按 q 停止 ----
            stop_flag = {"stop": False}

            def on_key(event):
                if event.key == "q":
                    stop_flag["stop"] = True
                    print("[PointCloud] 'q' pressed, stop playing.")

            fig.canvas.mpl_connect("key_press_event", on_key)

            # 点云散点（注意用 ix,iy,iz）
            scat_pc = ax.scatter(pts0[:, ix], pts0[:, iy], pts0[:, iz], s=s)

            # 决定 skeleton 使用什么来源
            skel_src = None
            skel_is_sequence = False
            if draw_skeleton:
                if skeleton is not None:
                    # 调用时传入了单帧 skeleton
                    if isinstance(skeleton, torch.Tensor):
                        skel_src = skeleton.detach().cpu().numpy().astype(np.float32)
                    else:
                        skel_src = np.asarray(skeleton, dtype=np.float32)
                    skel_is_sequence = False
                elif getattr(self, "skeletons", None) is not None:
                    # 使用内部 skeleton 序列
                    skel_src = self.skeletons
                    skel_is_sequence = True

            # skeleton 的散点和线条句柄
            scat_skel = None
            line_objs = []

            if draw_skeleton and skel_src is not None:
                # 根据是单帧还是序列，取对应帧
                if skel_is_sequence:
                    if skel_src.shape[0] == 1:
                        skel_pts = skel_src[0]
                    else:
                        skel_pts = skel_src[start]
                else:
                    skel_pts = skel_src  # 单帧

                scat_skel = ax.scatter(
                    skel_pts[:, ix], skel_pts[:, iy], skel_pts[:, iz],
                    s=s * 2, marker='o'
                )

                if self.bones is not None:
                    for i, j in self.bones:
                        p1 = skel_pts[i]
                        p2 = skel_pts[j]
                        line, = ax.plot(
                            [p1[ix], p2[ix]],
                            [p1[iy], p2[iy]],
                            [p1[iz], p2[iz]],
                        )
                        line_objs.append(line)

            # 轴标签 & 初始范围
            ax.set_xlabel(f"axis[{ix}]")
            ax.set_ylabel(f"axis[{iy}]")
            ax.set_zlabel(f"axis[{iz}]")
            self._set_equal_axes(ax, pts0[:, [ix, iy, iz]])

            frame_interval = 1.0 / max(fps, 1e-6)

            for idx in range(start, end + 1):
                if stop_flag["stop"]:
                    break

                pts = self.point_clouds[idx]

                # 更新点云散点
                scat_pc._offsets3d = (pts[:, ix], pts[:, iy], pts[:, iz])

                # 更新 skeleton
                if draw_skeleton and skel_src is not None:
                    if skel_is_sequence:
                        if skel_src.shape[0] == 1:
                            skel_pts = skel_src[0]
                        else:
                            skel_pts = skel_src[idx]
                    else:
                        skel_pts = skel_src

                    if scat_skel is not None:
                        scat_skel._offsets3d = (
                            skel_pts[:, ix],
                            skel_pts[:, iy],
                            skel_pts[:, iz],
                        )
                    # 更新骨架线
                    if self.bones is not None and line_objs:
                        for (i, j), line in zip(self.bones, line_objs):
                            p1 = skel_pts[i]
                            p2 = skel_pts[j]
                            line.set_data_3d(
                                [p1[ix], p2[ix]],
                                [p1[iy], p2[iy]],
                                [p1[iz], p2[iz]],
                            )

                # 更新标题
                if title is None:
                    ax.set_title(f"Frame {idx}")
                else:
                    ax.set_title(f"{title} - Frame {idx}")

                plt.draw()
                plt.pause(frame_interval)

            plt.ioff()
            plt.close(fig)

        # 不再后台线程，直接在当前线程播放
        _worker()

