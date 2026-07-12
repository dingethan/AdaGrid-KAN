"""
AdaptiveKANLayer: 自适应网格细化的 KAN 层
==========================================
核心设计：初始采用稀疏均匀 Grid（如 G=3），当训练陷入平台期时，
通过 expand_grid() 均匀翻倍扩展 Grid，并借助 F.interpolate 平滑迁移控制点。

expand_grid:   ★ 主方案 — 全局均匀翻倍 + F.interpolate 重采样
insert_knot:   可选 — Boehm 局部插入，使 grid 变为非均匀（高级接口）

B-Spline 基底函数基于 Cox-de Boor 递推公式，
支持任意阶数（默认 order=3，即三次 B-Spline）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdaptiveKANLayer(nn.Module):
    """
    自适应网格 KAN 层，适用于 1D 回归任务。

    参数:
        input_dim:    输入维度
        output_dim:   输出维度
        grid_size:    初始网格区间数（默认 3，极度稀疏）
        spline_order: B-Spline 阶数（默认 3，即三次样条）
        grid_range:   输入归一化范围（默认 [-1, 1]）
    """

    def __init__(self, input_dim=1, output_dim=1, grid_size=3, spline_order=3,
                 grid_range=(-1.0, 1.0)):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        # ----------------------------------------------------------------
        # 均匀网格：从 grid_range[0] 到 grid_range[1] 均匀划分 G 个区间
        # shape: (G+1,)
        # 注意：insert_knot 后 grid 变为非均匀，长度 = G_new + 1
        # ----------------------------------------------------------------
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size + 1)
        self.register_buffer('grid', grid)

        # ----------------------------------------------------------------
        # B-Spline 控制点（样条权重）
        # 对于 G 个区间 + k 阶 B-Spline，需要 G + k 个控制点
        # 形状为 (output_dim, input_dim, G + k)
        # 注意：insert_knot 后控制点数 = G_new + k
        # ----------------------------------------------------------------
        n_basis = grid_size + spline_order
        self.control_points = nn.Parameter(
            torch.randn(output_dim, input_dim, n_basis) * 0.1
        )

        # ----------------------------------------------------------------
        # 基础权重：SiLU 激活的线性组合（参照原始 KAN 论文）
        # shape: (output_dim, input_dim)
        # ----------------------------------------------------------------
        self.base_weight = nn.Parameter(
            torch.randn(output_dim, input_dim) * 0.1
        )

    # ====================================================================
    # B-Spline 核心实现（公有方法 =========================================
    # ====================================================================

    def _build_knot_vector(self, grid):
        """
        从网格点构造 clamped B-Spline 节点向量（knot vector）。

        Clamped 节点向量的构造规则：
        - 开头重复 k 次 grid[0]，确保样条在左边界插值端点
        - 中间依次放入所有 grid 点
        - 末尾重复 k-1 次 grid[-1]

        总节点数 = k + (G+1) + (k-1) = G + 2k
        基函数个数 = (G + 2k) - k = G + k  ← 与控制点数量匹配

        参数:
            grid: (G+1,) 网格点，可为非均匀
        返回:
            knots: (G + 2k,) 节点向量
        """
        k = self.spline_order
        device = grid.device
        left_pad = torch.full((k,), grid[0].item(), device=device, dtype=grid.dtype)
        right_pad = torch.full((k - 1,), grid[-1].item(), device=device, dtype=grid.dtype)
        knots = torch.cat([left_pad, grid, right_pad], dim=0)
        return knots  # (G + 2k,)

    def _eval_bspline_basis(self, x, knots):
        """
        使用 Cox-de Boor 递推公式计算 B-Spline 基函数值（全向量化版本）。

        参数:
            x: (batch_size,) 输入值，已 clamp 到 grid 范围内
            knots: (N,) 节点向量，N = G + 2k
        返回:
            basis: (batch_size, G + k) 基函数值
        """
        k = self.spline_order
        batch_size = x.shape[0]
        N = len(knots)           # G + 2k

        # ------------------------------------------------------------------
        # 第 1 步：计算一阶基函数 N_{i,1}(x)（向量化）
        # ------------------------------------------------------------------
        x_exp = x.unsqueeze(-1)               # (batch, 1)
        left_b = knots[:-1].unsqueeze(0)       # (1, N-1)
        right_b = knots[1:].unsqueeze(0)       # (1, N-1)

        N_prev = ((x_exp >= left_b) & (x_exp < right_b)).float()   # (batch, N-1)
        # 处理 x == knots[-1] 的边界情况：最后一个区间右端闭合
        N_prev[:, -1] += (x == knots[-1]).float()

        # ------------------------------------------------------------------
        # 第 2 步：递推计算高阶基函数 N_{i,p}（向量化内层循环）
        # ------------------------------------------------------------------
        for p in range(2, k + 1):
            n_curr = N - p

            t_i   = knots[0:n_curr].unsqueeze(0)        # (1, n_curr)
            t_ip1 = knots[p-1 : p-1+n_curr].unsqueeze(0) # (1, n_curr)
            t_i1  = knots[1 : 1+n_curr].unsqueeze(0)     # (1, n_curr)
            t_ip  = knots[p : p+n_curr].unsqueeze(0)      # (1, n_curr)

            # 左项: (x - t_i) / (t_{i+p-1} - t_i) * N_{i,p-1}
            d_left = t_ip1 - t_i
            safe_d_left = torch.where(d_left > 1e-12, d_left, torch.ones_like(d_left))
            left = (x_exp - t_i) / safe_d_left * N_prev[:, :n_curr]
            left = torch.where(d_left > 1e-12, left, torch.zeros_like(left))

            # 右项: (t_{i+p} - x) / (t_{i+p} - t_{i+1}) * N_{i+1,p-1}
            d_right = t_ip - t_i1
            safe_d_right = torch.where(d_right > 1e-12, d_right, torch.ones_like(d_right))
            right = (t_ip - x_exp) / safe_d_right * N_prev[:, 1:n_curr+1]
            right = torch.where(d_right > 1e-12, right, torch.zeros_like(right))

            N_prev = left + right  # (batch, n_curr)

        return N_prev

    def forward(self, x):
        """
        前向传播。

        计算公式: y = base_weight @ silu(x) + Σ_i control_points_i * B_i(x)
        """
        batch_size = x.shape[0]
        device = x.device
        grid = self.grid

        # SiLU 基础激活
        base_out = torch.einsum('bi,oi->bo', F.silu(x), self.base_weight)

        # B-Spline 样条部分
        spline_out = torch.zeros(batch_size, self.output_dim, device=device)

        for d in range(self.input_dim):
            x_d = x[:, d]
            grid_min = grid[0]
            grid_max = grid[-1]
            x_d_clamped = torch.clamp(x_d, grid_min, grid_max)

            knots = self._build_knot_vector(grid)
            basis = self._eval_bspline_basis(x_d_clamped, knots)

            cp_d = self.control_points[:, d, :]  # (output_dim, G+k)
            spline_out += torch.einsum('bn,on->bo', basis, cp_d)

        return base_out + spline_out

    # ====================================================================
    # 网格细化方法 =======================================================
    # ====================================================================

    def expand_grid(self, new_grid_size):
        """
        全局均匀加密（主方案，与 PlateauGridExpander / 浏览器 demo 一致）。

        通过 F.interpolate 整体 1D 线性重采样，将控制点映射到更密集的均匀网格。
        plateau 时调用：G → min(G×2, max_grid)。

        参数:
            new_grid_size: int，新的网格区间数（必须 > 当前 grid_size）
        """
        if new_grid_size <= self.grid_size:
            raise ValueError(
                f"new_grid_size ({new_grid_size}) must be > "
                f"current grid_size ({self.grid_size})"
            )

        old_G = self.grid_size
        new_G = new_grid_size
        k = self.spline_order

        old_n_basis = old_G + k
        new_n_basis = new_G + k

        old_cp = self.control_points.data
        old_cp_flat = old_cp.reshape(-1, old_n_basis).unsqueeze(1)

        new_cp_flat = F.interpolate(
            old_cp_flat, size=new_n_basis, mode='linear', align_corners=True
        )
        new_cp = new_cp_flat.squeeze(1).reshape(self.output_dim, self.input_dim, new_n_basis)

        self.grid_size = new_G
        new_grid = torch.linspace(
            self.grid_range[0], self.grid_range[1], new_G + 1,
            device=self.grid.device, dtype=self.grid.dtype
        )
        self.register_buffer('grid', new_grid)
        self.control_points = nn.Parameter(new_cp)

    def insert_knot(self, x_bar):
        """
        ★ 局部自适应加密（可选方法）：Boehm's Knot Insertion ★

        在输入空间位置 x_bar ∈ [grid[0], grid[-1]] 插入单个新节点。
        - 仅修改 k 个控制点（k = spline_order），其余控制点原样保留
        - 曲线形状**严格不变**——这是 B-Spline 的数学性质
        - grid 变为非均匀，后续可继续插入更多节点
        - 返回 migration_info dict，供外部完成 Adam 动量迁移

        相比 expand_grid 的优势：
        - 每次只新增 1 个控制点（vs 翻倍增加 (G+k) 个）
        - 可精准在"最难拟合"的区间发力
        - 无 F.interpolate 近似误差，零损失继承

        算法来源: Piegl & Tiller, "The NURBS Book", 2nd ed., Sec 5.1

        参数:
            x_bar: float 或 0-d tensor，节点插入位置
                   （必须 ∈ [grid[0], grid[-1]]）
        返回:
            migration_info: dict，包含 {old_param, new_param, i, p, old_knots, x_bar}
                            供 migrate_optimizer_for_insert() 使用
        """
        k = self.spline_order        # 阶数
        p = k - 1                    # 次数（degree）
        n_cp = self.grid_size + k    # 当前控制点数

        # ★ 保存旧 nn.Parameter 引用（供外部迁移 Adam 动量）
        old_param = self.control_points

        # ---- 1. 获取完整的 clamped 节点向量 ----
        old_grid = self.grid  # (G+1,)
        old_knots = self._build_knot_vector(old_grid)  # (G + 2k,)
        device = old_knots.device
        dtype = old_knots.dtype

        x_bar = float(x_bar) if not isinstance(x_bar, float) else x_bar
        assert self.grid_range[0] - 1e-9 <= x_bar <= self.grid_range[1] + 1e-9, \
            f"x_bar={x_bar} out of range {self.grid_range}"

        # 拒绝在过窄区间内插入
        for j in range(len(self.grid) - 1):
            if self.grid[j] - 1e-9 <= x_bar <= self.grid[j + 1] + 1e-9:
                if self.grid[j + 1] - self.grid[j] < 0.02:
                    return None
                break

        # ---- 2. 确定插入位置 i：使 old_knots[i] ≤ x_bar < old_knots[i+1] ----
        # 对于 clamped 节点，内部区间从 knots[k] 到 knots[-(k)] 有意义
        # 使用 searchsorted 定位
        knots_np = old_knots.detach().cpu().numpy()
        i = int(np.searchsorted(knots_np, x_bar, side='right')) - 1
        # 确保 i 在有效范围内
        i = max(k, min(i, len(knots_np) - k - 1))

        # ---- 3. 生成新的节点向量 ----
        new_knots_list = knots_np.tolist()
        new_knots_list.insert(i + 1, x_bar)
        new_knots = torch.tensor(new_knots_list, device=device, dtype=dtype)

        # ---- 4. 生成新的 grid（去 clamped 端点后的内部唯一节点） ----
        interior = new_knots[k : len(new_knots) - k + 1]
        unique = []
        for v in interior.tolist():
            if not unique or abs(v - unique[-1]) > 1e-10:
                unique.append(v)
        new_grid = torch.tensor(unique, device=device, dtype=dtype)

        # ---- 5. Boehm 算法：计算新控制点 ----
        # 参考 NURBS Book Sec 5.1: α_j = (u - u_j) / (u_{j+p} - u_j)
        #   新控制点 Q_j:
        #     j ≤ i-p     → Q_j = P_j           （不变）
        #     i-p < j ≤ i → Q_j = (1-α_j)·P_{j-1} + α_j·P_j  （混合）
        #     j > i       → Q_j = P_{j-1}         （右移）
        old_cp = self.control_points.data  # (output_dim, input_dim, n_cp)
        n_cp_new = n_cp + 1
        new_cp = torch.zeros(self.output_dim, self.input_dim, n_cp_new,
                             device=device, dtype=old_cp.dtype)

        # (A) j ≤ i-p：直接继承
        left_end = i - p
        if left_end >= 0:
            new_cp[:, :, :left_end + 1] = old_cp[:, :, :left_end + 1]

        # (B) i-p < j ≤ i：Boehm 混合（核心计算）
        blend_start = max(0, i - p + 1)
        blend_end = min(i, n_cp)
        for j in range(blend_start, blend_end + 1):
            # α_j = (x_bar - knots[j]) / (knots[j+p] - knots[j])
            denom = old_knots[j + p] - old_knots[j]
            if abs(denom.item()) < 1e-12:
                alpha = torch.zeros(1, device=device, dtype=dtype)
            else:
                alpha = (x_bar - old_knots[j]) / denom

            # Q_j = (1-α_j) * P_{j-1} + α_j * P_j
            cp_left = old_cp[:, :, j - 1]    # P_{j-1}
            cp_right = old_cp[:, :, j]       # P_j
            new_cp[:, :, j] = (1.0 - alpha) * cp_left + alpha * cp_right

        # (C) j > i：右移
        if i + 1 <= n_cp_new - 1:
            new_cp[:, :, i + 1:] = old_cp[:, :, i:]

        # ---- 6. 更新内部状态 ----
        self.grid_size = len(new_grid) - 1
        self.register_buffer('grid', new_grid)
        self.control_points = nn.Parameter(new_cp)

        # ★ 返回迁移信息（供外部做 Adam 动量迁移）
        return {
            'old_param': old_param,
            'new_param': self.control_points,
            'i': i,
            'p': p,
            'old_knots': old_knots,
            'x_bar': x_bar,
        }

    @staticmethod
    def _boehm_transform_1d(tensor_3d, i, p, old_knots, x_bar):
        """
        将一个 3D 张量沿最后一维做 Boehm 变换（用于迁移 Adam 的 m/v 动量）。

        对张量的每个 (out, in) 通道，沿最后一维应用与 insert_knot 相同的
        Boehm 混合公式，得到 (n_cp+1) 的新张量。

        参数:
            tensor_3d: (output_dim, input_dim, n_cp)  需要变换的张量
            i, p, old_knots, x_bar: 与 insert_knot 相同的 Boehm 参数
        返回:
            (output_dim, input_dim, n_cp + 1)
        """
        _, _, n_cp = tensor_3d.shape
        n_cp_new = n_cp + 1
        device = tensor_3d.device
        dtype = tensor_3d.dtype
        result = torch.zeros(tensor_3d.shape[0], tensor_3d.shape[1], n_cp_new,
                             device=device, dtype=dtype)

        # (A) j ≤ i-p：直接继承
        left_end = i - p
        if left_end >= 0:
            result[:, :, :left_end + 1] = tensor_3d[:, :, :left_end + 1]

        # (B) i-p < j ≤ i：Boehm 混合
        blend_start = max(0, i - p + 1)
        blend_end = min(i, n_cp)
        for j in range(blend_start, blend_end + 1):
            denom = old_knots[j + p] - old_knots[j]
            if abs(denom.item()) < 1e-12:
                alpha = 0.0
            else:
                alpha = (x_bar - old_knots[j].item()) / denom.item()
            result[:, :, j] = (1.0 - alpha) * tensor_3d[:, :, j - 1] + alpha * tensor_3d[:, :, j]

        # (C) j > i：右移
        if i + 1 <= n_cp_new - 1:
            result[:, :, i + 1:] = tensor_3d[:, :, i:]

        return result

    # ====================================================================
    # 区间残差分析（供调度器使用）
    # ====================================================================

    def compute_interval_residuals(self, x, y_true):
        """
        计算每个 B-Spline 区间上的平均残差，用于确定"最难拟合的区域"。

        对每个输入样本：
        1. 计算预测误差 (y_pred - y_true)²
        2. 根据 x 的值将其分配到对应的 grid 区间
        3. 按区间汇总平均 MSE

        参数:
            x: (batch_size, input_dim)  输入张量
            y_true: (batch_size, output_dim)  目标值

        返回:
            res_per_interval: (G,) 每个区间上的平均平方误差
            grid_pts:         (G+1,) 当前区间端点（即 self.grid）
        """
        with torch.no_grad():
            y_pred = self.forward(x)
            residual = (y_pred - y_true).pow(2).mean(dim=-1)  # (batch,)

            # 将每个样本分配到 grid 区间
            x_1d = x[:, 0]  # (batch,)
            grid = self.grid  # (G+1,)
            G = len(grid) - 1

            # bucketize 返回 idx ∈ [0, G-1]，因为 x 已在 [grid[0], grid[-1]] 内
            bucket_idx = torch.bucketize(x_1d, grid[1:-1], right=False)

            # scatter 汇总
            res_sum = torch.zeros(G, device=x.device, dtype=residual.dtype)
            cnt = torch.zeros(G, device=x.device, dtype=residual.dtype)
            res_sum.scatter_add_(0, bucket_idx, residual)
            cnt.scatter_add_(0, bucket_idx, torch.ones_like(bucket_idx, dtype=residual.dtype))

            # 避免除零：无样本的区间用全局均值填充
            global_mean = res_sum.sum() / max(cnt.sum(), 1)
            avg = torch.where(cnt > 0, res_sum / cnt.clamp(min=1),
                              torch.full_like(res_sum, global_mean))

            return avg, grid

    @property
    def n_params(self) -> int:
        """返回该层的可训练参数总数（control_points + base_weight）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (f'input_dim={self.input_dim}, output_dim={self.output_dim}, '
                f'grid_size={self.grid_size}, spline_order={self.spline_order}, '
                f'n_params={self.n_params}')
