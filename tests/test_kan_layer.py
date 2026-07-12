"""
AdaptiveKANLayer 单元测试
========================
覆盖前向传播、B-Spline 基函数、网格扩展、Boehm 节点插入与区间残差分析。
"""

from __future__ import annotations

import sys
import os

import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kan_layer import AdaptiveKANLayer


def _make_layer(input_dim=1, output_dim=1, grid_size=3, spline_order=3, seed=0):
    torch.manual_seed(seed)
    return AdaptiveKANLayer(
        input_dim=input_dim,
        output_dim=output_dim,
        grid_size=grid_size,
        spline_order=spline_order,
    )


def _rand_input(batch=16, input_dim=1, seed=0):
    torch.manual_seed(seed)
    return torch.rand(batch, input_dim) * 2 - 1  # uniform in [-1, 1]


# ============================================================
# 1. 初始化与形状
# ============================================================

class TestInit:
    def test_grid_shape(self):
        layer = _make_layer(grid_size=5)
        assert layer.grid.shape == (6,)   # G+1

    def test_control_points_shape(self):
        layer = _make_layer(input_dim=2, output_dim=3, grid_size=4, spline_order=3)
        assert layer.control_points.shape == (3, 2, 7)  # (output, input, G+k)

    def test_base_weight_shape(self):
        layer = _make_layer(input_dim=2, output_dim=3)
        assert layer.base_weight.shape == (3, 2)

    def test_grid_range_default(self):
        layer = _make_layer()
        assert abs(layer.grid[0].item() - (-1.0)) < 1e-6
        assert abs(layer.grid[-1].item() - 1.0) < 1e-6

    def test_extra_repr(self):
        layer = _make_layer(input_dim=1, output_dim=1, grid_size=3)
        s = layer.extra_repr()
        assert "grid_size=3" in s
        assert "spline_order=3" in s


# ============================================================
# 2. 前向传播
# ============================================================

class TestForward:
    def test_output_shape_1d(self):
        layer = _make_layer(input_dim=1, output_dim=1)
        x = _rand_input(batch=8, input_dim=1)
        y = layer(x)
        assert y.shape == (8, 1)

    def test_output_shape_multi(self):
        layer = _make_layer(input_dim=3, output_dim=2, grid_size=5)
        x = _rand_input(batch=10, input_dim=3)
        y = layer(x)
        assert y.shape == (10, 2)

    def test_output_is_finite(self):
        layer = _make_layer()
        x = _rand_input(batch=32)
        y = layer(x)
        assert torch.isfinite(y).all()

    def test_gradient_flows(self):
        layer = _make_layer()
        x = _rand_input(batch=8)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert layer.control_points.grad is not None
        assert layer.base_weight.grad is not None

    def test_input_clamped_at_boundary(self):
        """超出 grid_range 的输入不应导致 NaN 或报错。"""
        layer = _make_layer()
        x = torch.tensor([[-2.0], [2.0]])
        y = layer(x)
        assert torch.isfinite(y).all()

    def test_deterministic_with_same_seed(self):
        layer1 = _make_layer(seed=7)
        layer2 = _make_layer(seed=7)
        x = _rand_input(seed=7)
        assert torch.allclose(layer1(x), layer2(x))


# ============================================================
# 3. B-Spline 基函数
# ============================================================

class TestBSplineBasis:
    def test_basis_nonnegative(self):
        layer = _make_layer(grid_size=4, spline_order=3)
        x = _rand_input(batch=20).squeeze(1)
        knots = layer._build_knot_vector(layer.grid)
        basis = layer._eval_bspline_basis(x, knots)
        assert (basis >= -1e-6).all(), "B-Spline basis should be non-negative"

    def test_basis_partition_of_unity(self):
        """B-Spline 基函数在每个点上应求和为 1（partition of unity）。"""
        layer = _make_layer(grid_size=5, spline_order=3)
        x = torch.linspace(-0.99, 0.99, 50)
        knots = layer._build_knot_vector(layer.grid)
        basis = layer._eval_bspline_basis(x, knots)
        sums = basis.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            f"Basis sum deviates from 1, max diff {(sums - 1).abs().max().item():.2e}"

    def test_basis_shape(self):
        layer = _make_layer(grid_size=4, spline_order=3)
        x = torch.linspace(-1, 1, 10)
        knots = layer._build_knot_vector(layer.grid)
        basis = layer._eval_bspline_basis(x, knots)
        expected_n_basis = layer.grid_size + layer.spline_order
        assert basis.shape == (10, expected_n_basis)

    def test_build_knot_vector_length(self):
        layer = _make_layer(grid_size=5, spline_order=3)
        knots = layer._build_knot_vector(layer.grid)
        expected_len = layer.grid_size + 2 * layer.spline_order
        assert len(knots) == expected_len


# ============================================================
# 4. 网格扩展 expand_grid
# ============================================================

class TestExpandGrid:
    def test_grid_size_updated(self):
        layer = _make_layer(grid_size=3)
        layer.expand_grid(6)
        assert layer.grid_size == 6

    def test_grid_length_updated(self):
        layer = _make_layer(grid_size=3)
        layer.expand_grid(6)
        assert layer.grid.shape == (7,)

    def test_control_points_shape_updated(self):
        layer = _make_layer(input_dim=1, output_dim=1, grid_size=3, spline_order=3)
        layer.expand_grid(6)
        assert layer.control_points.shape == (1, 1, 9)   # (output, input, G+k) = (1,1,6+3)

    def test_output_still_finite_after_expand(self):
        layer = _make_layer(grid_size=3)
        layer.expand_grid(6)
        x = _rand_input(batch=16)
        y = layer(x)
        assert torch.isfinite(y).all()

    def test_expand_raises_on_smaller_size(self):
        layer = _make_layer(grid_size=6)
        with pytest.raises(ValueError):
            layer.expand_grid(3)

    def test_expand_multiple_times(self):
        layer = _make_layer(grid_size=3)
        layer.expand_grid(6)
        layer.expand_grid(12)
        assert layer.grid_size == 12
        x = _rand_input(batch=8)
        assert torch.isfinite(layer(x)).all()

    def test_expand_preserves_gradient(self):
        layer = _make_layer(grid_size=3)
        layer.expand_grid(6)
        x = _rand_input(batch=8)
        y = layer(x)
        y.sum().backward()
        assert layer.control_points.grad is not None


# ============================================================
# 5. Boehm 节点插入 insert_knot
# ============================================================

class TestInsertKnot:
    def test_grid_size_increases_by_one(self):
        layer = _make_layer(grid_size=5, spline_order=3)
        old_g = layer.grid_size
        layer.insert_knot(0.0)
        assert layer.grid_size == old_g + 1

    def test_control_points_count_increases_by_one(self):
        layer = _make_layer(grid_size=5, spline_order=3)
        old_n = layer.control_points.shape[-1]
        layer.insert_knot(0.3)
        assert layer.control_points.shape[-1] == old_n + 1

    def test_curve_shape_preserved(self):
        """Boehm 插入后，曲线形状应严格不变。"""
        layer = _make_layer(grid_size=5, spline_order=3, seed=42)
        x = torch.linspace(-0.9, 0.9, 30).unsqueeze(1)
        with torch.no_grad():
            y_before = layer(x).clone()

        layer.insert_knot(0.2)

        with torch.no_grad():
            y_after = layer(x)

        assert torch.allclose(y_before, y_after, atol=1e-5), \
            f"Boehm insertion shifted curve, max error {(y_before - y_after).abs().max().item():.2e}"

    def test_output_finite_after_insert(self):
        layer = _make_layer(grid_size=5, spline_order=3)
        layer.insert_knot(0.0)
        x = _rand_input(batch=16)
        assert torch.isfinite(layer(x)).all()

    def test_returns_migration_info(self):
        layer = _make_layer(grid_size=5, spline_order=3)
        info = layer.insert_knot(0.1)
        assert info is not None
        assert "old_param" in info
        assert "new_param" in info
        assert "i" in info


# ============================================================
# 6. 区间残差分析 compute_interval_residuals
# ============================================================

class TestNParams:
    def test_n_params_matches_manual_count(self):
        layer = _make_layer(input_dim=1, output_dim=1, grid_size=3, spline_order=3)
        expected = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        assert layer.n_params == expected

    def test_n_params_increases_after_expand(self):
        layer = _make_layer(grid_size=3)
        before = layer.n_params
        layer.expand_grid(6)
        assert layer.n_params > before

    def test_n_params_in_extra_repr(self):
        layer = _make_layer(grid_size=3)
        assert "n_params" in layer.extra_repr()


class TestIntervalResiduals:
    def test_output_shape(self):
        layer = _make_layer(grid_size=4)
        x = _rand_input(batch=50)
        y_true = torch.zeros(50, 1)
        res, grid_pts = layer.compute_interval_residuals(x, y_true)
        assert res.shape == (4,)        # G 个区间
        assert grid_pts.shape == (5,)   # G+1 个端点

    def test_residuals_nonnegative(self):
        layer = _make_layer(grid_size=4)
        x = _rand_input(batch=50)
        y_true = torch.zeros(50, 1)
        res, _ = layer.compute_interval_residuals(x, y_true)
        assert (res >= 0).all()

    def test_residuals_finite(self):
        layer = _make_layer(grid_size=4)
        x = _rand_input(batch=50)
        y_true = layer(x).detach()
        res, _ = layer.compute_interval_residuals(x, y_true)
        assert torch.isfinite(res).all()

    def test_zero_residual_when_perfect_fit(self):
        """当预测值等于真实值时，残差应为 0。"""
        layer = _make_layer(grid_size=4)
        x = _rand_input(batch=50)
        with torch.no_grad():
            y_true = layer(x)
        res, _ = layer.compute_interval_residuals(x, y_true)
        assert torch.allclose(res, torch.zeros_like(res), atol=1e-6)
