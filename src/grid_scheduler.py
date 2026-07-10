"""
ExtendGridOnPlateau: 均匀 Grid 自动生长调度器
==============================================
模仿 torch.optim.lr_scheduler.ReduceLROnPlateau 的 API 设计。

核心策略（与 visualization/index.html 一致）:
    Loss Plateau 检测
        ↓
    对所有 DynamicKANLayer 执行 refine_grid（均匀翻倍）
        ↓
    F.interpolate 迁移控制点 → 继续训练

关键行为：
  - 从 G=3 起步，plateau 时 G → min(G×2, max_grid)
  - Grid 始终保持均匀分布
  - 翻倍后重置 best_loss（与浏览器 demo 一致），便于在新 Grid 上继续优化

EcoGrowScheduler: 考虑参数收益的可逆 Grid 扩容（Cost-Aware Reversible Grid Expansion）
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.kan_layer import DynamicKANLayer


def count_trainable_parameters(model: nn.Module) -> int:
    """统计模型中 requires_grad=True 的参数元素总数。"""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@dataclass
class EcoGrowResult:
    """
    EcoGrowScheduler.step() 的返回结果。

    属性:
        model: 当前应继续训练使用的模型（扩容拒绝时可能为恢复的备份模型）
        action: none | growth_started | growth_accepted | growth_rejected | growth_blocked
        optimizer_reset_required: 是否需要重建 optimizer
        event: 扩容决策事件字典（仅 accept/reject 时非空）
    """
    model: nn.Module
    action: str
    optimizer_reset_required: bool
    event: Optional[Dict[str, Any]] = None


class EcoGrowScheduler:
    """
    EcoGrow-KAN 调度器：试探性扩容 → 验证集评估 → 接受或回退。

    状态机:
        normal   — 检测训练 loss plateau，触发试探性扩容
        trial    — 试用 trial_epochs 轮，跟踪最佳验证 loss
        cooldown — 扩容决策后冷却，避免立刻再次扩容

    扩容接受条件（同时满足）:
        relative_improvement >= min_improvement
        efficiency_score >= min_efficiency

    使用示例见 README「EcoGrow-KAN」章节。
    """

    EPS = 1e-12

    def __init__(
        self,
        model: nn.Module,
        patience: int = 8,
        max_grid: int = 50,
        growth_factor: float = 2,
        min_delta_rel: float = 0.005,
        min_delta: float = 1e-8,
        trial_epochs: int = 30,
        min_improvement: float = 0.01,
        min_efficiency: float = 0.05,
        cooldown_epochs: int = 30,
        verbose: bool = True,
    ):
        self.model = model
        self.patience = patience
        self.max_grid = max_grid
        self.growth_factor = growth_factor
        self.min_delta_rel = min_delta_rel
        self.min_delta = min_delta
        self.trial_epochs = trial_epochs
        self.min_improvement = min_improvement
        self.min_efficiency = min_efficiency
        self.cooldown_epochs = cooldown_epochs
        self.verbose = verbose

        self.state = "normal"
        self.best_loss = float("inf")
        self.num_bad_epochs = 0
        self._at_max_warned = False

        self.backup_model: Optional[nn.Module] = None
        self.grid_before = 0
        self.params_before = 0
        self.val_loss_before = float("inf")
        self.best_trial_val_loss = float("inf")
        self.trial_start_epoch: Optional[int] = None
        self.trial_epochs_elapsed = 0
        self.cooldown_remaining = 0

        self.events: List[Dict[str, Any]] = []
        self.rejected_growths: set[tuple[int, int]] = set()
        self.growth_exhausted = False
        self._blocked_events_recorded: set[tuple[int, int]] = set()

        self._kan_layers = self._find_kan_layers(model)

    def _find_kan_layers(self, model: nn.Module) -> List[DynamicKANLayer]:
        layers = [
            module for module in model.modules()
            if isinstance(module, DynamicKANLayer)
        ]
        if not layers:
            raise ValueError("No DynamicKANLayer found in the model.")
        return layers

    def _primary_grid_size(self) -> int:
        return self._kan_layers[0].grid_size

    def _all_at_max(self) -> bool:
        return all(layer.grid_size >= self.max_grid for layer in self._kan_layers)

    @staticmethod
    def _is_valid_loss(loss: float) -> bool:
        return math.isfinite(loss)

    def _effective_delta(self) -> float:
        if self.min_delta_rel > 0 and self.best_loss < float("inf"):
            return max(self.min_delta, self.best_loss * self.min_delta_rel)
        return self.min_delta

    def _update_plateau_counter(self, train_loss: float) -> None:
        if train_loss < self.best_loss - self._effective_delta():
            self.best_loss = train_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

    def _reset_plateau_counter(self) -> None:
        self.best_loss = float("inf")
        self.num_bad_epochs = 0

    def _deepcopy_model(self, model: nn.Module) -> nn.Module:
        device = next(model.parameters()).device
        backup = copy.deepcopy(model)
        return backup.to(device)

    def _refine_all_layers(self, new_grid: int) -> None:
        for layer in self._kan_layers:
            if layer.grid_size < new_grid:
                layer.refine_grid(new_grid)

    def _compute_metrics(
        self,
        val_loss_before: float,
        best_trial_val_loss: float,
        params_before: int,
        params_after: int,
    ) -> Dict[str, float]:
        relative_improvement = (
            (val_loss_before - best_trial_val_loss)
            / max(abs(val_loss_before), self.EPS)
        )
        relative_param_growth = (
            (params_after - params_before)
            / max(params_before, 1)
        )
        efficiency_score = (
            relative_improvement
            / max(relative_param_growth, self.EPS)
        )
        return {
            "relative_improvement": relative_improvement,
            "relative_param_growth": relative_param_growth,
            "efficiency_score": efficiency_score,
        }

    def _block_growth(self, old_grid: int, new_grid: int, epoch: int) -> EcoGrowResult:
        """跳过此前已被拒绝的扩容路径，仅记录一次 blocked 事件。"""
        self._reset_plateau_counter()
        self.growth_exhausted = True

        event: Optional[Dict[str, Any]] = None
        growth_key = (old_grid, new_grid)
        if growth_key not in self._blocked_events_recorded:
            self._blocked_events_recorded.add(growth_key)
            event = {
                "epoch": epoch,
                "old_grid": old_grid,
                "new_grid": new_grid,
                "decision": "blocked",
                "val_loss_before": 0.0,
                "best_trial_val_loss": 0.0,
                "relative_improvement": 0.0,
                "relative_param_growth": 0.0,
                "efficiency_score": 0.0,
            }
            self.events.append(event)
            if self.verbose:
                print(
                    f"[EcoGrow] Skip previously rejected growth: "
                    f"G={old_grid} → G={new_grid}"
                )
                print(
                    f"[EcoGrow] Growth exhausted at G={old_grid}. "
                    "No further expansion will be attempted."
                )

        return EcoGrowResult(
            model=self.model,
            action="growth_blocked",
            optimizer_reset_required=False,
            event=event,
        )

    def _start_trial_growth(self, train_loss: float, val_loss: float, epoch: int) -> EcoGrowResult:
        old_grid = self._primary_grid_size()
        new_grid = min(int(old_grid * self.growth_factor), self.max_grid)
        if new_grid <= old_grid:
            self._reset_plateau_counter()
            self.growth_exhausted = True
            return EcoGrowResult(
                model=self.model,
                action="none",
                optimizer_reset_required=False,
            )

        growth_key = (old_grid, new_grid)
        if growth_key in self.rejected_growths:
            return self._block_growth(old_grid, new_grid, epoch)

        self.backup_model = self._deepcopy_model(self.model)
        self.grid_before = old_grid
        self.params_before = count_trainable_parameters(self.model)
        self.val_loss_before = val_loss
        self.best_trial_val_loss = val_loss
        self.trial_start_epoch = epoch
        self.trial_epochs_elapsed = 0

        self._refine_all_layers(new_grid)
        params_after = count_trainable_parameters(self.model)

        self.state = "trial"
        self._reset_plateau_counter()

        if self.verbose:
            print(
                f"[EcoGrow] Trial growth started: G={old_grid} → G={new_grid}"
            )
            print(f"[EcoGrow] Validation baseline: {val_loss:.6f}")
            print(
                f"[EcoGrow] Parameters: {self.params_before} → {params_after}"
            )

        return EcoGrowResult(
            model=self.model,
            action="growth_started",
            optimizer_reset_required=True,
        )

    def _finish_trial(self, epoch: int) -> EcoGrowResult:
        old_grid = self.grid_before
        new_grid = self._primary_grid_size()
        params_after = count_trainable_parameters(self.model)

        metrics = self._compute_metrics(
            self.val_loss_before,
            self.best_trial_val_loss,
            self.params_before,
            params_after,
        )

        accepted = (
            metrics["relative_improvement"] >= self.min_improvement
            and metrics["efficiency_score"] >= self.min_efficiency
        )

        event: Dict[str, Any] = {
            "epoch": epoch,
            "old_grid": old_grid,
            "new_grid": new_grid,
            "decision": "accepted" if accepted else "rejected",
            "val_loss_before": self.val_loss_before,
            "best_trial_val_loss": self.best_trial_val_loss,
            **metrics,
        }
        self.events.append(event)

        optimizer_reset_required = False

        if accepted:
            self.backup_model = None
            if self.verbose:
                print(
                    f"[EcoGrow] Growth accepted: G={old_grid} → G={new_grid}"
                )
                print(
                    f"[EcoGrow] Validation improvement: "
                    f"{metrics['relative_improvement'] * 100:.2f}%"
                )
                print(
                    f"[EcoGrow] Parameter growth: "
                    f"{metrics['relative_param_growth'] * 100:.2f}%"
                )
                print(
                    f"[EcoGrow] Efficiency score: "
                    f"{metrics['efficiency_score']:.3f}"
                )
            action = "growth_accepted"
        else:
            self.model = self.backup_model
            self.backup_model = None
            self._kan_layers = self._find_kan_layers(self.model)
            optimizer_reset_required = True
            if self.verbose:
                print(
                    f"[EcoGrow] Growth rejected: G={new_grid} → "
                    f"rollback to G={old_grid}"
                )
                print(
                    f"[EcoGrow] Validation improvement: "
                    f"{metrics['relative_improvement'] * 100:.2f}%"
                )
                print(
                    f"[EcoGrow] Parameter growth: "
                    f"{metrics['relative_param_growth'] * 100:.2f}%"
                )
                print(
                    f"[EcoGrow] Efficiency score: "
                    f"{metrics['efficiency_score']:.3f}"
                )
            action = "growth_rejected"
            self.rejected_growths.add((old_grid, new_grid))

        self.state = "cooldown"
        self.cooldown_remaining = self.cooldown_epochs
        self.trial_start_epoch = None
        self.trial_epochs_elapsed = 0

        return EcoGrowResult(
            model=self.model,
            action=action,
            optimizer_reset_required=optimizer_reset_required,
            event=event,
        )

    def step(
        self,
        train_loss: float,
        val_loss: float,
        epoch: Optional[int] = None,
    ) -> EcoGrowResult:
        """
        每个 epoch 结束时调用。

        参数:
            train_loss: 当前 epoch 训练集 loss（用于 plateau 检测）
            val_loss:   当前 epoch 验证集 loss（用于扩容试用评估）
            epoch:      当前 epoch 编号（可选，用于日志与事件记录）

        返回:
            EcoGrowResult
        """
        if not self._is_valid_loss(train_loss) or not self._is_valid_loss(val_loss):
            raise ValueError(
                f"Invalid loss detected: train_loss={train_loss}, val_loss={val_loss}. "
                "EcoGrowScheduler requires finite loss values."
            )

        if self.state == "trial":
            self.best_trial_val_loss = min(self.best_trial_val_loss, val_loss)
            self.trial_epochs_elapsed += 1
            if self.trial_epochs_elapsed >= self.trial_epochs:
                return self._finish_trial(epoch if epoch is not None else -1)
            return EcoGrowResult(
                model=self.model,
                action="none",
                optimizer_reset_required=False,
            )

        if self.state == "cooldown":
            self.cooldown_remaining -= 1
            if self.cooldown_remaining <= 0:
                self.state = "normal"
                self._reset_plateau_counter()
            return EcoGrowResult(
                model=self.model,
                action="none",
                optimizer_reset_required=False,
            )

        # normal state
        if self.growth_exhausted:
            return EcoGrowResult(
                model=self.model,
                action="none",
                optimizer_reset_required=False,
            )

        if self._all_at_max():
            if not self._at_max_warned and self.verbose:
                print(
                    f"All KAN layers reached max_grid={self.max_grid}. "
                    "EcoGrowScheduler passive."
                )
                self._at_max_warned = True
            return EcoGrowResult(
                model=self.model,
                action="none",
                optimizer_reset_required=False,
            )

        self._update_plateau_counter(train_loss)

        if self.num_bad_epochs >= self.patience:
            return self._start_trial_growth(train_loss, val_loss, epoch if epoch is not None else -1)

        return EcoGrowResult(
            model=self.model,
            action="none",
            optimizer_reset_required=False,
        )

    @property
    def num_accepted(self) -> int:
        return sum(1 for e in self.events if e["decision"] == "accepted")

    @property
    def num_rejected(self) -> int:
        return sum(1 for e in self.events if e["decision"] == "rejected")

    @property
    def num_blocked(self) -> int:
        return sum(1 for e in self.events if e["decision"] == "blocked")


class ExtendGridOnPlateau:
    """
    均匀 Grid 自动生长调度器。

    当监控的 loss 在 patience 个 epoch 内没有改善时：
    对每个未达上限的 DynamicKANLayer 调用 refine_grid(new_G)，
    其中 new_G = min(current_G * 2, max_grid)。

    参数:
        model:          包含 DynamicKANLayer 的 nn.Module
        patience:       容忍停滞的 epoch 数（默认 8）
        max_grid:       Grid 区间数硬上限（默认 50，与浏览器 demo 一致）
        verbose:        是否打印翻倍日志
        min_delta:      视为「改善」的最小 loss 绝对变化量
        min_delta_rel:  视为「改善」的最小 loss 相对变化比例
                        实际阈值 = max(min_delta, best_loss * min_delta_rel)

    使用示例:
        scheduler = ExtendGridOnPlateau(model, patience=8, max_grid=50)
        for epoch in range(num_epochs):
            train_one_epoch(...)
            refined, events = scheduler.step(loss.item(), epoch=epoch)
            if refined:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    """

    def __init__(self, model, patience=8, max_grid=50, verbose=True,
                 min_delta=1e-8, min_delta_rel=0.005):
        self.model = model
        self.patience = patience
        self.max_grid = max_grid
        self.verbose = verbose
        self.min_delta = min_delta
        self.min_delta_rel = min_delta_rel

        self.best_loss = float('inf')
        self.num_bad_epochs = 0
        self.refine_history = []   # [(epoch, old_g, new_g), ...]
        self._at_max_warned = False

        self._kan_layers = self._find_kan_layers(model)

    def _find_kan_layers(self, model):
        layers = []
        for module in model.modules():
            if isinstance(module, DynamicKANLayer):
                layers.append(module)
        if len(layers) == 0:
            raise ValueError("No DynamicKANLayer found in the model.")
        return layers

    def _all_at_max(self):
        return all(layer.grid_size >= self.max_grid for layer in self._kan_layers)

    def step(self, current_loss, epoch=None):
        """
        每个 epoch 结束时调用。

        返回:
            refined: bool，本轮是否触发了均匀翻倍
            events:  list[tuple]，[(epoch, old_g, new_g), ...]
        """
        refined = False
        events = []

        if self._all_at_max():
            if not self._at_max_warned and self.verbose:
                print(f"All KAN layers reached max_grid={self.max_grid}. Scheduler passive.")
                self._at_max_warned = True
            return refined, events

        effective_delta = self.min_delta
        if self.min_delta_rel > 0 and self.best_loss < float('inf'):
            effective_delta = max(self.min_delta, self.best_loss * self.min_delta_rel)

        if current_loss < self.best_loss - effective_delta:
            self.best_loss = current_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            for layer in self._kan_layers:
                if layer.grid_size >= self.max_grid:
                    continue
                old_g = layer.grid_size
                new_g = min(old_g * 2, self.max_grid)
                if new_g <= old_g:
                    continue
                layer.refine_grid(new_g)
                refined = True
                record = (epoch, old_g, new_g)
                self.refine_history.append(record)
                events.append(record)
                if self.verbose:
                    epoch_str = f"Epoch {epoch}: " if epoch is not None else ""
                    print(
                        f"{epoch_str}Loss plateau "
                        f"(best={self.best_loss:.6f}, current={current_loss:.6f}). "
                        f"Uniform refine_grid {old_g} → {new_g}."
                    )

            if refined:
                self.best_loss = float('inf')
                self.num_bad_epochs = 0

        return refined, events

    def state_dict(self):
        return {
            'best_loss': self.best_loss,
            'num_bad_epochs': self.num_bad_epochs,
            'refine_history': self.refine_history,
            'at_max_warned': self._at_max_warned,
        }

    def load_state_dict(self, state_dict):
        self.best_loss = state_dict['best_loss']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.refine_history = state_dict['refine_history']
        self._at_max_warned = state_dict.get('at_max_warned', False)
