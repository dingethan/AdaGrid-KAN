"""
EcoGrowScheduler 单元测试
=========================
快速模拟 loss 序列，无需完整训练。
"""

from __future__ import annotations

import copy
import sys
import os

import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.grid_scheduler import EcoGrowScheduler, EcoGrowResult, count_trainable_parameters
from src.kan_layer import DynamicKANLayer


def _make_model(grid_size: int = 3, seed: int = 0) -> DynamicKANLayer:
    torch.manual_seed(seed)
    return DynamicKANLayer(in_dim=1, out_dim=1, grid_size=grid_size, spline_order=3)


def _plateau_train_losses(
    start: float,
    steps: int,
    end: float,
) -> list[float]:
    """模拟先下降后 plateau 的训练 loss 序列。"""
    losses = []
    current = start
    for i in range(steps):
        if i < steps // 2:
            current = current * 0.85
        else:
            current = end
        losses.append(current)
    return losses


class TestEcoGrowGrowthStarted:
    def test_growth_started_increases_grid(self):
        model = _make_model(3)
        scheduler = EcoGrowScheduler(
            model, patience=3, max_grid=50, trial_epochs=5,
            verbose=False,
        )

        train_losses = _plateau_train_losses(1.0, 10, 0.05)
        result = None
        for epoch, train_loss in enumerate(train_losses):
            result = scheduler.step(
                train_loss=train_loss,
                val_loss=0.1,
                epoch=epoch,
            )
            if result.action == "growth_started":
                break

        assert result is not None
        assert result.action == "growth_started"
        assert result.optimizer_reset_required is True
        assert model.grid_size == 6
        assert scheduler.state == "trial"


def _drive_to_growth_started(
    scheduler: EcoGrowScheduler,
    model: DynamicKANLayer,
    patience: int,
    val_loss: float = 0.10,
) -> EcoGrowResult:
    """用递减后 plateau 的训练 loss 触发 growth_started。"""
    train_losses = _plateau_train_losses(1.0, patience + 5, 0.05)
    result = EcoGrowResult(model=model, action="none", optimizer_reset_required=False)
    for epoch, train_loss in enumerate(train_losses):
        result = scheduler.step(
            train_loss=train_loss,
            val_loss=val_loss,
            epoch=epoch,
        )
        if result.action == "growth_started":
            return result
    raise AssertionError("growth_started was not triggered")


class TestEcoGrowAccept:
    def test_growth_accepted_keeps_large_grid(self):
        model = _make_model(3)
        scheduler = EcoGrowScheduler(
            model,
            patience=3,
            max_grid=50,
            trial_epochs=3,
            min_improvement=0.001,
            min_efficiency=0.001,
            cooldown_epochs=5,
            verbose=False,
        )

        result = _drive_to_growth_started(scheduler, model, patience=3)
        assert result.action == "growth_started"
        assert model.grid_size == 6
        start_epoch = 100

        final = None
        for i in range(3):
            val = 0.10 - (i + 1) * 0.02
            final = scheduler.step(
                train_loss=0.04, val_loss=val, epoch=start_epoch + i,
            )

        assert final is not None
        assert final.action == "growth_accepted"
        assert final.event["decision"] == "accepted"
        assert model.grid_size == 6
        assert final.optimizer_reset_required is False
        assert scheduler.backup_model is None


class TestEcoGrowReject:
    def test_growth_rejected_restores_model(self):
        model = _make_model(3, seed=42)
        x = torch.linspace(-1, 1, 20).unsqueeze(1)

        model.eval()
        with torch.no_grad():
            outputs_before = model(x).clone()
        params_before = count_trainable_parameters(model)
        grid_before = model.grid_size

        scheduler = EcoGrowScheduler(
            model,
            patience=3,
            max_grid=50,
            trial_epochs=3,
            min_improvement=0.5,
            min_efficiency=10.0,
            cooldown_epochs=5,
            verbose=False,
        )

        result = _drive_to_growth_started(scheduler, model, patience=3)
        assert result.action == "growth_started"
        assert model.grid_size == 6

        # Trial with no validation improvement → reject
        final = None
        for i in range(3):
            final = scheduler.step(
                train_loss=0.04, val_loss=0.11, epoch=200 + i,
            )

        assert final is not None
        assert final.action == "growth_rejected"
        assert final.event["decision"] == "rejected"
        assert final.optimizer_reset_required is True
        model = final.model
        assert model.grid_size == grid_before
        assert count_trainable_parameters(model) == params_before

        model.eval()
        with torch.no_grad():
            outputs_after = model(x)
        assert torch.allclose(outputs_before, outputs_after, atol=1e-5)


class TestEcoGrowMaxGrid:
    def test_grid_never_exceeds_max_grid(self):
        model = _make_model(40)
        scheduler = EcoGrowScheduler(
            model, patience=2, max_grid=50, trial_epochs=2,
            min_improvement=0.0, min_efficiency=0.0,
            cooldown_epochs=1, verbose=False,
        )

        for epoch in range(30):
            result = scheduler.step(
                train_loss=0.01, val_loss=0.01, epoch=epoch,
            )
            model = result.model
            assert model.grid_size <= 50

        # 40 * 2 = 80 capped to 50
        assert model.grid_size == 50


class TestEcoGrowCooldown:
    def test_no_growth_during_cooldown(self):
        model = _make_model(3)
        scheduler = EcoGrowScheduler(
            model,
            patience=2,
            max_grid=50,
            trial_epochs=2,
            min_improvement=0.5,
            min_efficiency=10.0,
            cooldown_epochs=10,
            verbose=False,
        )

        _drive_to_growth_started(scheduler, model, patience=2)

        # Finish trial with rejection
        result = None
        for i in range(2):
            result = scheduler.step(
                train_loss=0.05, val_loss=0.10, epoch=50 + i,
            )

        assert result is not None
        assert result.action == "growth_rejected"
        model = result.model
        assert model.grid_size == 3
        assert scheduler.state == "cooldown"

        grid_at_reject = model.grid_size

        # During cooldown, plateau should NOT trigger another growth
        for epoch in range(60, 70):
            result = scheduler.step(train_loss=0.05, val_loss=0.10, epoch=epoch)
            assert result.action == "none"
            assert model.grid_size == grid_at_reject

        # After cooldown ends, scheduler returns to normal
        for epoch in range(70, 80):
            result = scheduler.step(train_loss=0.05, val_loss=0.10, epoch=epoch)

        assert scheduler.state in ("normal", "trial", "cooldown")


class TestRejectionMemory:
    def test_rejected_growth_not_retried(self):
        """拒绝 G=24→G=48 后，cooldown 结束不再重复试探。"""
        model = _make_model(24)
        scheduler = EcoGrowScheduler(
            model,
            patience=2,
            max_grid=50,
            trial_epochs=2,
            min_improvement=0.5,
            min_efficiency=10.0,
            cooldown_epochs=3,
            verbose=False,
        )

        _drive_to_growth_started(scheduler, model, patience=2)
        assert model.grid_size == 48

        result = None
        for i in range(2):
            result = scheduler.step(train_loss=0.05, val_loss=0.10, epoch=100 + i)

        assert result is not None
        assert result.action == "growth_rejected"
        model = result.model
        assert model.grid_size == 24
        assert (24, 48) in scheduler.rejected_growths

        for epoch in range(110, 113):
            scheduler.step(train_loss=0.05, val_loss=0.10, epoch=epoch)

        blocked_count = 0
        growth_started_count = 0
        for epoch in range(113, 140):
            train_losses = _plateau_train_losses(1.0, 1, 0.05)
            result = scheduler.step(
                train_loss=train_losses[0],
                val_loss=0.10,
                epoch=epoch,
            )
            if result.action == "growth_started":
                growth_started_count += 1
            if result.action == "growth_blocked":
                blocked_count += 1
                assert result.optimizer_reset_required is False

        assert growth_started_count == 0
        assert blocked_count == 1
        assert model.grid_size == 24
        assert scheduler.growth_exhausted is True

        rejected_events = [e for e in scheduler.events if e["decision"] == "rejected"]
        blocked_events = [e for e in scheduler.events if e["decision"] == "blocked"]
        assert len(rejected_events) == 1
        assert len(blocked_events) == 1


class TestCountParameters:
    def test_count_trainable_parameters(self):
        model = _make_model(3)
        n = count_trainable_parameters(model)
        expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n == expected
        assert n > 0


class TestInvalidLoss:
    def test_nan_loss_raises(self):
        model = _make_model(3)
        scheduler = EcoGrowScheduler(model, verbose=False)
        with pytest.raises(ValueError, match="Invalid loss"):
            scheduler.step(train_loss=float("nan"), val_loss=0.1, epoch=0)


class TestEcoGrowStateDict:
    """EcoGrowScheduler.state_dict / load_state_dict 测试。"""

    def test_state_dict_roundtrip_normal_state(self):
        """normal 状态下 state_dict → load_state_dict 能完整恢复。"""
        model = _make_model(3)
        scheduler = EcoGrowScheduler(
            model, patience=3, max_grid=50, trial_epochs=3, verbose=False,
        )
        # 驱动几步
        for epoch in range(4):
            scheduler.step(train_loss=0.05, val_loss=0.1, epoch=epoch)

        sd = scheduler.state_dict()

        model2 = _make_model(3)
        scheduler2 = EcoGrowScheduler(
            model2, patience=3, max_grid=50, trial_epochs=3, verbose=False,
        )
        scheduler2.load_state_dict(sd)

        assert scheduler2.state == scheduler.state
        assert scheduler2.best_loss == scheduler.best_loss
        assert scheduler2.num_bad_epochs == scheduler.num_bad_epochs
        assert scheduler2.growth_exhausted == scheduler.growth_exhausted
        assert scheduler2.rejected_growths == scheduler.rejected_growths

    def test_state_dict_preserves_rejected_growths(self):
        """拒绝记录在 state_dict 中能被正确序列化与反序列化。"""
        model = _make_model(24)
        scheduler = EcoGrowScheduler(
            model,
            patience=2,
            max_grid=50,
            trial_epochs=2,
            min_improvement=0.5,
            min_efficiency=10.0,
            cooldown_epochs=1,
            verbose=False,
        )
        _drive_to_growth_started(scheduler, model, patience=2)
        for i in range(2):
            result = scheduler.step(train_loss=0.05, val_loss=0.10, epoch=100 + i)
        assert result.action == "growth_rejected"

        sd = scheduler.state_dict()
        assert [24, 48] in sd["rejected_growths"]

        model2 = _make_model(24)
        scheduler2 = EcoGrowScheduler(
            model2, patience=2, max_grid=50, trial_epochs=2,
            min_improvement=0.5, min_efficiency=10.0, verbose=False,
        )
        scheduler2.load_state_dict(sd)
        assert (24, 48) in scheduler2.rejected_growths

    def test_state_dict_keys_complete(self):
        """state_dict 包含所有必要字段。"""
        model = _make_model(3)
        scheduler = EcoGrowScheduler(model, verbose=False)
        sd = scheduler.state_dict()
        required_keys = {
            "state", "best_loss", "num_bad_epochs", "_at_max_warned",
            "grid_before", "params_before", "val_loss_before",
            "best_trial_val_loss", "trial_start_epoch", "trial_epochs_elapsed",
            "cooldown_remaining", "events", "rejected_growths",
            "growth_exhausted", "_blocked_events_recorded",
        }
        assert required_keys.issubset(set(sd.keys()))

    def test_repr_contains_state(self):
        model = _make_model(3)
        scheduler = EcoGrowScheduler(model, verbose=False)
        r = repr(scheduler)
        assert "EcoGrowScheduler" in r
        assert "state=" in r


class TestExtendGridOnPlateauRepr:
    def test_repr_contains_class_name(self):
        model = _make_model(3)
        scheduler = __import__(
            "src.grid_scheduler", fromlist=["ExtendGridOnPlateau"]
        ).ExtendGridOnPlateau(model, verbose=False)
        r = repr(scheduler)
        assert "ExtendGridOnPlateau" in r
        assert "patience=" in r
