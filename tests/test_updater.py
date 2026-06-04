"""测试 - DataUpdater: 空 DF 应计为 failed 而非 success（修复历史 bug）。"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from stockquant.data.updater import DataUpdater


@pytest.fixture
def updater(tmp_path, monkeypatch):
    """构造一个隔离的 DataUpdater，用临时 DuckDB。"""
    from stockquant.utils.config import Config
    cfg = Config()
    monkeypatch.setattr(cfg, "_data", {**cfg._data})
    cfg.set("database.path", str(tmp_path / "test.duckdb"))
    cfg.set("data_fetch.max_workers", 1)  # 串行，便于断言
    cfg.set("data_fetch.queue_maxsize", 10)

    from stockquant.data.database import Database
    db = Database()
    return DataUpdater(config=cfg, db=db)


class TestBatchUpdateConsumeBug:
    """历史 bug：fetch 返回空 DF 时被静默标记为成功，导致 0 行入库但日志显示成功。"""

    def test_empty_dataframe_counts_as_failed(self, updater):
        with patch.object(
            updater._source, "get_daily_bars",
            return_value=pd.DataFrame(),  # 模拟上游返回空
        ):
            results = updater.update_codes_daily(
                ["000001", "600000"],
                start_date="2024-01-01",
                end_date="2024-01-05",
            )
        # 空 DF 不应进入 results 字典
        assert results == {}

    def test_exception_counts_as_failed(self, updater):
        with patch.object(
            updater._source, "get_daily_bars",
            side_effect=ConnectionError("network"),
        ):
            results = updater.update_codes_daily(
                ["000001"],
                start_date="2024-01-01",
                end_date="2024-01-05",
            )
        assert results == {}

    def test_success_counts(self, updater):
        df = pd.DataFrame({
            "code":   ["000001", "000001"],
            "date":   pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open":   [10.0, 10.5],
            "high":   [11.0, 11.2],
            "low":    [9.8, 10.3],
            "close":  [10.8, 11.0],
            "volume": [1_000_000, 1_200_000],
            "amount": [10_500_000.0, 13_000_000.0],
            "turnover":   [0.5, 0.6],
            "pct_change": [0.0, 0.0185],
            "change":     [0.0, 0.2],
        })
        with patch.object(updater._source, "get_daily_bars", return_value=df):
            results = updater.update_codes_daily(
                ["000001"],
                start_date="2024-01-01",
                end_date="2024-01-05",
            )
        assert results.get("000001", 0) > 0
