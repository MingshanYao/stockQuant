"""测试 - AkShareDataSource: eastmoney 主路径与新浪 fallback。"""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import pandas as pd
import pytest

from stockquant.data.source_akshare import AkShareDataSource


def _fake_eastmoney_df() -> pd.DataFrame:
    """模拟东方财富 stock_zh_a_hist 返回结构。"""
    return pd.DataFrame({
        "日期": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "开盘": [10.0, 10.5],
        "最高": [11.0, 11.2],
        "最低": [9.8, 10.3],
        "收盘": [10.8, 11.0],
        "成交量": [1_000_000, 1_200_000],
        "成交额": [10_500_000.0, 13_000_000.0],
        "换手率": [0.5, 0.6],
        "涨跌幅": [0.0, 0.0185],
        "涨跌额": [0.0, 0.2],
    })


def _fake_sina_df() -> pd.DataFrame:
    """模拟新浪 stock_zh_a_daily 返回结构。"""
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "open": [10.0, 10.5],
        "high": [11.0, 11.2],
        "low":  [9.8, 10.3],
        "close": [10.8, 11.0],
        "volume": [1_000_000, 1_200_000],
        "amount": [10_500_000.0, 13_000_000.0],
        "outstanding_share": [1e9, 1e9],
        "turnover": [0.005, 0.006],
    })


class TestEastmoneyPrimary:
    def test_eastmoney_success_returns_data(self):
        src = AkShareDataSource()
        with patch("stockquant.data.source_akshare.ak.stock_zh_a_hist") as mock_hist, \
             patch("stockquant.data.source_akshare.ak.stock_zh_a_daily") as mock_sina:
            mock_hist.return_value = _fake_eastmoney_df()
            df = src.get_daily_bars("600000", "2024-01-01", "2024-01-05", adjust="hfq")

        assert not df.empty
        assert {"code", "date", "open", "high", "low", "close", "volume"}.issubset(df.columns)
        assert (df["code"] == "600000").all()
        # eastmoney 路径成功，sina 不应被调用
        mock_sina.assert_not_called()


class TestSinaFallback:
    def test_eastmoney_failure_falls_back_to_sina(self):
        src = AkShareDataSource()
        with patch("stockquant.data.source_akshare.ak.stock_zh_a_hist") as mock_hist, \
             patch("stockquant.data.source_akshare.ak.stock_zh_a_daily") as mock_sina:
            mock_hist.side_effect = ConnectionError("eastmoney refused")
            mock_sina.return_value = _fake_sina_df()
            df = src.get_daily_bars("600000", "2024-01-01", "2024-01-05", adjust="hfq")

        assert not df.empty
        assert (df["code"] == "600000").all()
        mock_sina.assert_called()

    def test_eastmoney_empty_falls_back_to_sina(self):
        src = AkShareDataSource()
        with patch("stockquant.data.source_akshare.ak.stock_zh_a_hist") as mock_hist, \
             patch("stockquant.data.source_akshare.ak.stock_zh_a_daily") as mock_sina:
            mock_hist.return_value = pd.DataFrame()  # 主路径返回空
            mock_sina.return_value = _fake_sina_df()
            df = src.get_daily_bars("600000", "2024-01-01", "2024-01-05", adjust="hfq")

        assert not df.empty
        mock_sina.assert_called()

    def test_both_fail_returns_empty(self):
        src = AkShareDataSource()
        with patch("stockquant.data.source_akshare.ak.stock_zh_a_hist") as mock_hist, \
             patch("stockquant.data.source_akshare.ak.stock_zh_a_daily") as mock_sina:
            mock_hist.side_effect = ConnectionError("eastmoney refused")
            mock_sina.side_effect = ConnectionError("sina refused")
            df = src.get_daily_bars("600000", "2024-01-01", "2024-01-05", adjust="hfq")

        assert df.empty
