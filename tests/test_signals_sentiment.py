"""测试 sentiment 模块：互动易 + 热榜 + 人气榜 + 概念命中。"""

import pandas as pd
import pytest


class TestGetIrmQa:
    """get_irm_qa 测试。"""

    def test_returns_dataframe(self):
        from stockquant.signals.sentiment import get_irm_qa

        df = get_irm_qa("002594", page_size=5)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        from stockquant.signals.sentiment import get_irm_qa

        df = get_irm_qa("002594", page_size=5)
        expected = ["code", "company", "question", "answer", "answerer", "ask_time"]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    def test_normalizes_code(self):
        from stockquant.signals.sentiment import get_irm_qa

        df = get_irm_qa("SZ002594", page_size=5)
        assert isinstance(df, pd.DataFrame)


class TestGetThsHotList:
    """get_ths_hot_list 测试。"""

    def test_returns_dataframe(self):
        from stockquant.signals.sentiment import get_ths_hot_list

        df = get_ths_hot_list()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        from stockquant.signals.sentiment import get_ths_hot_list

        df = get_ths_hot_list()
        expected = ["rank", "code", "name", "heat", "pct", "rank_chg", "concepts", "tag"]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    def test_hour_period_works(self):
        from stockquant.signals.sentiment import get_ths_hot_list

        df = get_ths_hot_list("hour")
        assert isinstance(df, pd.DataFrame)

    def test_day_period_works(self):
        from stockquant.signals.sentiment import get_ths_hot_list

        df = get_ths_hot_list("day")
        assert isinstance(df, pd.DataFrame)


class TestGetEmHotRank:
    """get_em_hot_rank 测试。"""

    def test_returns_dataframe(self):
        from stockquant.signals.sentiment import get_em_hot_rank

        df = get_em_hot_rank(10)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        from stockquant.signals.sentiment import get_em_hot_rank

        df = get_em_hot_rank(10)
        expected = ["rank", "code", "name", "price", "pct", "rank_chg"]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"


class TestGetEmHotConcept:
    """get_em_hot_concept 测试。"""

    def test_returns_dataframe(self):
        from stockquant.signals.sentiment import get_em_hot_concept

        df = get_em_hot_concept("600519")
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        from stockquant.signals.sentiment import get_em_hot_concept

        df = get_em_hot_concept("600519")
        expected = ["concept", "bk", "hit"]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    def test_normalizes_code(self):
        from stockquant.signals.sentiment import get_em_hot_concept

        df = get_em_hot_concept("SH600519")
        assert isinstance(df, pd.DataFrame)
