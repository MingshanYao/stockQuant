"""测试 news 模块：个股新闻 + 全球资讯。"""

import pandas as pd
import pytest


class TestGetStockNews:
    """get_stock_news 测试。"""

    def test_returns_dataframe(self):
        from stockquant.signals.news import get_stock_news

        df = get_stock_news("600519")
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        from stockquant.signals.news import get_stock_news

        df = get_stock_news("600519")
        expected = ["title", "content", "time", "source", "url"]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    def test_respects_page_size(self):
        from stockquant.signals.news import get_stock_news

        df = get_stock_news("600519", page_size=5)
        assert len(df) <= 5, f"请求5条, 实际返回{len(df)}行"

    def test_normalizes_code(self):
        from stockquant.signals.news import get_stock_news

        df = get_stock_news("SH600519", page_size=5)
        assert isinstance(df, pd.DataFrame)


class TestGetGlobalNews:
    """get_global_news 测试。"""

    def test_returns_dataframe(self):
        from stockquant.signals.news import get_global_news

        df = get_global_news(page_size=10)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        from stockquant.signals.news import get_global_news

        df = get_global_news(page_size=10)
        expected = ["title", "summary", "time"]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"
