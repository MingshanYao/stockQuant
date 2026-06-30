"""测试 announcement 模块：巨潮公告。"""

import pandas as pd
import pytest


class TestGetAnnouncements:
    """get_announcements 测试。"""

    def test_returns_dataframe(self):
        from stockquant.signals.announcement import get_announcements

        df = get_announcements("600519", page_size=10)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        from stockquant.signals.announcement import get_announcements

        df = get_announcements("600519", page_size=5)
        expected = ["title", "type", "date", "url"]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    def test_normalizes_code(self):
        from stockquant.signals.announcement import get_announcements

        df = get_announcements("SH600519", page_size=5)
        assert isinstance(df, pd.DataFrame)

    def test_unknown_code_may_return_empty(self):
        from stockquant.signals.announcement import get_announcements

        df = get_announcements("999999", page_size=5)
        assert isinstance(df, pd.DataFrame)
