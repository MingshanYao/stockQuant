"""测试 industry 模块：行业板块排名。"""

import pandas as pd
import pytest


class TestGetIndustryRanking:
    """get_industry_ranking 测试。"""

    def test_returns_dataframe_with_expected_columns(self):
        """返回非空 DataFrame 含预期列。"""
        from stockquant.signals.industry import get_industry_ranking

        df = get_industry_ranking(top_n=10)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "应有行业数据"
        expected = [
            "rank", "name", "change_pct", "code",
            "up_count", "down_count", "leader", "leader_change",
        ]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    def test_sorted_by_change_desc(self):
        """DataFrame 应按涨跌幅降序排列。"""
        from stockquant.signals.industry import get_industry_ranking

        df = get_industry_ranking(top_n=10)
        if len(df) >= 2:
            changes = df["change_pct"].tolist()
            assert changes == sorted(changes, reverse=True), \
                "行业排名应按涨跌幅降序"

    def test_has_industry_data_for_all_sectors(self):
        """应有足够多的行业（~100个东财行业板块）。"""
        from stockquant.signals.industry import get_industry_ranking

        df = get_industry_ranking(top_n=5)
        assert len(df) >= 50, f"应有~100个行业, 实际{len(df)}"
