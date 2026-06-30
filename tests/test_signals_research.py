"""测试 research 模块：研报 + 一致预期EPS。"""

import pandas as pd
import pytest


class TestGetResearchReports:
    """get_research_reports 测试。"""

    def test_returns_dataframe(self):
        from stockquant.signals.research import get_research_reports

        df = get_research_reports("600519", max_pages=1)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        from stockquant.signals.research import get_research_reports

        df = get_research_reports("600519", max_pages=1)
        expected = [
            "title", "publish_date", "org_name", "info_code",
            "eps_cur", "eps_next", "eps_next2", "rating",
            "industry", "attach_pages",
        ]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    def test_normalizes_code(self):
        from stockquant.signals.research import get_research_reports

        df = get_research_reports("SH600519", max_pages=1)
        assert isinstance(df, pd.DataFrame)

    def test_unknown_code_returns_empty(self):
        from stockquant.signals.research import get_research_reports

        df = get_research_reports("999999", max_pages=1)
        assert isinstance(df, pd.DataFrame)


class TestGetIndustryReports:
    """get_industry_reports 测试。"""

    def test_returns_dataframe(self):
        from stockquant.signals.research import get_industry_reports

        df = get_industry_reports("*", max_pages=1)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        from stockquant.signals.research import get_industry_reports

        df = get_industry_reports("*", max_pages=1)
        expected = [
            "title", "publish_date", "org_name", "info_code",
            "industry_name", "industry_code", "rating",
            "report_type", "attach_pages", "attach_size",
        ]
        for col in expected:
            assert col in df.columns, f"缺少列: {col}"

    def test_specific_industry_code_works(self):
        from stockquant.signals.research import get_industry_reports

        df = get_industry_reports("1238", max_pages=1)
        assert isinstance(df, pd.DataFrame)


class TestDownloadReportPdf:
    """download_report_pdf 测试。"""

    def test_empty_info_code_returns_none(self):
        from stockquant.signals.research import download_report_pdf

        assert download_report_pdf({}) is None

    def test_invalid_record_returns_none(self):
        from stockquant.signals.research import download_report_pdf

        assert download_report_pdf({"info_code": "bogus"}, target_dir="/tmp") is None


class TestGetConsensusEps:
    """get_consensus_eps 测试。"""

    def test_returns_dataframe(self):
        from stockquant.signals.research import get_consensus_eps

        df = get_consensus_eps("600519")
        assert isinstance(df, pd.DataFrame)

    def test_small_stock_may_be_empty(self):
        from stockquant.signals.research import get_consensus_eps

        df = get_consensus_eps("688017")
        assert isinstance(df, pd.DataFrame)

    def test_normalizes_code(self):
        from stockquant.signals.research import get_consensus_eps

        df = get_consensus_eps("SH600519")
        assert isinstance(df, pd.DataFrame)
