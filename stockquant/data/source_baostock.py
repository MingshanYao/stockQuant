"""
BaoStock 数据源适配器 — 完整实现。

BaoStock 提供 22 个 API 覆盖以下领域：
- A 股历史 K 线（日/周/月/分钟）+ 除权除息 + 复权因子
- 季频财务数据（盈利/营运/成长/偿债/现金流/杜邦/快报/预告）
- 证券基本资料 + 行业分类 + 成分股（上证50/沪深300/中证500）
- 交易日历 + 全量证券元信息
- 宏观经济（存贷款利率/准备金率/货币供应量）

频率限制：每天不超过 50,000 次（由调用方控制），接口速率 16 req/s
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import baostock as bs
import pandas as pd

from stockquant.data.data_source import (
    BaseDataSource,
    DataSourceFactory,
    standardize_daily,
    standardize_index,
)
from stockquant.utils.helpers import (
    normalize_stock_code, get_market_prefix, ensure_date,
    RateLimiter, call_with_retries,
)
from stockquant.utils.logger import get_logger

logger = get_logger("data.baostock")

# stock_info 表标准列（与 database schema 对齐）
STOCK_INFO_COLS = [
    "code", "name", "industry", "sector", "market",
    "list_date", "total_shares", "float_shares", "total_cap", "float_cap",
    "out_date", "status", "industry_source",
]

# 复权映射：标准名 → BaoStock adjustflag
_ADJUST_MAP = {"qfq": "2", "hfq": "1", "none": "3"}

# 分钟线频率 → BaoStock frequency 参数
_MINUTE_FREQ_MAP = {"5": "5", "15": "15", "30": "30", "60": "60"}

# ---- 估值 + 状态字段（BaoStock 日线专属，不在 standardize_daily 标准列中）----
_VALUATION_FIELDS = "peTTM,pbMRQ,psTTM,pcfNcfTTM,isST,tradestatus"

# ---- 日线基础字段（不含估值）----
_BASE_DAILY_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "adjustflag,turn,pctChg"
)

# ---- 完整日线字段 = 基础 + 估值 ----
_FULL_DAILY_FIELDS = _BASE_DAILY_FIELDS + "," + _VALUATION_FIELDS

# ---- 成分股查询路由 ----
_INDEX_CONSTITUENT_MAP = {
    "000016": "sz50",
    "000300": "hs300",
    "000905": "zz500",
}

# ---- 财报 category → (method_name, label) ----
_FINANCE_CATEGORIES: dict[str, tuple[str, str]] = {
    "profit":   ("query_profit_data",                "季频盈利能力"),
    "operation": ("query_operation_data",             "季频营运能力"),
    "growth":   ("query_growth_data",                 "季频成长能力"),
    "balance":  ("query_balance_data",                "季频偿债能力"),
    "cash_flow": ("query_cash_flow_data",             "季频现金流量"),
    "dupont":   ("query_dupont_data",                 "季频杜邦指数"),
    "express":  ("query_performance_express_report",  "业绩快报"),
    "forecast": ("query_forecast_report",             "业绩预告"),
}


class BaoStockDataSource(BaseDataSource):
    """基于 BaoStock 的数据源实现。

    BaoStock 客户端为进程内全局单例，登录后保持连接复用；
    析构时自动 logout。
    """

    def __init__(self) -> None:
        self._logged_in = False
        # Baostock 每天不超过 50,000 次（调用方自行控制每日总量）
        self._rate_limiter = RateLimiter(rate=16, burst=200)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _ensure_login(self) -> None:
        if self._logged_in:
            return
        lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"BaoStock 登录失败: {lg.error_msg}")
            return
        self._logged_in = True

    def __del__(self) -> None:
        if getattr(self, "_logged_in", False):
            try:
                bs.logout()
            except Exception:
                pass

    @staticmethod
    def _bs_code(code: str) -> str:
        """6 位代码转 BaoStock 格式（sh.600000 / sz.000001）。"""
        code = normalize_stock_code(code)
        return f"{get_market_prefix(code)}.{code}"

    @staticmethod
    def _bs_index_code(code: str) -> str:
        """指数代码转 BaoStock 格式。

        000xxx → sh（上证/沪深300/中证500），399xxx → sz（深证/创业板）。
        与股票代码不同，000 开头的指数属于上海交易所。
        """
        code = normalize_stock_code(code)
        if code.startswith("399"):
            return f"sz.{code}"
        # 000xxx 及其它指数默认归属上海
        return f"sh.{code}"

    def _rs_to_df(self, rs) -> pd.DataFrame:
        """将 BaoStock ResultSet 转为 DataFrame。

        每次调用计一次查询，节流点在这里统一收敛。
        """
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if rs.error_code != "0":
            logger.warning(f"BaoStock 查询异常 [{rs.error_code}]: {rs.error_msg}")
            self._logged_in = False
        return pd.DataFrame(rows, columns=rs.fields)

    def _query(self, fn, *args, label: str = "", **kwargs) -> pd.DataFrame:
        """带节流和重试的统一查询入口。

        每个 _query() 调用 = 一次 API 请求。
        """
        def _call() -> pd.DataFrame:
            self._throttle()
            self._ensure_login()
            rs = fn(*args, **kwargs)
            return self._rs_to_df(rs)

        return call_with_retries(
            _call, attempts=3, delay=2.0, backoff=2.0,
            label=label or fn.__name__,
        )

    def _to_numeric(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """将指定列转为数值类型。"""
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # ==================================================================
    # BaseDataSource 抽象方法
    # ==================================================================

    # ------------------------------------------------------------------
    def get_stock_list(self) -> pd.DataFrame:
        """获取全量 A 股股票列表。"""
        df = self._query(
            bs.query_stock_basic, label="股票列表",
        )
        df = df[df["type"] == "1"]  # 1=股票
        df = df.rename(columns={"code_name": "name"})
        df["code"] = df["code"].str.split(".").str[1]
        return df[["code", "name"]].reset_index(drop=True)

    # ------------------------------------------------------------------
    def get_daily_bars(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
        adjust: str = "hfq",
    ) -> pd.DataFrame:
        """获取日线行情（不复权价格 + 后复权因子）。

        存储策略：OHLC 为不复权价格（adjustflag='3'），
        adj_factor 为后复权因子（backAdjustFactor），
        复权价格 = 不复权价格 × adj_factor。

        Parameters
        ----------
        adjust : str
            保留参数兼容性，实际存储始终不复权。
        """
        bs_code = self._bs_code(code)
        sd = str(ensure_date(start_date))
        ed = str(ensure_date(end_date))

        # 不复权日线（adjustflag='3'）
        df = self._query(
            bs.query_history_k_data_plus,
            bs_code,
            _FULL_DAILY_FIELDS,
            start_date=sd, end_date=ed,
            frequency="d",
            adjustflag="3",
            label=f"日线 {code}",
        )

        if df.empty:
            return df

        numeric_cols = [
            "open", "high", "low", "close", "volume", "amount", "turn", "pctChg",
            "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",
        ]
        self._to_numeric(df, numeric_cols)

        # isST → int
        if "isST" in df.columns:
            df["isST"] = pd.to_numeric(df["isST"], errors="coerce").fillna(0).astype(int)

        # 复权因子先置 1.0，后续由 update_adj_factors 批量补齐
        df["adj_factor"] = 1.0

        return standardize_daily(df, normalize_stock_code(code), volume_unit="shares")

    # ------------------------------------------------------------------
    def get_minute_bars(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
        freq: str = "5",
        adjust: str = "hfq",
    ) -> pd.DataFrame:
        """获取分钟线行情（BaoStock 独有）。

        Parameters
        ----------
        freq : str
            "5", "15", "30", "60" 分钟。
        adjust : str
            qfq / hfq / none。
        """
        bs_code = self._bs_code(code)
        sd = str(ensure_date(start_date))
        ed = str(ensure_date(end_date))
        if freq not in _MINUTE_FREQ_MAP:
            raise ValueError(f"分钟线频率不支持: {freq}，可选: {list(_MINUTE_FREQ_MAP.keys())}")

        fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"

        df = self._query(
            bs.query_history_k_data_plus,
            bs_code,
            fields,
            start_date=sd, end_date=ed,
            frequency=_MINUTE_FREQ_MAP[freq],
            adjustflag=_ADJUST_MAP.get(adjust, "1"),
            label=f"分钟线 {code} ({freq}min)",
        )

        if df.empty:
            return df

        self._to_numeric(df, ["open", "high", "low", "close", "volume", "amount"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        df["code"] = normalize_stock_code(code)
        return df

    # ------------------------------------------------------------------
    def get_index_daily(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> pd.DataFrame:
        """获取指数日线行情。"""
        bs_code = self._bs_index_code(code)
        sd = str(ensure_date(start_date))
        ed = str(ensure_date(end_date))

        df = self._query(
            bs.query_history_k_data_plus,
            bs_code,
            "date,open,high,low,close,volume,amount",
            start_date=sd, end_date=ed,
            frequency="d",
            label=f"指数日线 {code}",
        )

        if df.empty:
            return df

        self._to_numeric(df, ["open", "high", "low", "close", "volume", "amount"])
        return standardize_index(df, normalize_stock_code(code))

    # ------------------------------------------------------------------
    def get_index_constituents(self, index_code: str) -> list[str]:
        """获取指数成分股。

        支持上证50(000016)、沪深300(000300)、中证500(000905)。
        """
        bs_key = _INDEX_CONSTITUENT_MAP.get(index_code)
        if bs_key is None:
            logger.warning(
                f"BaoStock 仅支持 {list(_INDEX_CONSTITUENT_MAP.keys())} 成分股查询，"
                f"不支持 {index_code}"
            )
            raise NotImplementedError(
                f"BaoStock 暂不支持指数 {index_code} 的成分股查询，"
                f"支持: 000016/000300/000905"
            )

        method_map = {
            "sz50": bs.query_sz50_stocks,
            "hs300": bs.query_hs300_stocks,
            "zz500": bs.query_zz500_stocks,
        }
        df = self._query(
            method_map[bs_key],
            label=f"成分股 {index_code}",
        )
        if df.empty:
            return []
        return sorted(df["code"].str.split(".").str[1].unique().tolist())

    # ------------------------------------------------------------------
    def get_finance_data(
        self,
        code: str,
        year: int | None = None,
        quarter: int | None = None,
        category: str = "profit",
    ) -> pd.DataFrame:
        """获取季频财务数据。

        Parameters
        ----------
        code : str
            股票代码（纯 6 位数字）。
        year : int, optional
            统计年份，省略时取当前年份。
        quarter : int, optional
            统计季度 (1/2/3/4)，省略时取最近有数据的季度。
        category : str
            数据类型：
            - ``"profit"``    季频盈利能力（默认）
            - ``"operation"`` 季频营运能力
            - ``"growth"``    季频成长能力
            - ``"balance"``   季频偿债能力
            - ``"cash_flow"`` 季频现金流量
            - ``"dupont"``    季频杜邦指数
            - ``"express"``   业绩快报（不支持 year/quarter，用 start_date/end_date）
            - ``"forecast"``  业绩预告（不支持 year/quarter，用 start_date/end_date）
        """
        if category not in _FINANCE_CATEGORIES:
            raise ValueError(
                f"不支持的数据类型: {category}，可选: {list(_FINANCE_CATEGORIES.keys())}"
            )

        bs_code = self._bs_code(code)
        method_name, label = _FINANCE_CATEGORIES[category]

        fn = getattr(bs, method_name)

        # express / forecast 使用 start_date/end_date 参数，不支持 year/quarter
        if category in ("express", "forecast"):
            start = f"{year or 2020}-01-01"
            end = f"{year or dt.date.today().year}-12-31"
            return self._query(
                fn, bs_code, start_date=start, end_date=end,
                label=f"{label} {code}",
            )

        # 标准季频 API: year, quarter 可选
        kwargs: dict[str, Any] = {"code": bs_code}
        if year is not None:
            kwargs["year"] = year
        if quarter is not None:
            kwargs["quarter"] = quarter
        return self._query(fn, **kwargs, label=f"{label} {code}")

    # ------------------------------------------------------------------
    def get_stock_info(self) -> pd.DataFrame:
        """获取全市场股票基本信息（CSRC 行业分类 + 退市标记 + 上市日期）。

        从 ``query_stock_basic`` 取证券基本资料（含退市日期/状态），
        从 ``query_stock_industry``（不传 code）取全量 CSRC 行业分类，
        合并后输出标准列。市值/股本字段留空，由 DataUpdater 后续补齐。
        """
        logger.info("通过 BaoStock 获取全市场股票基本信息...")

        # Step 1: 行业分类（全量，CSRC 证监会分类）
        logger.info("  拉取行业分类（query_stock_industry 全量）...")
        industry_df = self._query(
            bs.query_stock_industry,
            label="行业分类(全量)",
        )
        industry_map: dict[str, str] = {}
        if not industry_df.empty and "code" in industry_df.columns:
            for _, row in industry_df.iterrows():
                c = normalize_stock_code(str(row["code"]).split(".")[-1])
                ind = str(row.get("industry") or "")
                industry_map[c] = ind
        empty_industry_count = sum(1 for v in industry_map.values() if v == "")
        logger.info(
            f"  {len(industry_map)} 只股票有行业信息"
            + (f"（{empty_industry_count} 只为空，多为退市/历史股票）" if empty_industry_count else "")
        )

        # Step 2: 股票基本资料（含退市信息，不预过滤 type）
        logger.info("  拉取股票基本资料（query_stock_basic 全量）...")
        basic_df = self._query(
            bs.query_stock_basic,
            label="基本资料(全量)",
        )
        if basic_df.empty:
            return pd.DataFrame(columns=STOCK_INFO_COLS)

        # 只保留股票类型（含退市），排除指数/可转债等
        basic_df = basic_df[basic_df["type"].isin(["1"])]

        rows: list[dict] = []
        for _, row in basic_df.iterrows():
            code = normalize_stock_code(str(row["code"]).split(".")[-1])
            status_val = int(row["status"]) if str(row.get("status") or "").isdigit() else 1
            out_date = row.get("outDate") or None
            if out_date == "":
                out_date = None
            rows.append({
                "code": code,
                "name": str(row.get("code_name", "")),
                "industry": industry_map.get(code, ""),
                "sector": "",
                "market": self._infer_market(code),
                "list_date": row.get("ipoDate") or None,
                "total_shares": None,
                "float_shares": None,
                "total_cap": None,
                "float_cap": None,
                "out_date": out_date,
                "status": status_val,
                "industry_source": "csrc",
            })

        n_delisted = sum(1 for r in rows if r["status"] == 0)
        logger.info(f"  共 {len(rows)} 只股票（上市: {len(rows) - n_delisted}, 退市: {n_delisted}）")
        return pd.DataFrame(rows, columns=STOCK_INFO_COLS)

    @staticmethod
    def _infer_market(code: str) -> str:
        """根据代码前缀推断所属市场板块。"""
        if code.startswith("688"):
            return "科创板"
        if code.startswith("30"):
            return "创业板"
        if code.startswith("60"):
            return "沪市主板"
        if code.startswith("00"):
            return "深市主板"
        if code.startswith(("4", "8")):
            return "北交所"
        return "其他"

    # ------------------------------------------------------------------
    def get_financials(
        self,
        codes: list[str],
        years: list[int] | None = None,
        quarters: list[int] | None = None,
    ) -> pd.DataFrame:
        """批量获取季频财务数据（合并 profit + growth + balance 三类）。

        对每只股票依次查询盈利能力、成长能力、偿债能力三个 API，
        按 ``statDate`` 合并为单行，输出标准列。

        Parameters
        ----------
        codes : list[str]
            股票代码列表（纯 6 位数字）。
        years : list[int], optional
            年份列表。省略时取当前年份。
        quarters : list[int], optional
            季度列表 [1,2,3,4]。省略时取所有季度。

        Returns
        -------
        DataFrame
            标准列: code, report_date, pub_date, roe, eps, net_profit, revenue,
            gp_margin, np_margin, total_shares, float_shares,
            growth_equity, growth_asset, growth_ni,
            current_ratio, debt_ratio
        """
        import datetime as dt

        if years is None:
            years = [dt.date.today().year]
        if quarters is None:
            quarters = [1, 2, 3, 4]

        all_rows: list[dict] = []
        total = len(codes)

        for i, code in enumerate(codes, 1):
            code = normalize_stock_code(code)
            bs_code = self._bs_code(code)

            for year in years:
                for quarter in quarters:
                    # 盈利能力
                    profit_df = self._query(
                        bs.query_profit_data,
                        code=bs_code, year=year, quarter=quarter,
                        label=f"profit {code}",
                    )

                    # 成长能力
                    growth_df = self._query(
                        bs.query_growth_data,
                        code=bs_code, year=year, quarter=quarter,
                        label=f"growth {code}",
                    )

                    # 偿债能力
                    balance_df = self._query(
                        bs.query_balance_data,
                        code=bs_code, year=year, quarter=quarter,
                        label=f"balance {code}",
                    )

                    # 取第一个有效 statDate（三类数据 statDate 一致）
                    stat_date = None
                    pub_date = None
                    for df in (profit_df, growth_df, balance_df):
                        if not df.empty and "statDate" in df.columns:
                            sd = str(df["statDate"].iloc[0])
                            if sd and sd != "":
                                stat_date = sd
                                break
                    for df in (profit_df, growth_df, balance_df):
                        if not df.empty and "pubDate" in df.columns:
                            pd_val = str(df["pubDate"].iloc[0])
                            if pd_val and pd_val != "":
                                pub_date = pd_val
                                break

                    if stat_date is None:
                        continue

                    row: dict = {
                        "code": code,
                        "report_date": stat_date,
                        "pub_date": pub_date,
                    }

                    # 从 profit 提取
                    if not profit_df.empty:
                        row["roe"] = self._safe_float(profit_df, "roeAvg")
                        row["eps"] = self._safe_float(profit_df, "epsTTM")
                        row["net_profit"] = self._safe_float(profit_df, "netProfit")
                        row["revenue"] = self._safe_float(profit_df, "MBRevenue")
                        row["gp_margin"] = self._safe_float(profit_df, "gpMargin")
                        row["np_margin"] = self._safe_float(profit_df, "npMargin")
                        row["total_shares"] = self._safe_float(profit_df, "totalShare")
                        row["float_shares"] = self._safe_float(profit_df, "liqaShare")

                    # 从 growth 提取
                    if not growth_df.empty:
                        row["growth_equity"] = self._safe_float(growth_df, "YOYEquity")
                        row["growth_asset"] = self._safe_float(growth_df, "YOYAsset")
                        row["growth_ni"] = self._safe_float(growth_df, "YOYNI")

                    # 从 balance 提取
                    if not balance_df.empty:
                        row["current_ratio"] = self._safe_float(balance_df, "currentRatio")
                        row["debt_ratio"] = self._safe_float(balance_df, "debtAssetRatio")

                    all_rows.append(row)

            if i % 50 == 0 or i == total:
                logger.info(f"  [{i}/{total}] 财报拉取进度")

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        for col in ("report_date", "pub_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        logger.info(f"财报拉取完成: {len(df)} 条记录, {df['code'].nunique()} 只股票")
        return df

    @staticmethod
    def _safe_float(df: pd.DataFrame, col: str) -> float | None:
        """安全提取 DataFrame 中的浮点值，空或异常返回 None。"""
        if df.empty or col not in df.columns:
            return None
        val = df[col].iloc[0]
        if val is None or val == "" or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    def get_trade_dates(
        self,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> list[str]:
        """获取区间内交易日历。"""
        sd = str(ensure_date(start_date))
        ed = str(ensure_date(end_date))

        df = self._query(
            bs.query_trade_dates,
            start_date=sd, end_date=ed,
            label="交易日历",
        )
        return df[df["is_trading_day"] == "1"]["calendar_date"].tolist()

    # ==================================================================
    # BaoStock 独有公开方法
    # ==================================================================

    # ------------------------------------------------------------------
    # 除权除息
    # ------------------------------------------------------------------
    def get_dividend_data(
        self,
        code: str,
        year: str | int,
        yearType: str = "report",
    ) -> pd.DataFrame:
        """获取除权除息信息。

        Parameters
        ----------
        code : str
            股票代码。
        year : str | int
            年份，如 "2024" 或 2024。
        yearType : str
            ``"report"`` 预案公告年份（默认），``"operate"`` 除权除息年份。
        """
        bs_code = self._bs_code(code)
        return self._query(
            bs.query_dividend_data,
            code=bs_code, year=str(year), yearType=yearType,
            label=f"除权除息 {code}",
        )

    # ------------------------------------------------------------------
    # 复权因子
    # ------------------------------------------------------------------
    def get_adjust_factor(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> pd.DataFrame:
        """获取复权因子序列。

        可用于自定义前/后复权计算。
        """
        bs_code = self._bs_code(code)
        sd = str(ensure_date(start_date))
        ed = str(ensure_date(end_date))
        return self._query(
            bs.query_adjust_factor,
            code=bs_code, start_date=sd, end_date=ed,
            label=f"复权因子 {code}",
        )

    # ------------------------------------------------------------------
    # 指定日期全量股票状态
    # ------------------------------------------------------------------
    def get_all_stock_on_date(self, day: str | dt.date) -> pd.DataFrame:
        """获取指定交易日所有股票列表及交易状态。

        Parameters
        ----------
        day : str | date
            交易日期。返回 code, tradeStatus, code_name。
        """
        day_str = str(ensure_date(day)) if not isinstance(day, str) else day
        return self._query(
            bs.query_all_stock,
            day=day_str,
            label=f"全量股票 {day_str}",
        )

    # ------------------------------------------------------------------
    # 宏观经济
    # ------------------------------------------------------------------
    def get_deposit_rate(
        self,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
    ) -> pd.DataFrame:
        """获取存款利率。"""
        sd = str(ensure_date(start_date)) if start_date else None
        ed = str(ensure_date(end_date)) if end_date else None
        kwargs = {}
        if sd:
            kwargs["start_date"] = sd
        if ed:
            kwargs["end_date"] = ed
        return self._query(
            bs.query_deposit_rate_data, **kwargs,
            label="存款利率",
        )

    def get_loan_rate(
        self,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
    ) -> pd.DataFrame:
        """获取贷款利率。"""
        sd = str(ensure_date(start_date)) if start_date else None
        ed = str(ensure_date(end_date)) if end_date else None
        kwargs = {}
        if sd:
            kwargs["start_date"] = sd
        if ed:
            kwargs["end_date"] = ed
        return self._query(
            bs.query_loan_rate_data, **kwargs,
            label="贷款利率",
        )

    def get_required_reserve_ratio(
        self,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
    ) -> pd.DataFrame:
        """获取存款准备金率。"""
        sd = str(ensure_date(start_date)) if start_date else None
        ed = str(ensure_date(end_date)) if end_date else None
        kwargs = {}
        if sd:
            kwargs["start_date"] = sd
        if ed:
            kwargs["end_date"] = ed
        return self._query(
            bs.query_required_reserve_ratio_data, **kwargs,
            label="存款准备金率",
        )

    def get_money_supply_month(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """获取月度货币供应量。

        Parameters
        ----------
        start_date, end_date : str
            格式 "YYYY-MM"，如 "2020-01"。
        """
        kwargs = {}
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        return self._query(
            bs.query_money_supply_data_month, **kwargs,
            label="货币供应量(月)",
        )

    def get_money_supply_year(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """获取年度货币供应量（年底余额）。

        Parameters
        ----------
        start_date, end_date : str
            格式 "YYYY"，如 "2020"。
        """
        kwargs = {}
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        return self._query(
            bs.query_money_supply_data_year, **kwargs,
            label="货币供应量(年)",
        )


# 注册到工厂
DataSourceFactory.register("baostock", BaoStockDataSource)