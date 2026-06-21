"""
TickFlow 数据源适配器。

TickFlow 免费层（无需注册）提供：
- 历史日K线（1d/1w/1M）
- 标的元数据（total_shares, float_shares, listing_date）
- Shenwan 行业分类 universe（SW1/SW2/SW3）
- 全市场股票 universe（CN_Equity_A）

API 文档: https://tickflow.org/docs

不支持：分钟K线、实时行情、财务数据、成分股查询。
免费层 IP 限速 60 req/min，每请求最多 1000 个标的。
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from stockquant.data.data_source import (
    BaseDataSource,
    DataSourceFactory,
    standardize_daily,
    standardize_index,
)
from stockquant.utils.helpers import (
    normalize_stock_code, ensure_date, split_list,
    RateLimiter, call_with_retries,
)
from stockquant.utils.logger import get_logger

logger = get_logger("data.tickflow")

STOCK_INFO_COLS = [
    "code", "name", "industry", "sector", "market",
    "list_date", "total_shares", "float_shares", "total_cap", "float_cap",
]


def _to_tf_symbol(code: str) -> str:
    """纯6位数字 → TickFlow 格式 ``600000.SH``"""
    code = normalize_stock_code(code)
    suffix = "SH" if code.startswith(("6", "9")) else "SZ"
    return f"{code}.{suffix}"


def _from_tf_symbol(tf_sym: str) -> str:
    """TickFlow 格式 ``600000.SH`` → 纯6位数字"""
    return normalize_stock_code(tf_sym)


def _date_to_ms(d: dt.date) -> int:
    """date → unix 毫秒时间戳（本地时区 00:00）。"""
    return int(dt.datetime.combine(d, dt.time.min).timestamp() * 1000)


class TickFlowDataSource(BaseDataSource):
    """基于 TickFlow 免费层的数据源实现。

    提供 Shenwan 行业分类 + 股本数据 + 日线 K 线。
    不提供 index_constituents 和 finance_data。
    """

    def __init__(self) -> None:
        self._client = None
        self._industry_map: dict[str, str] | None = None
        # TickFlow 免费层: 60 req/min → 1 req/s, burst=60 支持冷启动快速拉取
        self._rate_limiter = RateLimiter(rate=1.0, burst=60)

    @property
    def client(self):
        """延迟初始化 TickFlow 客户端（避免 import 时打印 free tier 横幅）。"""
        if self._client is None:
            import tickflow as tf
            self._client = tf.TickFlow.free()
        return self._client

    # ------------------------------------------------------------------
    def get_stock_list(self) -> pd.DataFrame:
        """从 CN_Equity_A universe + instruments.batch 获取全量 A 股列表。"""
        logger.info("获取全量 A 股列表...")
        self._throttle()
        universe = self.client.universes.get("CN_Equity_A")
        symbols = universe.get("symbols", [])
        if not symbols:
            return pd.DataFrame(columns=["code", "name"])

        rows = []
        for chunk in split_list(symbols, 1000):
            self._throttle()
            try:
                insts = self.client.instruments.batch(chunk)
            except Exception:
                continue
            for inst in insts:
                tf_sym = inst.get("symbol", "")
                rows.append({
                    "code": _from_tf_symbol(tf_sym),
                    "name": inst.get("name", ""),
                })

        logger.info(f"  获取 {len(rows)} 只股票")
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _load_industry_map(self) -> dict[str, str]:
        """从所有 SW1 universe 构建 {6位code: SW1行业名} 映射。"""
        if self._industry_map is not None:
            return self._industry_map

        logger.info("加载 Shenwan SW1 行业分类...")
        self._throttle()
        univ_list = self.client.universes.list()
        sw1_ids = [u["id"] for u in univ_list if "SW1" in u.get("id", "")]

        # 使用 batch 接口避免逐个查询触发频率限制
        industry_map: dict[str, str] = {}
        for chunk_ids in split_list(sw1_ids, 50):
            self._throttle()
            try:
                batch_result = self.client.universes.batch(chunk_ids)
            except Exception:
                continue
            for uid, univ in batch_result.items():
                name = univ.get("name", "")
                industry_name = name.replace("SW1", "", 1) if name.startswith("SW1") else name
                for tf_sym in univ.get("symbols", []):
                    code = _from_tf_symbol(tf_sym)
                    industry_map[code] = industry_name

        self._industry_map = industry_map
        logger.info(f"  {len(sw1_ids)} 个 SW1 universe, {len(industry_map)} 只股票有行业")
        return industry_map

    # ------------------------------------------------------------------
    def get_stock_info(self) -> pd.DataFrame:
        """获取全市场股票基本信息：Shenwan SW1 行业 + 股本 + 上市日期。

        市值字段（total_cap/float_cap）留空，需由日线价格×股本另行计算。
        """
        industry_map = self._load_industry_map()

        logger.info("获取标的信息...")
        self._throttle()
        universe = self.client.universes.get("CN_Equity_A")
        all_symbols = universe.get("symbols", [])

        rows = []
        for chunk in split_list(all_symbols, 1000):
            self._throttle()
            try:
                insts = self.client.instruments.batch(chunk)
            except Exception:
                continue
            for inst in insts:
                tf_sym = inst.get("symbol", "")
                code = _from_tf_symbol(tf_sym)
                ext = inst.get("ext") or {}
                rows.append({
                    "code": code,
                    "name": inst.get("name", ""),
                    "industry": industry_map.get(code, ""),
                    "sector": "",
                    "market": "",
                    "list_date": ext.get("listing_date") or None,
                    "total_shares": ext.get("total_shares"),
                    "float_shares": ext.get("float_shares"),
                    "total_cap": None,
                    "float_cap": None,
                })

        logger.info(f"  获取 {len(rows)} 只股票信息")
        return pd.DataFrame(rows, columns=STOCK_INFO_COLS)

    # ------------------------------------------------------------------
    def get_daily_bars(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
        adjust: str = "hfq",
    ) -> pd.DataFrame:
        """获取日线行情。

        TickFlow adjust: qfq→forward, hfq→backward, none→none。
        """
        tf_sym = _to_tf_symbol(code)
        sd = ensure_date(start_date)
        ed = ensure_date(end_date)

        days = max((ed - sd).days + 1, 1)
        count = min(days * 2, 10000)

        adjust_map = {"qfq": "forward", "hfq": "backward", "none": "none"}
        tf_adjust = adjust_map.get(adjust, "backward")

        def _fetch() -> pd.DataFrame:
            self._throttle()
            return self.client.klines.get(
                tf_sym,
                period="1d",
                count=count,
                start_time=_date_to_ms(sd),
                end_time=_date_to_ms(ed),
                adjust=tf_adjust,
                as_dataframe=True,
            )

        try:
            df = call_with_retries(_fetch, attempts=3, label=f"TickFlow K线 {code}")
        except Exception as e:
            logger.warning(f"TickFlow K线获取失败 {code}: {e}")
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        if "trade_date" in df.columns:
            df["date"] = pd.to_datetime(df["trade_date"])
        elif "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")

        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col not in df.columns:
                df[col] = 0.0

        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= pd.Timestamp(sd)) & (df["date"] <= pd.Timestamp(ed))]
        if df.empty:
            return df
        return standardize_daily(df, normalize_stock_code(code))

    # ------------------------------------------------------------------
    def get_index_daily(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> pd.DataFrame:
        """获取指数日线。常用指数通过 .SH/.SZ 后缀查询。"""
        tf_sym = _to_tf_symbol(code)
        sd = ensure_date(start_date)
        ed = ensure_date(end_date)

        days = max((ed - sd).days + 1, 1)
        count = min(days * 2, 10000)

        def _fetch() -> pd.DataFrame:
            self._throttle()
            return self.client.klines.get(
                tf_sym,
                period="1d",
                count=count,
                start_time=_date_to_ms(sd),
                end_time=_date_to_ms(ed),
                as_dataframe=True,
            )

        try:
            df = call_with_retries(_fetch, attempts=3, label=f"TickFlow 指数K线 {code}")
        except Exception as e:
            logger.warning(f"TickFlow 指数K线获取失败 {code}: {e}")
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        if "trade_date" in df.columns:
            df["date"] = pd.to_datetime(df["trade_date"])
        elif "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")

        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col not in df.columns:
                df[col] = 0.0

        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= pd.Timestamp(sd)) & (df["date"] <= pd.Timestamp(ed))]
        if df.empty:
            return df
        return standardize_index(df, normalize_stock_code(code))

    # ------------------------------------------------------------------
    def get_index_constituents(self, index_code: str) -> list[str]:
        raise NotImplementedError(
            "TickFlow 暂不支持 get_index_constituents，请切换 AkShare 或 Tushare。"
        )

    # ------------------------------------------------------------------
    def get_finance_data(self, code: str) -> pd.DataFrame:
        raise NotImplementedError(
            "TickFlow 暂不支持 get_finance_data，请切换 AkShare 或 Tushare。"
        )

    # ------------------------------------------------------------------
    def get_trade_dates(
        self,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> list[str]:
        """TickFlow 免费层不提供交易日历接口，使用 pandas 默认。"""
        sd = ensure_date(start_date)
        ed = ensure_date(end_date)
        dates = pd.bdate_range(start=sd, end=ed)
        return [d.strftime("%Y-%m-%d") for d in dates]


# 注册到工厂
DataSourceFactory.register("tickflow", TickFlowDataSource)
