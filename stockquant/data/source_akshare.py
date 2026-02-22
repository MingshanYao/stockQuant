"""
AkShare 数据源适配器。
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import akshare as ak
import pandas as pd
import numpy as np

from stockquant.data.data_source import BaseDataSource, DataSourceFactory
from stockquant.utils.concurrent import parallel_fetch_serial_consume
from stockquant.utils.helpers import normalize_stock_code, get_market_prefix, ensure_date
from stockquant.utils.logger import get_logger

logger = get_logger("data.akshare")


class AkShareDataSource(BaseDataSource):
    """基于 AkShare 的数据源实现。"""

    # ------------------------------------------------------------------
    # 股票列表
    # ------------------------------------------------------------------
    def get_stock_list(self) -> pd.DataFrame:
        logger.info("通过 AkShare 获取 A 股股票列表")
        df = ak.stock_info_a_code_name()
        df.columns = ["code", "name"]
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 日线行情
    # ------------------------------------------------------------------
    def get_daily_bars(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
        adjust: str = "hfq",
    ) -> pd.DataFrame:
        code = normalize_stock_code(code)
        start_str = str(ensure_date(start_date)).replace("-", "")
        end_str = str(ensure_date(end_date)).replace("-", "")

        logger.debug(f"AkShare 日线: {code} [{start_str} ~ {end_str}] adjust={adjust}")

        adjust_map = {"qfq": "qfq", "hfq": "hfq", "none": ""}
        ak_adjust = adjust_map.get(adjust, "hfq")

        # 如果不需要复权，直接请求一次并返回
        try:
            if ak_adjust == "":
                raw = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_str,
                    end_date=end_str,
                )
                if raw is None or raw.empty:
                    return pd.DataFrame()
                return self._standardize_daily(raw, code)

            # 需要复权时，同时拉取未复权与复权数据，计算复权因子以调整成交量
            raw = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust="",
            )
            adj = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust=ak_adjust,
            )
        except Exception as e:
            logger.error(f"获取 {code} 日线数据失败: {e}")
            return pd.DataFrame()

        if adj is None or adj.empty:
            return pd.DataFrame()

        # 标准化两份数据，便于按日期对齐
        df_raw = self._standardize_daily(raw, code) if (raw is not None and not raw.empty) else pd.DataFrame()
        df_adj = self._standardize_daily(adj, code)

        # 若没有未复权数据，则直接返回复权数据（无法调整成交量）
        if df_raw.empty:
            return df_adj

        # 将成交量复权的逻辑交由单独方法处理
        result = self._compute_adjusted_volume(code, df_raw, df_adj)
        return result

    # ------------------------------------------------------------------
    # 指数日线
    # ------------------------------------------------------------------
    def get_index_daily(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> pd.DataFrame:
        code = normalize_stock_code(code)
        start_str = str(ensure_date(start_date)).replace("-", "")
        end_str = str(ensure_date(end_date)).replace("-", "")

        logger.debug(f"AkShare 指数日线: {code} [{start_str} ~ {end_str}]")

        # 优先使用 index_zh_a_hist（覆盖所有指数，含中证2000等）
        try:
            df = ak.index_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_str, end_date=end_str,
            )
            if not df.empty:
                return self._standardize_index(df, code)
        except Exception as e:
            logger.debug(f"index_zh_a_hist({code}) 失败，尝试备用接口: {e}")

        # 备用：stock_zh_index_daily（部分老指数仅此接口有数据）
        try:
            df = ak.stock_zh_index_daily(symbol=f"{get_market_prefix(code)}{code}")
            if not df.empty:
                # 统一 date 列为 datetime 再过滤
                df["date"] = pd.to_datetime(df["date"])
                sd = pd.Timestamp(ensure_date(start_date))
                ed = pd.Timestamp(ensure_date(end_date))
                df = df[(df["date"] >= sd) & (df["date"] <= ed)]
                return self._standardize_index(df, code)
        except Exception as e:
            logger.error(f"获取指数 {code} 日线失败: {e}")

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 基本财务
    # ------------------------------------------------------------------
    def get_finance_data(self, code: str) -> pd.DataFrame:
        code = normalize_stock_code(code)
        logger.debug(f"AkShare 财务数据: {code}")
        try:
            df = ak.stock_financial_abstract_ths(symbol=code)
        except Exception as e:
            logger.warning(f"获取 {code} 财务数据失败: {e}")
            return pd.DataFrame()
        return df

    # ------------------------------------------------------------------
    # 交易日历
    # ------------------------------------------------------------------
    def get_trade_dates(
        self,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> list[str]:
        try:
            df = ak.tool_trade_date_hist_sina()
            dates = pd.to_datetime(df["trade_date"])
            sd = pd.Timestamp(ensure_date(start_date))
            ed = pd.Timestamp(ensure_date(end_date))
            filtered = dates[(dates >= sd) & (dates <= ed)]
            return [d.strftime("%Y-%m-%d") for d in filtered]
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return []

    # ------------------------------------------------------------------
    # 股票基本信息（行业/市值等）
    # ------------------------------------------------------------------
    def get_stock_info(self) -> pd.DataFrame:
        """获取全市场股票基本信息（代码/名称/行业/市值等）。

        数据源:
        - ``stock_zh_a_spot_em``  — 全市场实时行情快照，提供市值/股本
        - ``stock_board_industry_name_em`` + ``stock_board_industry_cons_em``
          — 行业板块成分股，用于反查每只股票所属行业

        Returns
        -------
        DataFrame
            标准列: code, name, industry, sector, market,
            list_date, total_shares, float_shares, total_cap, float_cap
        """
        logger.info("通过 AkShare 获取全市场股票基本信息")

        # ---- 1. 全市场实时快照 ----
        try:
            spot = ak.stock_zh_a_spot_em()
        except Exception as e:
            logger.error(f"获取全市场实时行情失败: {e}")
            return pd.DataFrame()

        df = pd.DataFrame()
        df["code"] = spot["代码"].astype(str).str.zfill(6)
        df["name"] = spot["名称"]
        df["total_cap"] = pd.to_numeric(spot.get("总市值"), errors="coerce")
        df["float_cap"] = pd.to_numeric(spot.get("流通市值"), errors="coerce")

        # ---- 2. 推断 market ----
        df["market"] = df["code"].apply(self._infer_market)

        # ---- 3. 行业映射 ----
        industry_map = self._get_industry_map()
        df["industry"] = df["code"].map(industry_map).fillna("其他")

        # sector 暂用 market 字段（后续可扩展为申万一级行业）
        df["sector"] = df["market"]

        # 当前实时快照不包含股本/上市日期，留空后续可补充
        for col in ("list_date", "total_shares", "float_shares"):
            if col not in df.columns:
                df[col] = None

        # ---- 4. 列顺序 ----
        standard_cols = [
            "code", "name", "industry", "sector", "market",
            "list_date", "total_shares", "float_shares",
            "total_cap", "float_cap",
        ]
        df = df[[c for c in standard_cols if c in df.columns]]
        logger.info(f"获取股票基本信息完成: {len(df)} 只")
        return df

    @staticmethod
    def _infer_market(code: str) -> str:
        """根据代码前缀推断所属市场。"""
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

    @staticmethod
    def _get_industry_map() -> dict[str, str]:
        """构建 {股票代码: 行业名称} 映射表。

        遍历东方财富行业板块列表，并发拉取成分股，建立反向映射。
        跳过重名板块（如 Ⅱ/Ⅲ 同名的保留第一个）。
        """
        logger.info("构建行业映射表（stock_board_industry）")
        code_to_industry: dict[str, str] = {}

        try:
            boards = ak.stock_board_industry_name_em()
        except Exception as e:
            logger.error(f"获取行业板块列表失败: {e}")
            return code_to_industry

        # 过滤重复板块（Ⅱ/Ⅲ 后缀视为同一行业，保留一个即可）
        seen_names: set[str] = set()
        unique_boards: list[str] = []
        for _, row in boards.iterrows():
            name = row["板块名称"]
            base_name = name.rstrip("ⅡⅢⅣ").strip()
            if base_name not in seen_names:
                seen_names.add(base_name)
                unique_boards.append(name)

        logger.info(f"共 {len(unique_boards)} 个行业板块需要遍历")

        def _fetch(board_name: str) -> pd.DataFrame:
            return ak.stock_board_industry_cons_em(symbol=board_name)

        def _consume(
            board_name: str,
            df: pd.DataFrame | None,
            err: Exception | None,
        ) -> None:
            if err is not None:
                logger.debug(f"获取板块 {board_name} 成分股失败: {err}")
                return
            if df is None or df.empty:
                return
            clean_name = board_name.rstrip("ⅡⅢⅣ").strip()
            for code in df["代码"].astype(str).str.zfill(6):
                if code not in code_to_industry:
                    code_to_industry[code] = clean_name

        _, elapsed = parallel_fetch_serial_consume(
            items=unique_boards,
            fetch_fn=_fetch,
            consume_fn=_consume,
            max_workers=16,
            progress_interval=50,
            label="行业映射",
        )

        logger.info(
            f"行业映射构建完成: {len(code_to_industry)} 只股票, "
            f"耗时 {elapsed:.1f}s"
        )
        return code_to_industry

    # ------------------------------------------------------------------
    # 内部：列名标准化
    # ------------------------------------------------------------------
    @staticmethod
    def _standardize_daily(df: pd.DataFrame, code: str) -> pd.DataFrame:
        col_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
        }
        df = df.rename(columns=col_map)
        df["code"] = code
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        # 只保留数据库 daily_bars 表中定义的列，丢弃多余列（如 股票代码、振幅）
        standard_cols = [
            "code", "date", "open", "high", "low", "close",
            "volume", "amount", "turnover", "pct_change", "change",
        ]
        df = df[[c for c in standard_cols if c in df.columns]]
        return df

    @staticmethod
    def _standardize_index(df: pd.DataFrame, code: str) -> pd.DataFrame:
        """标准化指数日线 DataFrame，输出列与 index_daily 表对齐。"""
        df = df.copy()

        # index_zh_a_hist 返回中文列名，stock_zh_index_daily 返回英文列名
        col_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
        df = df.rename(columns=col_map)

        df["code"] = code
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # 只保留 index_daily 表定义的列
        standard_cols = ["code", "date", "open", "high", "low", "close", "volume", "amount"]
        df = df[[c for c in standard_cols if c in df.columns]]
        return df

    def _compute_adjusted_volume(
        self,
        code: str,
        df_raw: pd.DataFrame,
        df_adj: pd.DataFrame,
    ) -> pd.DataFrame:
        """基于未复权与复权收盘价计算复权因子，更新成交量。

        计算逻辑::

            adj_factor = close_adj / close_raw
            adjusted_volume = volume_raw / adj_factor

        本方法 **不做任何回退或填充**，所有异常数据均通过 logger
        记录 code 与受影响日期/行数，由上层决定是否剔除。

        Parameters
        ----------
        code : str
            股票代码，仅用于日志标识。
        df_raw : DataFrame
            未复权日线（至少含 date, close, volume）。
        df_adj : DataFrame
            复权日线（标准列）。

        Returns
        -------
        DataFrame
            与 df_adj 结构一致，volume 列已替换为复权成交量。
        """
        # ---- 0. 空数据检查 ----
        if df_raw is None or df_raw.empty:
            logger.warning(f"[{code}] 未复权数据为空，无法计算复权成交量")
            return df_adj

        # ---- 1. 按日期合并 ----
        merged = pd.merge(
            df_adj,
            df_raw[["date", "close", "volume"]].rename(
                columns={"close": "close_raw", "volume": "volume_raw"},
            ),
            on="date",
            how="left",
        )

        # 检查：合并后是否有日期未匹配（df_adj 有但 df_raw 无）
        unmatched = merged["close_raw"].isna()
        if unmatched.any():
            dates = merged.loc[unmatched, "date"].dt.strftime("%Y-%m-%d").tolist()
            logger.warning(
                f"[{code}] {len(dates)} 个交易日缺少未复权数据，"
                f"日期: {dates[:10]}{'...' if len(dates) > 10 else ''}"
            )

        # ---- 2. 数值类型强制转换 ----
        for col in ("close", "close_raw", "volume_raw"):
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

        # 检查：转换后产生的新 NaN（原始值为非数值字符串等）
        for col in ("close", "close_raw", "volume_raw"):
            nan_count = int(merged[col].isna().sum())
            if nan_count > 0:
                logger.warning(f"[{code}] 列 {col} 存在 {nan_count} 个非数值/NaN")

        # ---- 3. 异常值检查 ----
        # close_raw <= 0（含 0）→ 无法计算因子
        bad_close_raw = merged["close_raw"].le(0) & merged["close_raw"].notna()
        if bad_close_raw.any():
            dates = merged.loc[bad_close_raw, "date"].dt.strftime("%Y-%m-%d").tolist()
            logger.warning(
                f"[{code}] close_raw <= 0 共 {len(dates)} 行，"
                f"日期: {dates[:10]}{'...' if len(dates) > 10 else ''}"
            )
            merged.loc[bad_close_raw, "close_raw"] = np.nan

        # close_adj <= 0
        bad_close_adj = merged["close"].le(0) & merged["close"].notna()
        if bad_close_adj.any():
            dates = merged.loc[bad_close_adj, "date"].dt.strftime("%Y-%m-%d").tolist()
            logger.warning(
                f"[{code}] close_adj <= 0 共 {len(dates)} 行，"
                f"日期: {dates[:10]}{'...' if len(dates) > 10 else ''}"
            )

        # volume_raw < 0
        bad_vol = merged["volume_raw"].lt(0) & merged["volume_raw"].notna()
        if bad_vol.any():
            dates = merged.loc[bad_vol, "date"].dt.strftime("%Y-%m-%d").tolist()
            logger.warning(
                f"[{code}] volume_raw < 0 共 {len(dates)} 行，"
                f"日期: {dates[:10]}{'...' if len(dates) > 10 else ''}"
            )

        # ---- 4. 计算复权因子 ----
        adj_factor = merged["close"] / merged["close_raw"]  # close_raw <= 0 已置 NaN

        # 检查：因子异常（inf / 极端值）
        inf_mask = np.isinf(adj_factor)
        if inf_mask.any():
            dates = merged.loc[inf_mask, "date"].dt.strftime("%Y-%m-%d").tolist()
            logger.warning(
                f"[{code}] adj_factor 出现 inf 共 {len(dates)} 行，"
                f"日期: {dates[:10]}{'...' if len(dates) > 10 else ''}"
            )
            adj_factor = adj_factor.replace([np.inf, -np.inf], np.nan)

        factor_nan = adj_factor.isna()
        if factor_nan.any():
            logger.warning(f"[{code}] adj_factor 为 NaN 共 {int(factor_nan.sum())} 行")

        # ---- 5. 计算复权成交量 ----
        merged["volume"] = merged["volume_raw"] / adj_factor

        # 检查：最终 volume 异常
        vol = merged["volume"]
        vol_na = vol.isna()
        vol_neg = vol.lt(0) & vol.notna()
        vol_inf = np.isinf(vol)

        anomaly_parts: list[str] = []
        if vol_na.any():
            anomaly_parts.append(f"NaN={int(vol_na.sum())}")
        if vol_neg.any():
            anomaly_parts.append(f"负值={int(vol_neg.sum())}")
        if vol_inf.any():
            anomaly_parts.append(f"inf={int(vol_inf.sum())}")
            merged.loc[vol_inf, "volume"] = np.nan

        if anomaly_parts:
            logger.warning(
                f"[{code}] 复权成交量异常汇总: {', '.join(anomaly_parts)}，"
                f"请在上层剔除或处理"
            )

        # ---- 6. 输出标准列 ----
        standard_cols = [
            "code", "date", "open", "high", "low", "close",
            "volume", "amount", "turnover", "pct_change", "change",
        ]
        return merged[[c for c in standard_cols if c in merged.columns]]


# 注册到工厂
DataSourceFactory.register("akshare", AkShareDataSource)
