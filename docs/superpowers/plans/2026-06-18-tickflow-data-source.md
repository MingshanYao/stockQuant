# TickFlow 数据源集成实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 TickFlow 作为新数据源接入框架，补全 Shenwan 行业分类 + 股本数据，与现有 itick/tushare/akshare 并存。

**Architecture:** 新建 `source_tickflow.py`，使用 `tickflow` SDK 的 `TickFlow.free()` 免费客户端。核心能力：(1) SW1 行业分类通过 `universes.get("CN_Equity_SW1_XXX")` 获取；(2) 股本/上市日期通过 `instruments.batch()` 获取；(3) K线通过 `klines.get(as_dataframe=True)` 获取。现有 `source_itick.py` 保持不变（它是有效的适配器，仅受本环境防火墙限制）。

**Tech Stack:** tickflow SDK (free tier, `https://free-api.tickflow.org`), pandas

**限制:** 免费层 IP 限速 60 req/min，仅提供日线 K 线（1d/1w/1M），无分钟线和实时行情。

---

### Task 1: TickFlow 符号格式转换工具

**Files:**
- Create: `stockquant/data/source_tickflow.py`

- [ ] **Step 1: 创建文件，编写符号转换函数和类骨架**

```python
"""
TickFlow 数据源适配器。

TickFlow 免费层（无需注册）提供：
- 历史日K线（1d/1w/1M）
- 标的元数据（total_shares, float_shares, listing_date）
- Shenwan 行业分类 universe（SW1/SW2/SW3）
- 全市场股票 universe（CN_Equity_A）

不支持：分钟K线、实时行情、财务数据、成分股查询。

符号格式：内部 6 位数字 ←→ TickFlow ``code.SH`` / ``code.SZ``
"""

from __future__ import annotations

import datetime as dt
import time as _time
from typing import Any

import pandas as pd

from stockquant.data.data_source import (
    BaseDataSource,
    DataSourceFactory,
    standardize_daily,
    standardize_index,
)
from stockquant.utils.helpers import normalize_stock_code, ensure_date, split_list
from stockquant.utils.logger import get_logger

logger = get_logger("data.tickflow")

# stock_info 表标准列
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
    """date → unix 毫秒时间戳（Asia/Shanghai 00:00）"""
    return int(dt.datetime.combine(d, dt.time.min, tzinfo=dt.timezone.utc).timestamp() * 1000)


class TickFlowDataSource(BaseDataSource):
    """基于 TickFlow 免费层的数据源实现。

    提供 Shenwan 行业分类 + 股本数据 + 日线 K 线。
    不提供 index_constituents 和 finance_data。
    """

    def __init__(self) -> None:
        self._client = None  # 延迟初始化，避免 import 时触发 free tier 打印
        self._industry_map: dict[str, str] | None = None

    @property
    def client(self):
        if self._client is None:
            import tickflow as tf
            self._client = tf.TickFlow.free()
        return self._client
```

- [ ] **Step 2: 验证符号转换**

Run: `.venv/bin/python -c "
from stockquant.data.source_tickflow import _to_tf_symbol, _from_tf_symbol, _date_to_ms
assert _to_tf_symbol('000001') == '000001.SZ'
assert _to_tf_symbol('600000') == '600000.SH'
assert _from_tf_symbol('000001.SZ') == '000001'
assert _from_tf_symbol('600000.SH') == '600000'
import datetime as dt
ms = _date_to_ms(dt.date(2025, 1, 1))
assert ms == 1735660800000
print('OK: all conversion tests passed')
"`

Expected: `OK: all conversion tests passed`

- [ ] **Step 5: Commit**

```bash
git add stockquant/data/source_tickflow.py
git commit -m "feat: add TickFlow data source skeleton with symbol conversion"
```

---

### Task 2: 实现 get_stock_list — 从 CN_Equity_A universe 获取全量股票

**Files:**
- Modify: `stockquant/data/source_tickflow.py`

- [ ] **Step 1: 实现 get_stock_list 方法（追加到 TickFlowDataSource 类中）**

```python
    def get_stock_list(self) -> pd.DataFrame:
        """从 CN_Equity_A universe + instruments.batch 获取全量 A 股列表。"""
        logger.info("获取全量 A 股列表...")
        universe = self.client.universes.get("CN_Equity_A")
        symbols = universe.get("symbols", [])
        if not symbols:
            return pd.DataFrame(columns=["code", "name"])

        # 批量获取名称（每批 1000）
        rows = []
        for chunk in split_list(symbols, 1000):
            try:
                insts = self.client.instruments.batch(chunk)
            except Exception:
                continue
            for inst in insts:
                tf_sym = inst.get("symbol", "")
                code = _from_tf_symbol(tf_sym)
                name = inst.get("name", "")
                rows.append({"code": code, "name": name})

        logger.info(f"  获取 {len(rows)} 只股票")
        return pd.DataFrame(rows)
```

- [ ] **Step 2: 运行验证**

Run: `.venv/bin/python -c "
from stockquant.data.source_tickflow import TickFlowDataSource
ds = TickFlowDataSource()
df = ds.get_stock_list()
print(f'Stock count: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(df.head())
assert len(df) > 4000
assert 'code' in df.columns and 'name' in df.columns
"`

Expected: `Stock count: >4000` with code/name columns

- [ ] **Step 3: Commit**

```bash
git add stockquant/data/source_tickflow.py
git commit -m "feat: implement get_stock_list via TickFlow CN_Equity_A universe"
```

---

### Task 3: 实现 get_stock_info — Shenwan 行业分类 + 股本

**Files:**
- Modify: `stockquant/data/source_tickflow.py`

- [ ] **Step 1: 添加行业分类加载方法**

```python
    def _load_industry_map(self) -> dict[str, str]:
        """从所有 SW1 universe 构建 {6位code: SW1行业名} 映射。

        缓存在 self._industry_map，后续 get_stock_info 复用。
        """
        if self._industry_map is not None:
            return self._industry_map

        logger.info("加载 Shenwan SW1 行业分类...")
        univ_list = self.client.universes.list()
        sw1_ids = [u["id"] for u in univ_list if "SW1" in u.get("id", "")]

        industry_map: dict[str, str] = {}
        for uid in sw1_ids:
            try:
                univ = self.client.universes.get(uid)
            except Exception:
                continue
            name = univ.get("name", "")
            industry_name = name.replace("SW1", "") if name.startswith("SW1") else name
            for tf_sym in univ.get("symbols", []):
                code = _from_tf_symbol(tf_sym)
                industry_map[code] = industry_name

        self._industry_map = industry_map
        logger.info(f"  {len(sw1_ids)} 个 SW1 universe, {len(industry_map)} 只股票有行业")
        return industry_map
```

- [ ] **Step 2: 实现 get_stock_info 方法**

```python
    def get_stock_info(self) -> pd.DataFrame:
        """获取全市场股票基本信息：行业（Shenwan SW1） + 股本 + 上市日期。

        分两步：
        1. SW1 universe → 行业分类
        2. instruments.batch → total_shares, float_shares, listing_date
        """
        industry_map = self._load_industry_map()

        logger.info("获取标的信息...")
        universe = self.client.universes.get("CN_Equity_A")
        all_symbols = universe.get("symbols", [])

        rows = []
        for chunk in split_list(all_symbols, 1000):
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
        df = pd.DataFrame(rows, columns=STOCK_INFO_COLS)
        # 计算市值 = close × shares（需日线数据补充）
        # 市值计算留待后续数据管线处理
        return df
```

- [ ] **Step 3: 运行验证（仅拉取前几只股票以节约 API 配额）**

Run: `.venv/bin/python -c "
from stockquant.data.source_tickflow import TickFlowDataSource
ds = TickFlowDataSource()

# 先测行业 map
imap = ds._load_industry_map()
print(f'Industry map: {len(imap)} stocks')
sample = list(imap.items())[:5]
for code, ind in sample:
    print(f'  {code}: {ind}')
# 验证常见行业存在
industries = set(imap.values())
print(f'Unique industries: {len(industries)}')
assert len(industries) >= 25, f'Expected >=25 SW1 industries, got {len(industries)}'
assert len(imap) > 3000, f'Expected >3000 stocks with industry, got {len(imap)}'
print('OK: industry map valid')
"`

Expected: `>=28 unique industries, >3000 stocks with industry`

- [ ] **Step 4: Commit**

```bash
git add stockquant/data/source_tickflow.py
git commit -m "feat: implement get_stock_info with Shenwan SW1 industry via TickFlow"
```

---

### Task 4: 实现 get_daily_bars — 通过 TickFlow K线接口获取日线

**Files:**
- Modify: `stockquant/data/source_tickflow.py`

- [ ] **Step 1: 实现 get_daily_bars 方法**

```python
    def get_daily_bars(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
        adjust: str = "hfq",
    ) -> pd.DataFrame:
        """获取日线行情。

        TickFlow adjust 参数映射：qfq→"forward", hfq→"backward", none→"none"
        """
        tf_sym = _to_tf_symbol(code)
        sd = ensure_date(start_date)
        ed = ensure_date(end_date)

        # 估算 K 线数量（实际由 API 按时间范围返回）
        days = max((ed - sd).days + 1, 1)
        count = min(days * 2, 10000)  # 留余量应对非交易日

        adjust_map = {"qfq": "forward", "hfq": "backward", "none": "none"}
        tf_adjust = adjust_map.get(adjust, "backward")

        try:
            df = self.client.klines.get(
                tf_sym,
                period="1d",
                count=count,
                start_time=_date_to_ms(sd),
                end_time=_date_to_ms(ed),
                adjust=tf_adjust,
                as_dataframe=True,
            )
        except Exception as e:
            logger.warning(f"TickFlow K线获取失败 {code}: {e}")
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # TickFlow DataFrame 列: symbol, name, timestamp, trade_date,
        #   trade_time, open, high, low, close, volume, amount
        if "trade_date" in df.columns:
            df["date"] = pd.to_datetime(df["trade_date"])
        elif "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")

        # 映射到标准列名
        df = df.rename(columns={
            "trade_date": "date",
        })

        # 只保留需要的列
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col not in df.columns:
                df[col] = 0.0

        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= pd.Timestamp(sd)) & (df["date"] <= pd.Timestamp(ed))]
        if df.empty:
            return df
        return standardize_daily(df, normalize_stock_code(code))
```

- [ ] **Step 2: 运行验证**

Run: `.venv/bin/python -c "
from stockquant.data.source_tickflow import TickFlowDataSource
ds = TickFlowDataSource()
df = ds.get_daily_bars('000001', '2020-01-01', '2020-01-31', adjust='hfq')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(df.head(3))
assert not df.empty
assert list(df.columns) == ['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover', 'pct_change', 'change']
print('OK: daily_bars valid')
"`

Expected: non-empty DataFrame with standard columns

- [ ] **Step 3: Commit**

```bash
git add stockquant/data/source_tickflow.py
git commit -m "feat: implement get_daily_bars via TickFlow klines API"
```

---

### Task 5: 实现 get_index_daily 和其他接口

**Files:**
- Modify: `stockquant/data/source_tickflow.py`

- [ ] **Step 1: 实现 get_index_daily**

```python
    def get_index_daily(
        self,
        code: str,
        start_date: str | dt.date,
        end_date: str | dt.date,
    ) -> pd.DataFrame:
        """获取指数日线。

        常用指数在 TickFlow 中需要 .SH/.SZ 后缀:
        000001.SH(上证), 000300.SH(沪深300), 000905.SH(中证500),
        399001.SZ(深证), 399005.SZ(中小板), 399006.SZ(创业板)
        """
        tf_sym = _to_tf_symbol(code)
        sd = ensure_date(start_date)
        ed = ensure_date(end_date)

        days = max((ed - sd).days + 1, 1)
        count = min(days * 2, 10000)

        try:
            df = self.client.klines.get(
                tf_sym,
                period="1d",
                count=count,
                start_time=_date_to_ms(sd),
                end_time=_date_to_ms(ed),
                as_dataframe=True,
            )
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
```

- [ ] **Step 2: 实现不支持的方法 + get_trade_dates**

```python
    def get_index_constituents(self, index_code: str) -> list[str]:
        raise NotImplementedError(
            "TickFlow 暂不支持 get_index_constituents，请切换 AkShare 或 Tushare。"
        )

    def get_finance_data(self, code: str) -> pd.DataFrame:
        raise NotImplementedError(
            "TickFlow 暂不支持 get_finance_data，请切换 AkShare 或 Tushare。"
        )

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
```

- [ ] **Step 3: 注册到工厂**

```python
# 注册到工厂（文件末尾）
DataSourceFactory.register("tickflow", TickFlowDataSource)
```

- [ ] **Step 4: 运行验证**

Run: `.venv/bin/python -c "
from stockquant.data.source_tickflow import TickFlowDataSource
ds = TickFlowDataSource()
df = ds.get_index_daily('000300', '2020-01-01', '2020-01-31')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
assert not df.empty
assert list(df.columns) == ['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']
print('OK: index_daily valid')

# 验证工厂注册
from stockquant.data.data_source import DataSourceFactory
ds2 = DataSourceFactory.create('tickflow')
assert isinstance(ds2, TickFlowDataSource)
print('OK: factory registration valid')

# 验证不支持方法正确 raise
try:
    ds.get_index_constituents('000300')
    assert False, 'should have raised'
except NotImplementedError:
    print('OK: get_index_constituents raises NotImplementedError')
"`

Expected: all assertions pass

- [ ] **Step 5: Commit**

```bash
git add stockquant/data/source_tickflow.py
git commit -m "feat: complete TickFlow data source with index, trade_dates, factory registration"
```

---

### Task 6: 集成测试 — 完整数据源对比 & 填充 stock_info

**Files:**
- Create: `tests/test_source_tickflow.py`

- [ ] **Step 1: 编写集成测试**

```python
"""TickFlow 数据源集成测试。"""
import pytest
from stockquant.data.source_tickflow import TickFlowDataSource, _to_tf_symbol, _from_tf_symbol


class TestSymbolConversion:
    def test_sh_stock(self):
        assert _to_tf_symbol("600000") == "600000.SH"
        assert _to_tf_symbol("688001") == "688001.SH"

    def test_sz_stock(self):
        assert _to_tf_symbol("000001") == "000001.SZ"
        assert _to_tf_symbol("300001") == "300001.SZ"

    def test_from_tf_symbol(self):
        assert _from_tf_symbol("600000.SH") == "600000"
        assert _from_tf_symbol("000001.SZ") == "000001"


class TestTickFlowDataSource:
    @pytest.fixture
    def ds(self):
        return TickFlowDataSource()

    def test_get_stock_list(self, ds):
        df = ds.get_stock_list()
        assert len(df) > 4000
        assert "code" in df.columns
        assert "name" in df.columns

    def test_industry_map(self, ds):
        imap = ds._load_industry_map()
        assert len(imap) > 3000
        industries = set(imap.values())
        assert len(industries) >= 25

    def test_get_daily_bars(self, ds):
        df = ds.get_daily_bars("000001", "2020-01-01", "2020-01-31")
        assert not df.empty
        assert list(df.columns) == [
            "code", "date", "open", "high", "low", "close",
            "volume", "amount", "turnover", "pct_change", "change",
        ]

    def test_get_index_daily(self, ds):
        df = ds.get_index_daily("000300", "2020-01-01", "2020-01-31")
        assert not df.empty
        assert list(df.columns) == [
            "code", "date", "open", "high", "low", "close", "volume", "amount",
        ]

    def test_factory_registration(self):
        from stockquant.data.data_source import DataSourceFactory
        DataSourceFactory.reset_instances()
        ds = DataSourceFactory.create("tickflow")
        assert isinstance(ds, TickFlowDataSource)

    def test_not_implemented(self, ds):
        with pytest.raises(NotImplementedError):
            ds.get_index_constituents("000300")
        with pytest.raises(NotImplementedError):
            ds.get_finance_data("000001")
```

- [ ] **Step 2: 运行集成测试**

Run: `.venv/bin/pytest tests/test_source_tickflow.py -v --tb=short`

Expected: 8 tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_source_tickflow.py stockquant/data/source_tickflow.py
git commit -m "test: add integration tests for TickFlow data source"
```

---

### Task 7: 用 TickFlow 行业数据更新 stock_info 表

**Files:**
- Create: `/tmp/update_stock_info_tickflow.py`

- [ ] **Step 1: 编写数据更新脚本**

```python
"""用 TickFlow Shenwan 行业分类更新 stock_info 表。"""
import pandas as pd
import numpy as np
from stockquant.data.database import Database
from stockquant.data.source_tickflow import TickFlowDataSource

print("1. 从 TickFlow 获取行业+股本数据...")
ds = TickFlowDataSource()
stock_info = ds.get_stock_info()
print(f"   总计: {len(stock_info)} 只股票")
print(f"   有行业: {stock_info['industry'].notna().sum()}")
print(f"   有 total_shares: {stock_info['total_shares'].notna().sum()}")
print(f"   有 float_shares: {stock_info['float_shares'].notna().sum()}")

# 保留现有市值数据（Sina 已填充），仅更新行业和股本
db = Database()
existing = db.query("SELECT code, total_cap, float_cap FROM stock_info")
if not existing.empty:
    existing_map = existing.set_index("code")
    for col in ["total_cap", "float_cap"]:
        stock_info[col] = stock_info["code"].map(
            lambda c: existing_map.loc[c, col]
            if c in existing_map.index and pd.notna(existing_map.loc[c, col])
            else stock_info.loc[stock_info["code"] == c, col].values[0]
        )

print("2. 写入数据库...")
db.truncate("stock_info")
n = db.insert_or_ignore(stock_info, "stock_info")
db.close()
print(f"   写入 {n} 条记录")

# 验证
db2 = Database()
cnt = db2.conn.execute("SELECT COUNT(*) FROM stock_info").fetchone()[0]
ind_cnt = db2.conn.execute(
    "SELECT COUNT(*) FROM stock_info WHERE industry IS NOT NULL AND industry != ''"
).fetchone()[0]
fc_cnt = db2.conn.execute(
    "SELECT COUNT(*) FROM stock_info WHERE float_cap IS NOT NULL"
).fetchone()[0]
print(f"\n验证: {cnt} 行, {ind_cnt} 有行业, {fc_cnt} 有流通市值")
db2.close()
```

- [ ] **Step 2: 运行脚本**

Run: `.venv/bin/python /tmp/update_stock_info_tickflow.py`

Expected: `>5000 rows, >3000 with industry, existing market cap preserved`

- [ ] **Step 3: Commit**

```bash
git add stockquant/data/source_tickflow.py
git commit -m "feat: update stock_info with TickFlow Shenwan SW1 industry classification"
```

---

## File Summary

| 文件 | 操作 | 说明 |
|------|------|------|
| `stockquant/data/source_tickflow.py` | 新建 | TickFlow 数据源完整实现 |
| `stockquant/data/source_itick.py` | 不变 | 保留现有 iTick 适配器 |
| `tests/test_source_tickflow.py` | 新建 | 集成测试 |
| `/tmp/update_stock_info_tickflow.py` | 临时脚本 | 用 TickFlow 行业更新 stock_info |
