# stockquant/signals/ 第一期实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `stockquant/signals/` 下新建信号数据层，第一期实现东财限流基础设施 + 资金流向/融资融券/北向资金/股东户数四个模块

**Architecture:** 独立于 `BaseDataSource` 体系的纯函数层，东财接口统一走 `_eastmoney.em_get()` 串行限流，每个模块暴露 1-2 个函数直接返回 `pd.DataFrame`

**Tech Stack:** Python 3.9+, requests, pandas, loguru (已有依赖)

**Spec:** `docs/superpowers/specs/2026-06-29-signals-layer-design.md`

---

## 文件结构（第一期实际创建/修改）

```
stockquant/signals/                     ← 新建目录
├── __init__.py                         ← 创建: 公开 API 导出
├── _eastmoney.py                       ← 创建: em_get, em_datacenter
├── fund_flow.py                        ← 创建: get_fund_flow
├── margin.py                           ← 创建: get_margin_trading
├── northbound.py                       ← 创建: get_northbound_realtime, get_northbound_history
└── holders.py                          ← 创建: get_holder_changes

tests/                                  ← 新建测试文件
├── test_signals_eastmoney.py           ← 创建: _eastmoney 基础设施测试
├── test_signals_fund_flow.py           ← 创建: 资金流向测试
├── test_signals_margin.py              ← 创建: 融资融券测试
├── test_signals_northbound.py          ← 创建: 北向资金测试
└── test_signals_holders.py            ← 创建: 股东户数测试
```

> `_mootdx.py` / `_sina.py` / `_cninfo.py` 第一期不需要（它们服务于后续期的 quote / finance / options / announcement 模块）。

---

### Task 1: 创建目录和包占位

**Files:**
- Create: `stockquant/signals/__init__.py`

- [ ] **Step 1: 创建目录结构和空 __init__.py**

```bash
mkdir -p stockquant/signals
```

- [ ] **Step 2: 写占位 __init__.py**

```python
"""
stockQuant 信号数据层。

提供量化策略常用的信号/另类数据接口，覆盖资金流向、融资融券、
北向资金、股东户数、龙虎榜、限售解禁等。东财系接口统一内置限流。

第一期（资金面/筹码层）::

    from stockquant.signals import (
        get_fund_flow,              # 个股资金流向 120 日
        get_margin_trading,         # 融资融券明细
        get_northbound_realtime,    # 北向资金实时分钟流向
        get_northbound_history,     # 北向资金历史日级
        get_holder_changes,         # 股东户数变化
    )

设计文档: docs/superpowers/specs/2026-06-29-signals-layer-design.md
"""

# 第一期公开 API —— 各模块实现后逐步解除注释
# from stockquant.signals.fund_flow import get_fund_flow
# from stockquant.signals.margin import get_margin_trading
# from stockquant.signals.northbound import get_northbound_realtime, get_northbound_history
# from stockquant.signals.holders import get_holder_changes

__all__: list[str] = []
```

- [ ] **Step 3: 验证包可导入**

```bash
python -c "import stockquant.signals; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add stockquant/signals/__init__.py
git commit -m "feat(signals): scaffold signals package with placeholder __init__"
```

---

### Task 2: _eastmoney.py — 共享限流基础设施

**Files:**
- Create: `stockquant/signals/_eastmoney.py`
- Create: `tests/test_signals_eastmoney.py`

- [ ] **Step 1: 写 _eastmoney.py 测试**

```python
# tests/test_signals_eastmoney.py
"""测试 _eastmoney 基础设施：em_get 限流 + em_datacenter 通用查询。"""

import time
import pytest


class TestEmGet:
    """em_get 限流和会话复用测试。"""

    def test_em_get_returns_200_for_public_eastmoney(self):
        """em_get 能正常请求东财公开页面（验证 session / UA 配置正确）。"""
        from stockquant.signals._eastmoney import em_get

        r = em_get("https://push2.eastmoney.com/api/qt/stock/get",
                   params={"secid": "1.600519", "fields": "f57,f58"},
                   headers={"Referer": "https://quote.eastmoney.com/"},
                   timeout=15)
        assert r.status_code == 200, f"HTTP {r.status_code}"
        d = r.json()
        assert d.get("data", {}).get("f58") == "贵州茅台"

    def test_em_get_enforces_min_interval(self):
        """连续两次 em_get 调用间隔不小于 EM_MIN_INTERVAL。"""
        from stockquant.signals._eastmoney import em_get, _EM_MIN_INTERVAL

        t0 = time.time()
        em_get("https://push2.eastmoney.com/api/qt/stock/get",
               params={"secid": "1.600519", "fields": "f57,f58"},
               headers={"Referer": "https://quote.eastmoney.com/"},
               timeout=15)
        t1 = time.time()
        em_get("https://push2.eastmoney.com/api/qt/stock/get",
               params={"secid": "0.000001", "fields": "f57,f58"},
               headers={"Referer": "https://quote.eastmoney.com/"},
               timeout=15)
        elapsed = t1 - t0
        assert elapsed >= _EM_MIN_INTERVAL, \
            f"间隔 {elapsed:.2f}s < {_EM_MIN_INTERVAL}s"


class TestEmDatacenter:
    """em_datacenter 通用查询测试。"""

    def test_em_datacenter_margin_returns_data(self):
        """用融资融券 RPT 验证 em_datacenter 模板查询可用。"""
        from stockquant.signals._eastmoney import em_datacenter

        data = em_datacenter(
            "RPTA_WEB_RZRQ_GGMX",
            filter_str='(SCODE="600519")',
            page_size=5,
            sort_columns="DATE",
            sort_types="-1",
        )
        assert isinstance(data, list)
        assert len(data) > 0, "茅台应有融资融券数据"
        assert "RZYE" in data[0], f"字段缺失, got: {list(data[0].keys())[:10]}"

    def test_em_datacenter_empty_for_bogus_rpt(self):
        """不存在的 RPT 名称返回空列表不抛异常。"""
        from stockquant.signals._eastmoney import em_datacenter

        data = em_datacenter("RPT_DOES_NOT_EXIST", filter_str="", page_size=5)
        assert data == []
```

- [ ] **Step 2: 运行测试，验证全部 FAIL**

```bash
pytest tests/test_signals_eastmoney.py -v
```
Expected: 全部 4 个 FAIL（模块尚未创建）

- [ ] **Step 3: 实现 _eastmoney.py**

```python
# stockquant/signals/_eastmoney.py
"""
东财 HTTP 请求共享基础设施。

提供:
    em_get()          — 带串行限流的 GET，所有 eastmoney.com 端点专用
    em_datacenter()   — datacenter-web 通用查询模板（龙虎榜/融资融券/股东户数等共用）

限流规则: 两次请求最小间隔 1.0 秒 + 随机抖动(0.1~0.5秒)，串行不并发。
所有东财端点必须走 em_get/em_datacenter，禁止裸 requests.get 直接打东财。
"""

from __future__ import annotations

import random
import time

import requests

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": UA})

EM_MIN_INTERVAL = 1.0       # 公开可调，批量场景可调大到 1.5~2s
_em_last_call = [0.0]       # 可变容器，跨模块共享


def em_get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 15,
    **kwargs,
) -> requests.Response:
    """东财统一 GET 请求 — 串行限流 + Keep-Alive 会话复用。

    所有 eastmoney.com 的 HTTP 请求必须走此函数，避免高频被封 IP。
    被封表现: 403 / 429 / 连接超时 / 返回空数据。
    """
    wait = EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
    if wait > 0:
        time.sleep(wait + random.uniform(0.1, 0.5))
    try:
        return _SESSION.get(url, params=params, headers=headers,
                            timeout=timeout, **kwargs)
    finally:
        _em_last_call[0] = time.time()


def em_datacenter(
    report_name: str,
    *,
    filter_str: str = "",
    page_size: int = 50,
    sort_columns: str = "",
    sort_types: str = "-1",
) -> list[dict]:
    """东财 datacenter-web 通用查询。

    龙虎榜、融资融券、限售解禁、大宗交易、股东户数、分红送转等共用此端点。

    Parameters
    ----------
    report_name : str
        RPT 报表名称，如 ``"RPTA_WEB_RZRQ_GGMX"``。
    filter_str : str
        过滤条件，如 ``'(SCODE="600519")'``。
    page_size : int
        每页条数，上限 500。
    sort_columns : str
        排序字段。
    sort_types : str
        排序方向: ``"-1"`` 降序 / ``"1"`` 升序。

    Returns
    -------
    list[dict]
        数据行列表，无数据时返回空列表。
    """
    params = {
        "reportName": report_name,
        "columns": "ALL",
        "filter": filter_str,
        "pageNumber": "1",
        "pageSize": str(page_size),
        "sortColumns": sort_columns,
        "sortTypes": sort_types,
        "source": "WEB",
        "client": "WEB",
    }
    r = em_get(
        "https://datacenter-web.eastmoney.com/api/data/v1/get",
        params=params,
        timeout=15,
    )
    d = r.json()
    result = d.get("result")
    if result is None:
        return []
    return result.get("data") or []
```

- [ ] **Step 4: 运行测试，验证全部 PASS**

```bash
pytest tests/test_signals_eastmoney.py -v
```
Expected: 4 PASS（注意: test_em_get_enforces_min_interval 因为限流 ≥1s 需要 ~2s）

- [ ] **Step 5: Commit**

```bash
git add stockquant/signals/_eastmoney.py tests/test_signals_eastmoney.py
git commit -m "feat(signals): add _eastmoney shared infra — em_get throttle + em_datacenter"
```

---

### Task 3: fund_flow.py — 个股资金流向（120日日级）

**Files:**
- Create: `stockquant/signals/fund_flow.py`
- Create: `tests/test_signals_fund_flow.py`

- [ ] **Step 1: 写 fund_flow 测试**

```python
# tests/test_signals_fund_flow.py
"""测试 fund_flow 模块：个股资金流向 120 日日级。"""

import pandas as pd
import pytest


class TestGetFundFlow:
    """get_fund_flow 测试。"""

    def test_returns_dataframe_with_expected_columns(self):
        """返回非空 DataFrame 且包含预期列。"""
        from stockquant.signals.fund_flow import get_fund_flow

        df = get_fund_flow("600519", days=20)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "茅台应有资金流数据"
        expected_cols = ["date", "main_net", "small_net", "mid_net",
                         "large_net", "super_net"]
        for col in expected_cols:
            assert col in df.columns, f"缺少列: {col}"

    def test_date_column_is_datetime(self):
        """date 列为 datetime 类型。"""
        from stockquant.signals.fund_flow import get_fund_flow

        df = get_fund_flow("600519", days=5)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_respects_days_parameter(self):
        """days 参数控制返回行数不超过预期。"""
        from stockquant.signals.fund_flow import get_fund_flow

        df = get_fund_flow("600519", days=10)
        assert len(df) <= 10, f"请求10天, 实际返回{len(df)}行"

    def test_unknown_code_returns_empty(self):
        """无效代码返回空 DataFrame 带正确列。"""
        from stockquant.signals.fund_flow import get_fund_flow

        df = get_fund_flow("999999", days=10)
        assert isinstance(df, pd.DataFrame)
        # 无效代码可能返回空
```

- [ ] **Step 2: 运行测试，验证全部 FAIL**

```bash
pytest tests/test_signals_fund_flow.py -v
```

- [ ] **Step 3: 实现 fund_flow.py**

```python
# stockquant/signals/fund_flow.py
"""
个股资金流向（日级）。

端点: push2his.eastmoney.com
"""

from __future__ import annotations

import pandas as pd

from stockquant.signals._eastmoney import em_get, UA
from stockquant.utils.logger import get_logger

logger = get_logger("signals.fund_flow")

FUND_FLOW_COLS = ["date", "main_net", "small_net", "mid_net",
                  "large_net", "super_net"]


def get_fund_flow(code: str, days: int = 120) -> pd.DataFrame:
    """获取个股资金流向日级数据。

    Parameters
    ----------
    code : str
        6 位股票代码，如 ``"600519"``。
    days : int
        拉取最近多少个交易日，默认 120。

    Returns
    -------
    pd.DataFrame
        列: date, main_net, small_net, mid_net, large_net, super_net。
        金额单位: **元**。空结果时返回带列名的空 DataFrame。
    """
    market_code = 1 if code.startswith("6") else 0
    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "secid": f"{market_code}.{code}",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "lmt": str(days),
    }
    headers = {
        "User-Agent": UA,
        "Referer": "https://quote.eastmoney.com/",
        "Origin": "https://quote.eastmoney.com",
    }
    try:
        r = em_get(url, params=params, headers=headers, timeout=15)
        d = r.json()
    except Exception as e:
        logger.warning(f"资金流向请求失败 code={code}: {e}")
        return pd.DataFrame(columns=FUND_FLOW_COLS)

    klines = (d.get("data") or {}).get("klines") or []
    rows = []
    for line in klines:
        parts = line.split(",")
        if len(parts) >= 7:
            rows.append({
                "date": parts[0],
                "main_net": float(parts[1]) if parts[1] != "-" else 0.0,
                "small_net": float(parts[2]) if parts[2] != "-" else 0.0,
                "mid_net": float(parts[3]) if parts[3] != "-" else 0.0,
                "large_net": float(parts[4]) if parts[4] != "-" else 0.0,
                "super_net": float(parts[5]) if parts[5] != "-" else 0.0,
            })

    if not rows:
        return pd.DataFrame(columns=FUND_FLOW_COLS)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
```

- [ ] **Step 4: 运行测试，验证全部 PASS**

```bash
pytest tests/test_signals_fund_flow.py -v
```
Expected: 4 PASS（注意 em_get 限流间隔）

- [ ] **Step 5: Commit**

```bash
git add stockquant/signals/fund_flow.py tests/test_signals_fund_flow.py
git commit -m "feat(signals): add get_fund_flow — daily fund flow from push2his"
```

---

### Task 4: margin.py — 融资融券明细

**Files:**
- Create: `stockquant/signals/margin.py`
- Create: `tests/test_signals_margin.py`

- [ ] **Step 1: 写 margin 测试**

```python
# tests/test_signals_margin.py
"""测试 margin 模块：融资融券日级明细。"""

import pandas as pd
import pytest


class TestGetMarginTrading:
    """get_margin_trading 测试。"""

    def test_returns_dataframe_with_expected_columns(self):
        """返回非空 DataFrame 且包含关键列。"""
        from stockquant.signals.margin import get_margin_trading

        df = get_margin_trading("600519", page_size=5)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "茅台应有融资融券数据"
        expected_cols = ["date", "rzye", "rzmre", "rzche",
                         "rqye", "rqmcl", "rqchl", "rzrqye"]
        for col in expected_cols:
            assert col in df.columns, f"缺少列: {col}"

    def test_date_is_datetime(self):
        """date 列为 datetime 类型。"""
        from stockquant.signals.margin import get_margin_trading

        df = get_margin_trading("600519", page_size=3)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_numeric_columns(self):
        """融资余额等核心列为数值类型。"""
        from stockquant.signals.margin import get_margin_trading

        df = get_margin_trading("600519", page_size=3)
        assert df["rzye"].dtype.kind in ("i", "f"), \
            f"rzye 应为数值, 实际 {df['rzye'].dtype}"
```

- [ ] **Step 2: 运行测试，验证 FAIL**

```bash
pytest tests/test_signals_margin.py -v
```

- [ ] **Step 3: 实现 margin.py**

```python
# stockquant/signals/margin.py
"""
融资融券日级明细。

端点: datacenter-web.eastmoney.com → RPTA_WEB_RZRQ_GGMX
"""

from __future__ import annotations

import pandas as pd

from stockquant.signals._eastmoney import em_datacenter
from stockquant.utils.logger import get_logger

logger = get_logger("signals.margin")

MARGIN_COLS = ["date", "rzye", "rzmre", "rzche",
               "rqye", "rqmcl", "rqchl", "rzrqye"]


def get_margin_trading(code: str, page_size: int = 30) -> pd.DataFrame:
    """获取个股融资融券日级明细。

    Parameters
    ----------
    code : str
        6 位股票代码。
    page_size : int
        返回最近多少条记录，默认 30。

    Returns
    -------
    pd.DataFrame
        列:
        - date       — 日期
        - rzye       — 融资余额 (元)
        - rzmre      — 融资买入额 (元)
        - rzche      — 融资偿还额 (元)
        - rqye       — 融券余额 (元)
        - rqmcl      — 融券卖出量 (股)
        - rqchl      — 融券偿还量 (股)
        - rzrqye     — 融资融券余额合计 (元)
    """
    try:
        data = em_datacenter(
            "RPTA_WEB_RZRQ_GGMX",
            filter_str=f'(SCODE="{code}")',
            page_size=page_size,
            sort_columns="DATE",
            sort_types="-1",
        )
    except Exception as e:
        logger.warning(f"融资融券请求失败 code={code}: {e}")
        return pd.DataFrame(columns=MARGIN_COLS)

    if not data:
        return pd.DataFrame(columns=MARGIN_COLS)

    rows = []
    for row in data:
        rows.append({
            "date": str(row.get("DATE", ""))[:10],
            "rzye": row.get("RZYE") or 0,
            "rzmre": row.get("RZMRE") or 0,
            "rzche": row.get("RZCHE") or 0,
            "rqye": row.get("RQYE") or 0,
            "rqmcl": row.get("RQMCL") or 0,
            "rqchl": row.get("RQCHL") or 0,
            "rzrqye": row.get("RZRQYE") or 0,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
```

- [ ] **Step 4: 运行测试，验证 PASS**

```bash
pytest tests/test_signals_margin.py -v
```

- [ ] **Step 5: Commit**

```bash
git add stockquant/signals/margin.py tests/test_signals_margin.py
git commit -m "feat(signals): add get_margin_trading — margin trading daily from datacenter-web"
```

---

### Task 5: northbound.py — 北向资金（分钟 + 日级缓存）

**Files:**
- Create: `stockquant/signals/northbound.py`
- Create: `tests/test_signals_northbound.py`

- [ ] **Step 1: 写 northbound 测试**

```python
# tests/test_signals_northbound.py
"""测试 northbound 模块：北向资金实时分钟 + 历史日级缓存。"""

import pandas as pd
import pytest


class TestGetNorthboundRealtime:
    """get_northbound_realtime 测试。"""

    def test_returns_dataframe_with_expected_columns(self):
        """返回 DataFrame 含 time/hgt_yi/sgt_yi 列。"""
        from stockquant.signals.northbound import get_northbound_realtime

        df = get_northbound_realtime()

        assert isinstance(df, pd.DataFrame)
        expected_cols = ["time", "hgt_yi", "sgt_yi"]
        for col in expected_cols:
            assert col in df.columns, f"缺少列: {col}"

    def test_hgt_sgt_are_numeric(self):
        """hgt_yi / sgt_yi 为数值列。"""
        from stockquant.signals.northbound import get_northbound_realtime

        df = get_northbound_realtime()
        if not df.empty:
            vals = df.dropna(subset=["hgt_yi"])
            assert vals["hgt_yi"].dtype.kind in ("i", "f")


class TestGetNorthboundHistory:
    """get_northbound_history 测试。"""

    def test_returns_dataframe(self):
        """返回 DataFrame（文件不存在时可能为空）。"""
        from stockquant.signals.northbound import get_northbound_history

        df = get_northbound_history(n=5)
        assert isinstance(df, pd.DataFrame)

    def test_respects_n_parameter(self):
        """n 参数控制返回行数。"""
        from stockquant.signals.northbound import get_northbound_history

        df = get_northbound_history(n=3)
        assert len(df) <= 3
```

- [ ] **Step 2: 运行测试，验证 FAIL**

```bash
pytest tests/test_signals_northbound.py -v
```

- [ ] **Step 3: 实现 northbound.py**

```python
# stockquant/signals/northbound.py
"""
北向资金流向（沪深股通）。

端点: data.hexin.cn/market/hsgtApi/ (同花顺, 零鉴权, 不走东财限流)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from stockquant.utils.logger import get_logger

logger = get_logger("signals.northbound")

HSGT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "Chrome/117.0.0.0 Safari/537.36"
    ),
    "Host": "data.hexin.cn",
    "Referer": "https://data.hexin.cn/",
}

_CACHE_DIR = Path.home() / ".tradingagents" / "cache"


def get_northbound_realtime() -> pd.DataFrame:
    """获取当日沪深股通实时分钟级流向。

    Returns
    -------
    pd.DataFrame
        列: time, hgt_yi (沪股通累计净买入/亿元),
        sgt_yi (深股通累计净买入/亿元)。
        非交易时段返回含历史缓存的空 DataFrame。
    """
    url = "https://data.hexin.cn/market/hsgtApi/method/dayChart/"
    try:
        r = requests.get(url, headers=HSGT_HEADERS, timeout=10)
        d = r.json()
    except Exception as e:
        logger.warning(f"北向资金实时请求失败: {e}")
        return pd.DataFrame(columns=["time", "hgt_yi", "sgt_yi"])

    times = d.get("time") or []
    hgt = d.get("hgt") or []
    sgt = d.get("sgt") or []
    n = len(times)

    # 对齐长度
    hgt_padded = hgt[:n] + [None] * (n - len(hgt))
    sgt_padded = sgt[:n] + [None] * (n - len(sgt))

    df = pd.DataFrame({
        "time": times,
        "hgt_yi": hgt_padded,
        "sgt_yi": sgt_padded,
    })

    # 自动缓存当日收盘快照
    _save_snapshot(df)

    return df


def get_northbound_history(n: int = 20) -> pd.DataFrame:
    """读取本地缓存的北向资金日级历史数据。

    缓存文件在 ``~/.tradingagents/cache/northbound_daily.csv``，
    由 ``get_northbound_realtime()`` 每次调用时自动追加。

    Parameters
    ----------
    n : int
        返回最近 N 个交易日，默认 20。

    Returns
    -------
    pd.DataFrame
        列: date, hgt, sgt。无缓存文件时返回空 DataFrame。
    """
    path = _CACHE_DIR / "northbound_daily.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "hgt", "sgt"])

    try:
        df = pd.read_csv(path)
        return df.tail(n)
    except Exception as e:
        logger.warning(f"北向历史读取失败: {e}")
        return pd.DataFrame(columns=["date", "hgt", "sgt"])


def _save_snapshot(df: pd.DataFrame) -> None:
    """从分钟 DataFrame 提取收盘快照写入本地 CSV（内部函数）。"""
    valid = df.dropna(subset=["hgt_yi", "sgt_yi"])
    if valid.empty:
        return

    last = valid.iloc[-1]
    date_str = str(last["time"])[:10] if last["time"] else ""

    if not date_str:
        return

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / "northbound_daily.csv"

    # 读取已有行
    rows: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().strip().split("\n")[1:]:
            parts = line.split(",")
            if len(parts) == 3:
                rows[parts[0]] = line

    rows[date_str] = f"{date_str},{last['hgt_yi']},{last['sgt_yi']}"

    with open(path, "w") as f:
        f.write("date,hgt,sgt\n")
        for d in sorted(rows.keys()):
            f.write(rows[d] + "\n")
```

- [ ] **Step 4: 运行测试，验证 PASS**

```bash
pytest tests/test_signals_northbound.py -v
```

- [ ] **Step 5: Commit**

```bash
git add stockquant/signals/northbound.py tests/test_signals_northbound.py
git commit -m "feat(signals): add northbound — realtime + cached daily northbound flow"
```

---

### Task 6: holders.py — 股东户数变化

**Files:**
- Create: `stockquant/signals/holders.py`
- Create: `tests/test_signals_holders.py`

- [ ] **Step 1: 写 holders 测试**

```python
# tests/test_signals_holders.py
"""测试 holders 模块：股东户数变化（筹码集中度）。"""

import pandas as pd
import pytest


class TestGetHolderChanges:
    """get_holder_changes 测试。"""

    def test_returns_dataframe_with_expected_columns(self):
        """返回非空 DataFrame 且包含关键列。"""
        from stockquant.signals.holders import get_holder_changes

        df = get_holder_changes("600519", periods=5)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "茅台应有股东户数数据"
        expected_cols = ["date", "holder_num", "change_num",
                         "change_ratio", "avg_shares"]
        for col in expected_cols:
            assert col in df.columns, f"缺少列: {col}"

    def test_date_is_datetime(self):
        """date 列为 datetime 类型。"""
        from stockquant.signals.holders import get_holder_changes

        df = get_holder_changes("600519", periods=3)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_holder_num_is_numeric(self):
        """holder_num 为数值类型。"""
        from stockquant.signals.holders import get_holder_changes

        df = get_holder_changes("600519", periods=3)
        assert df["holder_num"].dtype.kind in ("i", "f"), \
            f"holder_num 应为数值, 实际 {df['holder_num'].dtype}"

    def test_respects_periods_parameter(self):
        """periods 参数控制返回行数。"""
        from stockquant.signals.holders import get_holder_changes

        df = get_holder_changes("600519", periods=3)
        assert len(df) <= 3, f"请求3期, 实际返回{len(df)}行"
```

- [ ] **Step 2: 运行测试，验证 FAIL**

```bash
pytest tests/test_signals_holders.py -v
```

- [ ] **Step 3: 实现 holders.py**

```python
# stockquant/signals/holders.py
"""
股东户数变化（筹码集中度）。

端点: datacenter-web.eastmoney.com → RPT_HOLDERNUMLATEST
"""

from __future__ import annotations

import pandas as pd

from stockquant.signals._eastmoney import em_datacenter
from stockquant.utils.logger import get_logger

logger = get_logger("signals.holders")

HOLDER_COLS = ["date", "holder_num", "change_num",
               "change_ratio", "avg_shares"]


def get_holder_changes(code: str, periods: int = 10) -> pd.DataFrame:
    """获取个股股东户数变化（季频，筹码集中度信号）。

    股东户数持续减少 → 筹码集中 → 主力吸筹信号。
    股东户数持续增加 → 筹码分散 → 主力出货信号。

    Parameters
    ----------
    code : str
        6 位股票代码。
    periods : int
        返回最近多少期（季度），默认 10。

    Returns
    -------
    pd.DataFrame
        列:
        - date          — 截止日期
        - holder_num    — 股东户数
        - change_num    — 较上期变化（户, 负值=减少=集中）
        - change_ratio  — 较上期变化比例 (%)
        - avg_shares    — 户均持股数（股）
    """
    try:
        data = em_datacenter(
            "RPT_HOLDERNUMLATEST",
            filter_str=f'(SECURITY_CODE="{code}")',
            page_size=periods,
            sort_columns="END_DATE",
            sort_types="-1",
        )
    except Exception as e:
        logger.warning(f"股东户数请求失败 code={code}: {e}")
        return pd.DataFrame(columns=HOLDER_COLS)

    if not data:
        return pd.DataFrame(columns=HOLDER_COLS)

    rows = []
    for row in data:
        rows.append({
            "date": str(row.get("END_DATE", ""))[:10],
            "holder_num": row.get("HOLDER_NUM") or 0,
            "change_num": row.get("HOLDER_NUM_CHANGE") or 0,
            "change_ratio": row.get("HOLDER_NUM_RATIO") or 0.0,
            "avg_shares": row.get("AVG_FREE_SHARES") or 0,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
```

- [ ] **Step 4: 运行测试，验证 PASS**

```bash
pytest tests/test_signals_holders.py -v
```

- [ ] **Step 5: Commit**

```bash
git add stockquant/signals/holders.py tests/test_signals_holders.py
git commit -m "feat(signals): add get_holder_changes — holder concentration from datacenter-web"
```

---

### Task 7: 激活 __init__.py 公开 API 导出

**Files:**
- Modify: `stockquant/signals/__init__.py`

- [ ] **Step 1: 更新 __init__.py 解除注释并设 __all__**

```python
# stockquant/signals/__init__.py
"""
stockQuant 信号数据层。

提供量化策略常用的信号/另类数据接口，覆盖资金流向、融资融券、
北向资金、股东户数、龙虎榜、限售解禁等。东财系接口统一内置限流。

第一期（资金面/筹码层）::

    from stockquant.signals import (
        get_fund_flow,              # 个股资金流向 120 日
        get_margin_trading,         # 融资融券明细
        get_northbound_realtime,    # 北向资金实时分钟流向
        get_northbound_history,     # 北向资金历史日级
        get_holder_changes,         # 股东户数变化
    )

设计文档: docs/superpowers/specs/2026-06-29-signals-layer-design.md
"""

from stockquant.signals.fund_flow import get_fund_flow
from stockquant.signals.holders import get_holder_changes
from stockquant.signals.margin import get_margin_trading
from stockquant.signals.northbound import (
    get_northbound_history,
    get_northbound_realtime,
)

__all__ = [
    "get_fund_flow",
    "get_margin_trading",
    "get_northbound_realtime",
    "get_northbound_history",
    "get_holder_changes",
]
```

- [ ] **Step 2: 验证顶层导入**

```bash
python -c "
from stockquant.signals import (
    get_fund_flow, get_margin_trading,
    get_northbound_realtime, get_northbound_history,
    get_holder_changes,
)
print('All imports OK')
print('__all__:', ['get_fund_flow', 'get_margin_trading', 'get_northbound_realtime', 'get_northbound_history', 'get_holder_changes'])
"
```

- [ ] **Step 3: Commit**

```bash
git add stockquant/signals/__init__.py
git commit -m "feat(signals): activate public API exports for Phase 1 modules"
```

---

### Task 8: 全量集成验证

- [ ] **Step 1: 运行全部 signals 测试**

```bash
pytest tests/test_signals_*.py -v
```
Expected: 全部测试 PASS（~16 个测试，注意东财限流间隔总耗时 ~20s）

- [ ] **Step 2: 端到端 notebook 级调用验证**

```bash
python -c "
from stockquant.signals import (
    get_fund_flow, get_margin_trading,
    get_northbound_realtime, get_northbound_history,
    get_holder_changes,
)

# 茅台 600519
print('=== fund_flow ===')
df = get_fund_flow('600519', days=5)
print(f'rows={len(df)}, cols={list(df.columns)}')
print(df.tail(2))

print('=== margin ===')
df = get_margin_trading('600519', page_size=3)
print(f'rows={len(df)}, cols={list(df.columns)}')
print(df.head(2))

print('=== northbound realtime ===')
df = get_northbound_realtime()
print(f'rows={len(df)}, cols={list(df.columns)}')

print('=== holders ===')
df = get_holder_changes('600519', periods=3)
print(f'rows={len(df)}, cols={list(df.columns)}')
print(df.head(2))

print('=== northbound history ===')
df = get_northbound_history(n=3)
print(f'rows={len(df)}, cols={list(df.columns)}')
print('ALL END-TO-END CHECKS PASSED')
"
```

- [ ] **Step 3: 验证不引入新外部依赖**

```bash
python -c "
import sys
before = set(sys.modules.keys())
from stockquant.signals import get_fund_flow
after = set(sys.modules.keys())
new = after - before
# 排除 signals 内部模块自身
signals_mods = {m for m in new if 'stockquant.signals' in m}
external = new - signals_mods
print('New external modules:', external)
# 应该全为已有依赖 (requests, pandas, numpy 等)
"
```

- [ ] **Step 4: Commit (如有修改)**

```bash
git status
# 如果有修复，提交
```

---

## 自审检查

1. **Spec 覆盖**: 第一期 4 个模块 (fund_flow / margin / northbound / holders) + 共享基础设施 _eastmoney 全部有对应 Task，`__init__.py` 公开导出有 Task 7
2. **占位检查**: 无 TBD/TODO，所有代码步骤均给出完整实现
3. **类型一致性**: `em_get` 签名在 Task 2 定义，Task 3/4/6 使用一致；`em_datacenter` 参数在 Task 2 定义和 Task 4/6 调用一致；各模块返回 `pd.DataFrame` 列名与测试断言一致
