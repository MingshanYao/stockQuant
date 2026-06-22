# Phase 1: Config + Data 层重构 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 Config 的 API Key 泄露和单例陷阱，拆分 TypedConfig；修复 Data 层 fetch_daily 自动下载和 StockUniverse 无时间过滤的问题。

**Architecture:** 5 个独立 Task，顺序执行。Task 1-3 处理 Config，Task 4-5 处理 Data。每个 Task 可独立测试和 commit。

**Tech Stack:** Python 3.12, DuckDB, pandas

## Global Constraints

- 所有改动保持向后兼容，现有测试全部通过
- API Key 只从环境变量读取，不写入配置文件
- fetch_daily 去掉远程回退后，`_fetch_and_persist_daily` 方法保留不删
- Config 默认值统一在 `config/default.yaml` 中定义，代码中不重复 fallback

---

### Task 1: 移除泄露的 API Key

**Files:**
- Modify: `config/default.yaml:17`
- Modify: `stockquant/data/source_tickflow.py:81`

**Interfaces:**
- Produces: `source_tickflow.py` 从 `os.environ.get("SQ_TICKFLOW_API_KEY")` 读取，与 `source_itick.py` / `source_tushare.py` 一致

- [ ] **Step 1: 删除 default.yaml 中的 API Key**

Edit `config/default.yaml` line 17:

```yaml
# Before:
  tickflow_api_key: "tk_6350a68e20e244e684710771976d87e1"  # 实时行情（可选）

# After:
  # tickflow_api_key 通过环境变量 SQ_TICKFLOW_API_KEY 设置
```

- [ ] **Step 2: 修改 source_tickflow.py 读取方式**

Edit `stockquant/data/source_tickflow.py` line 81:

```python
# Before:
api_key = Config().get("data_source.tickflow_api_key", None)

# After:
api_key = os.environ.get("SQ_TICKFLOW_API_KEY")
```

确保文件顶部已有 `import os`（检查 line ~12）。

- [ ] **Step 3: 验证**

```bash
# 不设环境变量时应返回 None
.venv/bin/python -c "
from stockquant.data.source_tickflow import TickFlowSource
s = TickFlowSource()
print('api_key:', 'None' if s._api_key is None else 'EXISTS (BAD!)')
"
```

Expected: `api_key: None`

- [ ] **Step 4: Commit**

```bash
git add config/default.yaml stockquant/data/source_tickflow.py
git commit -m "fix: remove API key from default.yaml, use SQ_TICKFLOW_API_KEY env var"
```

---

### Task 2: 修复 Config 单例陷阱

**Files:**
- Modify: `stockquant/utils/config.py:30-45` (__init__)
- Test: `tests/test_config.py` (若不存在则 create)

**Interfaces:**
- Produces: `Config(user_path)` 在已初始化后传入不同路径时抛出 `RuntimeError`

- [ ] **Step 1: 检查现有 test_config.py**

```bash
grep -rn "class.*TestConfig\|def test_" tests/test_config.py | head -10
```

- [ ] **Step 2: 添加测试**

在 `tests/test_config.py` 中添加：

```python
def test_singleton_rejects_different_path(tmp_path):
    """Config 已初始化后，传入不同路径应抛 RuntimeError。"""
    import yaml
    from stockquant.utils.config import Config

    Config.reset()

    # 创建两个不同的临时配置文件
    f1 = tmp_path / "config1.yaml"
    f2 = tmp_path / "config2.yaml"
    f1.write_text(yaml.dump({"project": {"name": "test1"}}))
    f2.write_text(yaml.dump({"project": {"name": "test2"}}))

    cfg1 = Config(str(f1))
    assert cfg1.get("project.name") == "test1"

    # 用不同路径再次构造应抛错
    with pytest.raises(RuntimeError, match="已初始化"):
        Config(str(f2))

    Config.reset()
```

- [ ] **Step 3: 实现 Config.__init__ 的路径检查**

Edit `stockquant/utils/config.py` `__init__` 方法：

```python
# 在 _data 已存在且 config_path 不为 None 时，检查路径是否一致
def __init__(self, config_path: str | None = None) -> None:
    if self._data:
        if config_path is not None and config_path != self._config_path:
            raise RuntimeError(
                f"Config 已从 {self._config_path} 初始化，"
                f"不能再用 {config_path} 初始化。先用 Config.reset() 重置。"
            )
        return
    # ... 原有初始化逻辑 ...
```

需要新增类属性 `_config_path: str | None = None` 并在初始化时记录。

- [ ] **Step 4: 运行测试**

```bash
.venv/bin/python -m pytest tests/test_config.py::test_singleton_rejects_different_path -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add stockquant/utils/config.py tests/test_config.py
git commit -m "fix: Config singleton raises RuntimeError on path mismatch"
```

---

### Task 3: TypedConfig 封装

**Files:**
- Create: `stockquant/utils/typed_config.py`
- Modify: `stockquant/utils/__init__.py`

**Interfaces:**
- Produces: `TypedConfig(cfg)` 提供属性访问 `cfg.backtest.initial_capital`
- 新旧共存，不强制迁移

- [ ] **Step 1: 创建 typed_config.py**

```python
"""TypedConfig — Config 的类型安全包装，提供属性访问和 IDE 自动补全。"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    commission: float = 0.0003
    slippage: float = 0.01

@dataclass
class DataFetchConfig:
    start_date: str = "2020-01-01"
    adjust: str = "hfq"
    timeout: int = 30
    batch_size: int = 50

@dataclass
class RiskConfig:
    max_drawdown_green: float = 0.04
    max_drawdown_yellow: float = 0.07
    max_drawdown_orange: float = 0.10
    max_positions: int = 50
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.20
    trailing_stop_pct: float = 0.05

class TypedConfig:
    """Config 的类型安全包装。

    用法:
        from stockquant.utils.config import Config
        cfg = TypedConfig(Config())

        print(cfg.backtest.initial_capital)  # 属性访问 + IDE 补全
    """

    def __init__(self, raw=None):
        from stockquant.utils.config import Config as _Config
        self._raw = raw if raw is not None else _Config()

    @cached_property
    def backtest(self) -> BacktestConfig:
        return BacktestConfig(
            initial_capital=float(self._raw.get("backtest.initial_capital", 1_000_000.0)),
            commission=float(self._raw.get("backtest.commission", 0.0003)),
            slippage=float(self._raw.get("backtest.slippage", 0.01)),
        )

    @cached_property
    def data_fetch(self) -> DataFetchConfig:
        return DataFetchConfig(
            start_date=self._raw.get("data_fetch.start_date", "2020-01-01"),
            adjust=self._raw.get("data_fetch.adjust", "hfq"),
            timeout=int(self._raw.get("data_fetch.timeout", 30)),
            batch_size=int(self._raw.get("data_fetch.batch_size", 50)),
        )

    @cached_property
    def risk(self) -> RiskConfig:
        return RiskConfig(
            max_drawdown_green=float(self._raw.get("risk.max_drawdown_green", 0.04)),
            max_drawdown_yellow=float(self._raw.get("risk.max_drawdown_yellow", 0.07)),
            max_drawdown_orange=float(self._raw.get("risk.max_drawdown_orange", 0.10)),
            max_positions=int(self._raw.get("risk.max_positions", 50)),
            stop_loss_pct=float(self._raw.get("risk.stop_loss_pct", 0.08)),
            take_profit_pct=float(self._raw.get("risk.take_profit_pct", 0.20)),
            trailing_stop_pct=float(self._raw.get("risk.trailing_stop_pct", 0.05)),
        )
```

- [ ] **Step 2: 导出 TypedConfig**

Edit `stockquant/utils/__init__.py`:

```python
from stockquant.utils.typed_config import TypedConfig, BacktestConfig, DataFetchConfig, RiskConfig
```

并在 `__all__` 中添加 `"TypedConfig"`。

- [ ] **Step 3: 验证可 import 和基本用法**

```bash
.venv/bin/python -c "
from stockquant.utils import TypedConfig
cfg = TypedConfig()
print('Backtest capital:', cfg.backtest.initial_capital)
print('Risk max positions:', cfg.risk.max_positions)
print('Data start:', cfg.data_fetch.start_date)
"
```

Expected: 输出与 `config/default.yaml` 一致的值。

- [ ] **Step 4: Commit**

```bash
git add stockquant/utils/typed_config.py stockquant/utils/__init__.py
git commit -m "feat: add TypedConfig for type-safe config access"
```

---

### Task 4: fetch_daily 去掉远程回退

**Files:**
- Modify: `stockquant/data/data_manager.py:115-147`
- Test: `tests/test_data_manager.py` (若不存在则 create)

**Interfaces:**
- Consumes: 无
- Produces: `fetch_daily()` 只查本地 DB，数据缺失返回空 DataFrame

- [ ] **Step 1: 修改 fetch_daily**

Edit `stockquant/data/data_manager.py` lines 115-147，替换 `fetch_daily` 方法体：

```python
def fetch_daily(
    self,
    code: str,
    start_date: str | dt.date | None = None,
    end_date: str | dt.date | None = None,
    adjust: str | None = None,
) -> pd.DataFrame:
    """获取日线数据（仅本地 DB，不触发远程下载）。

    如需更新数据，请先运行 DataUpdater 同步到本地。
    """
    code = normalize_stock_code(code)
    start_date = ensure_date(start_date or self.cfg.get("data_fetch.start_date"))
    end_date = ensure_date(end_date) or dt.date.today()
    adjust = adjust or self.cfg.get("data_fetch.adjust", "hfq")

    local_df = self._load_local_daily(code, start_date, end_date)
    if local_df.empty:
        return pd.DataFrame()

    latest = local_df["date"].max()
    latest_date = latest.date() if hasattr(latest, "date") else latest
    if latest_date < end_date:
        logger.debug(
            f"{code}: 本地数据截止 {latest_date}，"
            f"早于请求 {end_date}。运行 DataUpdater 获取增量。"
        )

    return apply_price_adjustment(local_df, method=adjust)
```

注意：不再调用 `self._fetch_and_persist_daily()`。`_fetch_and_persist_daily` 方法（line 286）**保留不动**，`DataUpdater` 未来可能需要。

- [ ] **Step 2: 添加测试**

Create/modify `tests/test_data_manager.py`:

```python
import pandas as pd
import pytest
from stockquant.data.data_manager import DataManager


class TestFetchDailyLocalOnly:
    """fetch_daily 只读本地 DB，不触发远程下载。"""

    def test_returns_empty_for_nonexistent_code(self):
        dm = DataManager()
        df = dm.fetch_daily("999999", "2010-01-01", "2010-12-31")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_returns_data_for_existing_code(self):
        dm = DataManager()
        df = dm.fetch_daily("000001", "2010-01-01", "2010-12-31")
        assert isinstance(df, pd.DataFrame)
        # 000001 should exist in local DB
        assert not df.empty, "000001 should have data in local DB"
        assert "close" in df.columns
```

- [ ] **Step 3: 运行测试**

```bash
.venv/bin/python -m pytest tests/test_data_manager.py -v
```

Expected: PASS（若有 000001 在 DB 中）

- [ ] **Step 4: 确保现有测试仍然通过**

```bash
.venv/bin/python -m pytest tests/test_universe.py tests/test_updater.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add stockquant/data/data_manager.py tests/test_data_manager.py
git commit -m "fix: fetch_daily only reads local DB, never triggers remote download"
```

---

### Task 5: StockUniverse.load() 上市日期预过滤

**Files:**
- Modify: `stockquant/data/universe.py:440-470` (load method)
- Modify: `stockquant/data/universe.py` (add `_filter_by_list_date` method)
- Test: `tests/test_universe.py`

**Interfaces:**
- Consumes: `stock_info` 表的 `list_date`, `out_date` 列
- Produces: `_filter_by_list_date(codes, start, end) -> list[str]` — 过滤后的代码列表

- [ ] **Step 1: 添加 _filter_by_list_date 方法**

在 `StockUniverse` 类中添加（与 `_apply_pool_exclude` 附近，line ~517）：

```python
def _filter_by_list_date(
    self, codes: list[str], start: str, end: str
) -> list[str]:
    """排除在请求区间内不可能有数据的股票。

    利用 stock_info 的 list_date / out_date 进行过滤：
    - 上市日在 end 之后 → 跳过（未上市）
    - 退市日在 start 之前 → 跳过（已退市）
    - stock_info 不存在或筛选后为空 → 回退不过滤
    """
    try:
        df = self._dm.get_stock_info(codes)
        if df.empty or "list_date" not in df.columns:
            return codes

        end_dt = pd.Timestamp(end)
        start_dt = pd.Timestamp(start)

        valid: set[str] = set()
        for _, row in df.iterrows():
            code = str(row["code"]).zfill(6)
            # 未上市（上市日晚于请求结束日）
            ld = row.get("list_date")
            if pd.notna(ld) and pd.Timestamp(ld) > end_dt:
                continue
            # 已退市（退市日早于请求起始日）
            od = row.get("out_date")
            if pd.notna(od) and pd.Timestamp(od) < start_dt:
                continue
            valid.add(code)

        before = len(codes)
        filtered = [c for c in codes if c in valid]
        skipped = before - len(filtered)
        if skipped:
            logger.info(
                f"上市日期预过滤: {before} → {len(filtered)}"
                f"（跳过 {skipped} 只区间外股票）"
            )
        return filtered
    except Exception:
        # 回退：任何异常都不应阻断加载流程
        return codes
```

确保文件顶部 `import pandas as pd` 已存在。

- [ ] **Step 2: 在 load() 中插入过滤调用**

Edit `stockquant/data/universe.py` 的 `load()` 方法，在 `final_codes = self.codes()` 之后、`# 默认从配置读取复权方式` 之前（line ~442）插入：

```python
# --- 预过滤：利用 stock_info 排除区间外的股票 ---
final_codes = self._filter_by_list_date(final_codes, start_date, end_date)
```

- [ ] **Step 3: 添加测试**

在 `tests/test_universe.py` 中添加：

```python
class TestListDateFilter:

    def test_filters_unlisted_stocks(self, mocker):
        """上市日在 end_date 之后的股票应被过滤掉。"""
        from stockquant.data.universe import StockUniverse, Pool

        mock_info = pd.DataFrame({
            "code": ["000001", "999999"],
            "list_date": [pd.Timestamp("1991-04-03"), pd.Timestamp("2025-01-01")],
            "out_date": [None, None],
        })

        uni = StockUniverse()
        mocker.patch.object(uni._dm, "get_stock_info", return_value=mock_info)

        result = uni._filter_by_list_date(
            ["000001", "999999"],
            "2010-01-01", "2016-12-31"
        )
        assert "000001" in result
        assert "999999" not in result  # 2016 年还未上市

    def test_filters_delisted_stocks(self, mocker):
        """退市日在 start_date 之前的股票应被过滤掉。"""
        from stockquant.data.universe import StockUniverse

        mock_info = pd.DataFrame({
            "code": ["000001", "000888"],
            "list_date": [pd.Timestamp("1991-04-03"), pd.Timestamp("2000-01-01")],
            "out_date": [None, pd.Timestamp("2005-12-31")],
        })

        uni = StockUniverse()
        mocker.patch.object(uni._dm, "get_stock_info", return_value=mock_info)

        result = uni._filter_by_list_date(
            ["000001", "000888"],
            "2010-01-01", "2016-12-31"
        )
        assert "000001" in result
        assert "000888" not in result  # 2005年已退市

    def test_passthrough_when_no_stock_info(self, mocker):
        """stock_info 为空时应不过滤。"""
        from stockquant.data.universe import StockUniverse

        uni = StockUniverse()
        mocker.patch.object(uni._dm, "get_stock_info", return_value=pd.DataFrame())

        result = uni._filter_by_list_date(
            ["000001", "000002"],
            "2010-01-01", "2016-12-31"
        )
        assert result == ["000001", "000002"]
```

- [ ] **Step 4: 运行新测试**

```bash
.venv/bin/python -m pytest tests/test_universe.py::TestListDateFilter -v
```

Expected: 3 PASS

- [ ] **Step 5: 运行全部 universe 测试**

```bash
.venv/bin/python -m pytest tests/test_universe.py -v
```

Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add stockquant/data/universe.py tests/test_universe.py
git commit -m "feat: StockUniverse.load filters by stock_info.list_date to skip unlisted stocks"
```

---

## Phase 1 完成验证

全部 5 个 Task 完成后：

```bash
# 全部数据相关测试
.venv/bin/python -m pytest tests/test_config.py tests/test_data_manager.py tests/test_universe.py tests/test_updater.py tests/test_data_cleaner.py -v

# 快速加载验证：不应出现 TickFlow 限速告警
.venv/bin/python -c "
from stockquant.data.universe import Pool, StockUniverse
ds = StockUniverse().scope(Pool.ALL_A).exclude(Pool.STAR, Pool.BSE).load('2010-01-01', '2016-12-31')
print(ds.summary())
"
```

Expected: 加载完成，日志中显示 "上市日期预过滤: 4708 → ~2800"，无 TickFlow rate limit Warning。
