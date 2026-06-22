# stockQuant 架构重构设计

日期: 2026-06-22

## 背景

架构审查发现 14 个问题，按依赖关系分 4 个阶段重构。每个阶段独立可测、可合入。

---

## 阶段 1：基础设施（Config + Data 层）

### 1.1 Config 修复

**1.1a 移除泄露的 API Key**

`config/default.yaml` 第 17 行 `tickflow_api_key: "tk_6350..."` → 删除。`source_tickflow.py` 改为从环境变量 `SQ_TICKFLOW_API_KEY` 读取（与 `source_itick.py` / `source_tushare.py` 一致）。

**1.1b TypedConfig 封装**

新增 `stockquant/utils/typed_config.py`：

```python
@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    commission: float = 0.0003
    ...

class TypedConfig:
    """Config 的类型安全包装，提供属性访问和 IDE 自动补全。"""
    def __init__(self, raw: Config):
        self._raw = raw

    @cached_property
    def backtest(self) -> BacktestConfig:
        return BacktestConfig(
            initial_capital=float(self._raw.get("backtest.initial_capital", 1_000_000.0)),
            ...
        )
```

`TypedConfig` 从 `Config` 读取一次，转为 dataclass。调用方用 `cfg.backtest.initial_capital` 替代 `cfg.get("backtest.initial_capital", 1000000)`。不强制所有模块立即迁移——新旧可共存，渐进替换。

**1.1c 修复单例陷阱**

`Config.__init__` 在已初始化后若收到不同的 `config_path`，抛 `RuntimeError` 而非静默忽略。

### 1.2 Data 层修复

**1.2a fetch_daily 去掉远程回退**

`DataManager.fetch_daily()`: 删掉 `_fetch_and_persist_daily()` 调用。`_fetch_and_persist_daily` 方法本身保留（`DataUpdater` 未来可复用），仅移除 fetch_daily 中的调用路径。

改动后逻辑：
```
_load_local_daily(code, start, end) → return apply_price_adjustment(df)
                                    → 空则返回空 DataFrame
```

**1.2b StockUniverse.load() 上市日期预过滤**

在 `load()` 的 `final_codes = self.codes()` 之后、循环之前插入：

```python
final_codes = self._filter_by_list_date(final_codes, start_date, end_date)
```

`_filter_by_list_date()` 查询 `stock_info.list_date` / `out_date`：
- `list_date > end_date` → 跳过（未上市）
- `out_date < start_date` → 跳过（已退市）
- `stock_info` 不存在或列为空 → 回退不过滤

加载日志从 `[i/4708]` 变为 `[i/~2800]`（对 2010-2016 区间）。

**1.2c DataManager / DataUpdater 共享 Database**

`DataManager.__init__` 接受可选 `db: Database | None` 参数（与 `DataUpdater.__init__` 一致）。默认仍创建新实例。

### 1.3 验证

- `pytest tests/test_universe.py tests/test_data_cleaner.py -v` 通过
- `StockUniverse().scope(Pool.ALL_A).load("2010-01-01", "2016-12-31")` 无 TickFlow 告警
- API Key 不再出现在 `default.yaml` 中

---

## 阶段 2：解耦 core 域模型

### 2.1 新建 stockquant/core/

```python
# stockquant/core/__init__.py — 以下类型从 core 统一导出

# Entity
from stockquant.core.entity import Order, Position, Trade

# Market
from stockquant.core.market import Bar, Portfolio, LightweightBar
```

**2.1a `stockquant/core/entity.py`** — 从 `strategy/base_strategy.py` 迁出 `Order`, `Position`

**2.1b `stockquant/core/market.py`** — 从 `backtest/bar.py` 迁出 `Bar`, `LightweightBar`；从 `backtest/context.py` 迁出 `Portfolio`

### 2.2 消除循环依赖

**改动前**:
```
strategy/base_strategy.py → backtest/context.py (Position)
backtest/context.py        → strategy/base_strategy.py (Context)
backtest/broker.py         → strategy/base_strategy.py (Order, Position)
risk/stop_loss.py          → strategy/base_strategy.py (Position)
```

**改动后**:
```
所有模块 → stockquant/core  (单向依赖)
strategy ← → backtest       (不再直接互相引用)
strategy ← → risk           (不再直接互相引用)
```

具体操作：
- `strategy/base_strategy.py` 中的 `Order`, `Position` 类 → 移入 `core/entity.py`，原位置保留 re-export（兼容）
- `backtest/context.py` 中的 `Portfolio` → 移入 `core/market.py`
- `backtest/bar.py` → 移入 `core/market.py`
- `backtest/context.py` 对 `Position` 的引用 → 改为从 `stockquant.core` import

### 2.3 验证

- `pytest tests/ -v -k "backtest or strategy or risk"` 全部通过
- 无新增循环 import

---

## 阶段 3：拆分 God 模块

### 3.1 AlphaResearcher (880 行 → ~300 行)

| 移出内容 | 目标 |
|----------|------|
| `plot_ic_decay()`, `plot_factor_returns()`, `plot_backtest_curves()` 等绘图方法 | `stockquant/visualization/research_plots.py` |
| 内联 matplotlib 代码 | 同上 |
| 保留：`get_alpha_panel()`, `run_backtest()`, `metrics_table()`, `evaluate_factors()` | `alpha_researcher.py` |

### 3.2 DataUpdater (902 行 → ~300 行)

| 拆出 | 目标 |
|------|------|
| `update_all_daily()`, `update_codes_daily()` + 相关私有方法 | `stockquant/data/updaters/daily_updater.py` |
| `update_index_daily()`, `update_benchmark_indices()` | `stockquant/data/updaters/index_updater.py` |
| `update_stock_info()`, `update_market_cap()` | `stockquant/data/updaters/info_updater.py` |
| `update_financials()` | `stockquant/data/updaters/financials_updater.py` |
| CLI 入口 | 保留在 `updater.py`，委托给各子 updater |

### 3.3 Alpha191 (1674 行 → 拆文件)

按因子编号分组, 沿用现有 `def alpha030(self)` → `engine.compute_factor(30)` 分发模式：

```
indicators/alpha191/
    __init__.py
    alpha191.py          # Alpha191Engine + Alpha191Indicators + compute_factor/ALL
    factors_001_050.py   # alpha001–alpha050
    factors_051_100.py   # alpha051–alpha100
    factors_101_150.py   # alpha101–alpha150
    factors_151_191.py   # alpha151–alpha191
    operators.py         # 不变
```

`Alpha191Engine` 中每个 `alphaXXX()` 方法改为从分组文件 import 的静态方法。对外 API 完全不变。

### 3.4 FactorEvaluator (1106 行 → ~400 行)

| 拆出 | 目标 |
|------|------|
| 中性化逻辑 `_neutralize_panel()`, `_neutralize_panels_batch()` | `analysis/neutralizer.py` |
| Decile 分析 `decile_analysis()` | `analysis/decile.py` |
| 保留：`evaluate()`, `evaluate_system()`, `factor_correlation_matrix()`, `factor_count_analysis()` | `evaluator.py` |

### 3.5 验证

- `pytest tests/ -v` 全部通过
- 各子模块可独立 import 和测试

---

## 阶段 4：工程化完善

### 4.1 CLI 入口

`pyproject.toml` 加：
```toml
[project.scripts]
stockquant-update = "stockquant.data.updater:main"
stockquant-backtest = "stockquant.cli.backtest:main"
```

### 4.2 指标层去 DB 依赖

`alpha101.py:404` 中 `from stockquant.data.database import Database` → 改为接受调用方传入已加载的数据。

### 4.3 清理硬编码默认值

搜索全部 `cfg.get("...", "2020-01-01")` → 删除 fallback，改为直接用 `cfg.get("...")`（默认值已在 YAML 中定义）。若 YAML 真缺了 key，抛明确错误。

### 4.4 架构文档

`docs/architecture.md`：模块依赖图、数据流、关键类职责说明。

### 4.5 验证

- `pip install -e .` 后 `stockquant-update --help` 可用
- 全量测试通过
- `docs/architecture.md` 可读

---

## 总体验证

全部阶段完成后：
```bash
pytest tests/ -v                    # 全部通过
python notebooks/alpha191/_run_replication.py  # 正常运行，无 TickFlow 限速告警
```
