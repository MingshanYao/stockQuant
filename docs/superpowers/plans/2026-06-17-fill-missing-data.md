# 填充缺失的行业/市值/基准收益数据 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 填充 `stock_info` 表（行业/市值）并传入基准收益，使 FactorEvaluator 拥有完整的 BARRA 风格因子（Beta + Vol + Mom + 行业哑变量 + 市值因子）。

**Architecture:** 通过 `DataUpdater.update_stock_info()` 从 AkShare 拉取全市场快照填充 `stock_info` 表；从已有 `index_daily` 表计算基准收益；修改复现脚本和 strict_profile 脚本将 industry/market_cap/benchmark_returns 传入 FactorEvaluator。

**Tech Stack:** Python, AkShare, DuckDB, pandas

---

## 背景

当前数据库状态：
- `stock_info` 表：**0 行**（从未填充）
- `daily_bars` 表：2010-2016，2255 只股票（无市值列，设计如此）
- `index_daily` 表：000905 中证500 有 3247 行，覆盖 2010-01-04 ~ 2026-06-02

当前 `_run_replication.py` 第 158 行：
```python
evaluator = FactorEvaluator(close_panel=engine.close)
```
没有传入 `industry`、`market_cap`、`benchmark_returns`，导致风格中性化只有 Vol+Mom 两个因子。

---

### Task 1: 填充 stock_info 表

**Files:**
- Run: CLI 命令，不修改文件

- [ ] **Step 1: 运行 DataUpdater.update_stock_info()**

```bash
cd /Users/mingshan.yao/stockQuant && .venv/bin/python -c "
from stockquant.data.updater import DataUpdater
u = DataUpdater()
n = u.update_stock_info()
print(f'写入 {n} 只股票')
"
```

预期：从 AkShare 拉取全市场股票快照（行业板块映射 + 市值数据），写入 `stock_info` 表。耗时约 1-3 分钟（取决于行业板块数量，约 90 个板块并发拉取）。

- [ ] **Step 2: 验证 stock_info 表已填充**

```bash
cd /Users/mingshan.yao/stockQuant && .venv/bin/python -c "
from stockquant.data.database import Database
db = Database()
n = db.conn.execute('SELECT COUNT(*) FROM stock_info').fetchone()[0]
print(f'stock_info 总行数: {n}')
ni = db.conn.execute('SELECT COUNT(*) FROM stock_info WHERE industry IS NOT NULL AND industry != \'其他\'').fetchone()[0]
print(f'有行业数据: {ni}')
nc = db.conn.execute('SELECT COUNT(*) FROM stock_info WHERE total_cap IS NOT NULL').fetchone()[0]
print(f'有市值数据: {nc}')
db.close()
"
```

预期：stock_info > 4000 行，行业覆盖率 > 95%，市值数据有效。

---

### Task 2: 修改 _run_replication.py 传入 industry/market_cap/benchmark_returns

**Files:**
- Modify: `notebooks/alpha191/_run_replication.py`

- [ ] **Step 1: 在第 4 章 FactorEvaluator 构造前添加数据加载逻辑**

在第 158 行的 `evaluator = FactorEvaluator(close_panel=engine.close)` 之前，插入以下代码加载行业、市值和基准收益：

```python
# ── 加载行业 & 市值数据（来自 stock_info 表）──
stock_info = db.query("SELECT code, industry, total_cap, float_cap FROM stock_info")
if not stock_info.empty:
    industry_map = stock_info.set_index("code")["industry"]
    # 用 float_cap 作为市值代理（静态快照近似）
    market_cap = stock_info.set_index("code")["float_cap"]
    print(f"  行业映射: {industry_map.nunique()} 个行业, {len(industry_map)} 只股票")
    print(f"  市值数据: {market_cap.notna().sum()} 只股票")
else:
    industry_map = None
    market_cap = None
    print("  ⚠️ stock_info 为空，无法加载行业/市值")

# ── 加载基准收益（用于 Beta 计算）──
benchmark_bars = db.query(
    "SELECT date, close FROM index_daily WHERE code = '000905'"
    " AND date >= '2010-01-01' AND date <= '2016-12-31' ORDER BY date"
)
benchmark_bars["date"] = pd.to_datetime(benchmark_bars["date"])
benchmark_returns = benchmark_bars.set_index("date")["close"].pct_change()
print(f"  基准收益: {len(benchmark_returns.dropna())} 个交易日")
```

- [ ] **Step 2: 修改 FactorEvaluator 构造传入新参数**

将第 158 行的：
```python
evaluator = FactorEvaluator(close_panel=engine.close)
```
改为：
```python
evaluator = FactorEvaluator(
    close_panel=engine.close,
    industry=industry_map,
    market_cap=market_cap,
    benchmark_returns=benchmark_returns,
)
```

---

### Task 3: 修改 _strict_profile.py 传入 industry/market_cap/benchmark_returns

**Files:**
- Modify: `notebooks/alpha191/_strict_profile.py`

- [ ] **Step 1: 在 Phase 2 前添加数据加载**

在第 110 行的 `evaluator = FactorEvaluator(close_panel=engine.close)` 之前，插入相同的加载逻辑：

```python
# ── 加载行业 & 市值 & 基准收益 ──
stock_info = db.query("SELECT code, industry, float_cap FROM stock_info")
if not stock_info.empty:
    industry_map = stock_info.set_index("code")["industry"]
    market_cap = stock_info.set_index("code")["float_cap"]
else:
    industry_map = None
    market_cap = None

benchmark_bars = db.query(
    "SELECT date, close FROM index_daily WHERE code = '000905'"
    " AND date >= ? AND date <= ? ORDER BY date",
    [START_DATE, END_DATE],
)
benchmark_bars["date"] = pd.to_datetime(benchmark_bars["date"])
benchmark_returns = benchmark_bars.set_index("date")["close"].pct_change()
```

- [ ] **Step 2: 修改 FactorEvaluator 构造**

将：
```python
evaluator = FactorEvaluator(close_panel=engine.close)
```
改为：
```python
evaluator = FactorEvaluator(
    close_panel=engine.close,
    industry=industry_map,
    market_cap=market_cap,
    benchmark_returns=benchmark_returns,
)
```

---

### Task 4: 修改 _benchmark_full.py 传入 industry/market_cap/benchmark_returns

**Files:**
- Modify: `notebooks/alpha191/_benchmark_full.py`

- [ ] **Step 1: 在第 71 行前添加加载逻辑并修改构造**

在第 71 行的 `evaluator = FactorEvaluator(close_panel=engine.close)` 之前插入：

```python
# ── 加载行业 & 市值 & 基准收益 ──
stock_info_df = db.query("SELECT code, industry, float_cap FROM stock_info")
if not stock_info_df.empty:
    industry_map = stock_info_df.set_index("code")["industry"]
    market_cap = stock_info_df.set_index("code")["float_cap"]
else:
    industry_map = None
    market_cap = None

benchmark_bars = db.query(
    "SELECT date, close FROM index_daily WHERE code='000905'"
    " AND date>='2010-01-01' AND date<='2016-12-31' ORDER BY date"
)
benchmark_bars["date"] = pd.to_datetime(benchmark_bars["date"])
benchmark_returns = benchmark_bars.set_index("date")["close"].pct_change()
```

然后将第 71 行的 `evaluator = FactorEvaluator(close_panel=engine.close)` 改为：
```python
evaluator = FactorEvaluator(
    close_panel=engine.close,
    industry=industry_map,
    market_cap=market_cap,
    benchmark_returns=benchmark_returns,
)
```

---

### Task 5: 运行 strict_profile 验证

- [ ] **Step 1: 运行 strict_profile**

```bash
cd /Users/mingshan.yao/stockQuant && .venv/bin/python notebooks/alpha191/_strict_profile.py
```

预期：
- FactorEvaluator 构造时间略增（Beta 计算需并行处理 200 只股票的滚动回归）
- Phase 2 eval_system 时间可能增加（因为有行业哑变量，中性化回归变量增多）
- model_predictive_power 的 IC 不应再为 NaN
- 总 wall-clock 时间应在 30-40 秒内

- [ ] **Step 2: 检查关键输出**

确认以下输出非 NaN：
- `model_predictive_power: IC=X.XXXX`（非 nan）
- `factor_count_analysis` 中的 ic_mean/ic_ir 非 NaN
- `factor_correlation_matrix` 正常计算

---

### Task 6: 运行完整复现验证

- [ ] **Step 1: 运行 _run_replication.py**

```bash
cd /Users/mingshan.yao/stockQuant && .venv/bin/python notebooks/alpha191/_run_replication.py
```

注意：完整复现约需 46 分钟（190 因子 × 2255 股 × 1700 天）。

- [ ] **Step 2: 检查关键输出**

确认：
- `合成 Alpha 预测 IC: 均值=X.XXXX` 非 NaN
- `factor_count_analysis` 结果非 NaN
- 回测结果正常产出

- [ ] **Step 3: 提交**

```bash
cd /Users/mingshan.yao/stockQuant
git add notebooks/alpha191/_run_replication.py \
        notebooks/alpha191/_strict_profile.py \
        notebooks/alpha191/_benchmark_full.py
git commit -m "feat: pass industry, market_cap, and benchmark_returns to FactorEvaluator"
```
