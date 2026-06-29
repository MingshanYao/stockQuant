# stockQuant 信号数据层设计

日期: 2026-06-29
来源: a-stock-data skill 集成

## 背景

当前 `stockquant/data/` 覆盖行情（K线）和基础财务，但缺少量化策略常用的信号/另类数据。
a-stock-data skill 提供十层数据源，其中大量端点可直连 HTTP API、零鉴权，具备集成条件。

## 设计原则

- **独立层，不侵入现有 `data/`**：`signals/` 与 `BaseDataSource` 体系平行，不修改抽象接口
- **纯函数，notebook 友好**：每个业务模块暴露少量函数，`(code, ...) -> pd.DataFrame | list[dict]`
- **东财走统一限流**：所有 `eastmoney.com` 请求经 `_eastmoney.em_get()`，串行 ≥1s + 随机抖动
- **零鉴权优先**：除 iwencai（语义搜索需要 API Key）外全部免费无 key
- **不重复造轮子**：行情（K线/PE/PB）和基础财务已有 BaoStock/TickFlow，signals 层仅补增量

## 模块结构

```
stockquant/signals/
├── __init__.py              # 公开 API 统一导出
├── _eastmoney.py            # em_get 限流, em_datacenter, Session, UA
├── _mootdx.py               # tdx_client() — 规避 mootdx 0.11.x BESTIP bug
├── _sina.py                 # 新浪 GBK 解析 + Referer header
├── _cninfo.py               # 巨潮 orgId 动态映射 + 公告查询 helper
│
├── quote.py                 # mootdx 五档盘口/逐笔成交 + 腾讯PE/PB/涨跌停 + 百度K线MA
├── research.py              # 东财个股/行业研报+PDF下载 + 同花顺一致预期EPS
├── iwencai.py               # iwencai NL语义搜索研报（需 API Key）
├── hot.py                   # 同花顺当日强势股 + 题材归因 reason tags
├── northbound.py            # 北向资金（沪股通/深股通分钟 + 本地缓存日级历史）
├── concept.py               # 个股板块/概念归属（东财 slist, spt=3）
├── fund_flow.py             # 资金流向（分钟级 push2 + 120日日级 push2his）
├── dragon_tiger.py          # 龙虎榜（个股席位 + 全市场榜单 + 机构动向）
├── lockup.py                # 限售解禁日历（历史 + 未来90天）
├── industry.py              # 行业板块排名（全市场涨跌/涨跌家数/领涨股）
├── margin.py                # 融资融券（日级明细）
├── block_trade.py           # 大宗交易（成交价量 + 营业部 + 溢价率）
├── holders.py               # 股东户数变化（筹码集中度）
├── dividend.py              # 分红送转历史
├── news.py                  # 东财个股新闻 + 东财全球资讯（7×24快讯）
├── finance.py               # mootdx 季报快照(37字段) + F10(9类文本) + 新浪三表
├── announcement.py          # 巨潮公告全文检索
├── limit_up.py              # 涨停/炸板/跌停/昨涨停池 + 涨停原因 + 打板情绪速算
├── options.py               # ETF期权（合约清单 + T型报价 + 希腊字母 + IV）
└── sentiment.py             # 互动易问答 + 同花顺热榜 + 东财人气榜 + 概念命中
```

## 共享基础设施

### `_eastmoney.py` — 东财统一请求

```python
# 模块级 Session + 限流，所有东财端点共用
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": UA})
_EM_MIN_INTERVAL = 1.0       # 批量场景可调大到 1.5~2s

em_get(url, params, headers, timeout)   # 带限流的 GET，东财端点专用
em_datacenter(report_name, **kw)        # datacenter-web 通用查询模板
```

共享 RPT 名称表（`em_datacenter` 的 `report_name` 参数）：

| RPT 名称 | 用途 | 使用模块 |
|----------|------|---------|
| `RPT_DAILYBILLBOARD_DETAILSNEW` | 龙虎榜上榜记录 | dragon_tiger |
| `RPT_BILLBOARD_DAILYDETAILSBUY/SELL` | 龙虎榜买卖席位 | dragon_tiger |
| `RPT_LIFT_STAGE` | 限售解禁 | lockup |
| `RPTA_WEB_RZRQ_GGMX` | 融资融券明细 | margin |
| `RPT_DATA_BLOCKTRADE` | 大宗交易 | block_trade |
| `RPT_HOLDERNUMLATEST` | 股东户数 | holders |
| `RPT_SHAREBONUS_DET` | 分红送转 | dividend |

### `_mootdx.py` — 通达信 TCP 客户端

```python
# 顺序探测可用服务器 → 回退 bestip → 回退裸 factory → RuntimeError
# 规避 mootdx 0.11.x BESTIP.HQ 空串导致的 ValueError
tdx_client(market='std') -> Quotes
```

### `_sina.py` — 新浪数据 helper

```python
# GBK 编码，逗号分隔，去 "var hq_str_XXX=\"...\"" 壳
# 共享 Referer header: https://stock.finance.sina.com.cn/
_sina_opt_list(param) -> list[str]
```

### `_cninfo.py` — 巨潮公告 helper

```python
# 模块级缓存 orgId 映射表（szse_stock.json, ~6200只）
# 动态查官方映射优先，硬编码 fallback（修复 601xxx 公告为空的问题）
_cninfo_orgid(code) -> str
```

## 第一期实现范围：资金面/筹码层

按用户确认的优先级，第一期覆盖四个模块：

| 模块 | 函数 | 数据源 | 形态 |
|------|------|--------|------|
| `fund_flow.py` | `get_fund_flow(code, days=120)` | 东财 push2his | 日级时序 |
| `margin.py` | `get_margin_trading(code, page_size=30)` | 东财 datacenter | 日级时序 |
| `northbound.py` | `get_northbound_realtime()` / `get_history(n=20)` | 同花顺 hsgtApi | 分钟/日级时序 |
| `holders.py` | `get_holder_changes(code, periods=10)` | 东财 datacenter | 季频时序 |

### `fund_flow.py` — 资金流向

```python
get_fund_flow(code: str, days: int = 120) -> pd.DataFrame
```
- 端点: `push2his.eastmoney.com/api/qt/stock/fflow/daykline/get`
- 列: date, main_net, small_net, mid_net, large_net, super_net
- 单位: 元
- 后续扩展: 分钟级 `get_fund_flow_minute(code)` → `push2.eastmoney.com`

### `margin.py` — 融资融券

```python
get_margin_trading(code: str, page_size: int = 30) -> pd.DataFrame
```
- 端点: `datacenter-web` → `RPTA_WEB_RZRQ_GGMX`
- 列: date, rzye, rzmre, rzche, rqye, rqmcl, rqchl, rzrqye
- 融资净买入 = rzmre - rzche

### `northbound.py` — 北向资金

```python
get_northbound_realtime() -> pd.DataFrame        # 当日分钟级
get_northbound_history(n: int = 20) -> pd.DataFrame  # 日级缓存
```
- 端点: `data.hexin.cn/market/hsgtApi/`（同花顺，不走东财限流）
- 列: time, hgt_yi, sgt_yi（单位：亿元）
- 缓存: `~/.tradingagents/cache/northbound_daily.csv`，每次拉实时后自动写

### `holders.py` — 股东户数

```python
get_holder_changes(code: str, periods: int = 10) -> pd.DataFrame
```
- 端点: `datacenter-web` → `RPT_HOLDERNUMLATEST`
- 列: date, holder_num, change_num, change_ratio, avg_shares
- 连续多期 holder_num 下降 = 筹码集中信号

## `__init__.py` 公开导出

```python
from stockquant.signals.fund_flow import get_fund_flow
from stockquant.signals.margin import get_margin_trading
from stockquant.signals.northbound import get_northbound_realtime, get_northbound_history
from stockquant.signals.holders import get_holder_changes
```

notebook 用法: `from stockquant.signals import get_fund_flow` 即可。

## 后续各期规划

- **第二期**（事件型）: dragon_tiger, lockup, block_trade, dividend
- **第三期**（截面/文本型）: research, hot, concept, limit_up, industry, sentiment
- **第四期**（行情/财务补充 + ETF/公告）: quote, finance, announcement, options, news, iwencai

每期独立可测、可合入，不依赖后续期。

## 与现有系统的关系

- **不修改** `BaseDataSource` 抽象接口
- **不修改** `DataManager` / `DataUpdater` / `Database`
- **不引入新依赖**：复用现有 `requests`, `pandas`, `mootdx`
- `_eastmoney.em_get` 的限流机制与 `RateLimiter` 平行但独立（东财需要串行 1s+，现有 RateLimiter 设计不同）

## 测试策略

- 每个模块单元测试：用公开数据（茅台/平安等）验证返回非空、列完整
- `_eastmoney.em_datacenter` 集成测试：覆盖所有 RPT 名称
- 不 mock 外部 API — 测试直连真实端点（限流控制在测试间隔内）
