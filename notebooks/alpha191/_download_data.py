"""批量下载 2010-2016 全A股日线数据 + 基准指数（使用 BaoStock）。"""
import sys
sys.path.insert(0, ".")

from stockquant.data.updater import DataUpdater
from stockquant.data.source_baostock import BaoStockDataSource

updater = DataUpdater()
updater._source = BaoStockDataSource()

# 1. 更新 Benchmark 指数日线（沪深300）
print("=" * 60)
print("Step 1: 更新 Benchmark 指数日线 (BaoStock)")
print("=" * 60)
idx_result = updater.update_benchmark_indices(
    start_date="2010-01-01",
    end_date="2016-12-31",
)
for code, rows in idx_result.items():
    print(f"  指数 {code}: {rows} 行")

# 2. 获取全A股代码列表（BaoStock）
print("\n" + "=" * 60)
print("Step 2: 获取全A股代码列表")
print("=" * 60)
source = BaoStockDataSource()
stock_df = source.get_stock_list()
all_codes = stock_df["code"].astype(str).str.zfill(6).tolist()

# 排除科创板(688) 和 北交所(4xx, 8xx except 688)
filtered = [c for c in all_codes if not c.startswith(("688", "43", "83", "87"))]
print(f"全A股: {len(all_codes)} 只 → 排除科创/北交后: {len(filtered)} 只")

# 3. 批量下载日线
print("\n" + "=" * 60)
print("Step 3: 批量下载全A股日线 (2010-2016)")
print("=" * 60)
result = updater.update_codes_daily(
    codes=filtered,
    start_date="2010-01-01",
    end_date="2016-12-31",
)
total = sum(result.values())
success = sum(1 for v in result.values() if v > 0)
print(f"\n✅ 下载完成: {success}/{len(result)} 只股票, 共 {total:,} 条日线")
