"""
全量 Alpha101 IC 汇总示例 — 中证800 (HS300 ∪ CSI500) 2020-至今。

复用框架公开 API:
  StockUniverse.load()        → 从本地 DuckDB 加载多股票面板
  AlphaResearcher.ic_summary  → 向量化批量 IC 评估

前置：本地 daily_bars 表已通过 ``python -m stockquant.data.updater --mode index ...`` 入库。
"""
from __future__ import annotations

from pathlib import Path

from stockquant.data.universe import Pool, StockUniverse
from stockquant.research import AlphaResearcher

START = "2020-01-01"
END = "2026-06-03"
OUT_DIR = Path("scripts/alpha101_out")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = (
        StockUniverse()
        .scope(Pool.CSI300, Pool.CSI500)
        .load(START, END, benchmark=Pool.CSI300)
    )

    researcher = AlphaResearcher(dataset)
    summary = researcher.ic_summary()

    out_path = OUT_DIR / "alpha101_summary.csv"
    summary.to_csv(out_path, index=False)

    print("\n========== 总览 ==========")
    print(f"评估因子数:           {len(summary)}")
    print(f"覆盖率均值:           {summary['coverage'].mean():.2%}")
    print(f"覆盖率 < 50% 因子:    {(summary['coverage'] < 0.5).sum()} 个")
    print(f"|IC|>0.02 因子数:     {(summary['ic_mean'].abs() > 0.02).sum()} 个")
    print(f"|IC|>0.05 因子数:     {(summary['ic_mean'].abs() > 0.05).sum()} 个")

    print("\n========== |IC| Top 15 ==========")
    print(summary.head(15).to_string(index=False))

    print(f"\n[输出] {out_path}")


if __name__ == "__main__":
    main()
