"""
因子评价器 — 单因子四指标检验 + 多周期分析 + 因子体系评价。

借鉴国泰君安 Alpha191 报告的评价方法论：
  风格正交化 → IC / Factor Return / IR / T统计量 → 多周期扫描 → 体系相关性

风格因子使用 BARRA CNE5 指数加权算法：
  - Beta: 半衰60日指数加权回归（窗口252日）
  - Volatility: DASTD（半衰40日）+ CMRA + HSIGMA
  - Momentum: RSTR（半衰120日累计对数收益，T=504, L=21）
"""

from __future__ import annotations

import atexit
import os
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from stockquant.analysis.factor import FactorAnalyzer
from stockquant.utils.logger import get_logger

logger = get_logger("analysis.evaluator")

TRADING_DAYS_PER_YEAR = 252

# 模块级全局用于 multiprocessing fork（避免 pickle evaluator）
_evaluator: "FactorEvaluator | None" = None

# 模块级进程池复用——避免每次 neutralize_panel 重新 fork
_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()


def _get_pool(max_workers: int) -> ProcessPoolExecutor:
    """获取或创建模块级持久进程池（fork 模式）。

    仅在 MainProcess 中调用；子进程检测到会走串行路径。
    """
    global _pool
    if _pool is None:
        ctx = multiprocessing.get_context("fork")
        _pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
    return _pool


def close_pool() -> None:
    """关闭模块级进程池，释放 worker 进程。

    在不再需要中性化计算时调用（例如回测前），释放内存。
    """
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=True, cancel_futures=True)
            _pool = None


atexit.register(close_pool)


def _eval_one(item: tuple):
    """模块级函数，用于 ProcessPoolExecutor fork。"""
    name, panel, forward_period, kwargs = item
    return name, _evaluator.evaluate(panel, forward_period=forward_period, **kwargs)


def _vectorized_factor_return(f: np.ndarray, r: np.ndarray) -> np.ndarray:
    """逐行截面因子收益率——纯 numpy 向量化，避免逐日 Python 循环。

    f, r: (n_dates, n_stocks), 形状相同，可能含 NaN。
    返回 (n_dates,) 数组。
    """
    valid = np.isfinite(f) & np.isfinite(r)
    n_valid = valid.sum(axis=1)

    # Z-score f per row (date), using valid-only statistics
    f_mean = np.nanmean(f, axis=1, keepdims=True)
    f_std = np.nanstd(f, axis=1, keepdims=True)
    f_norm = np.where(valid & (f_std > 1e-10), (f - f_mean) / f_std, 0.0)

    # Center r, using valid-only mean
    r_mean = np.nanmean(r, axis=1, keepdims=True)
    r_ctr = np.where(valid, r - r_mean, 0.0)

    # FR = cov(f_norm, r) / var(f_norm) = cov(f_norm, r) since var(f_norm)=1
    cov = (f_norm * r_ctr).sum(axis=1) / np.maximum(n_valid - 1, 1)

    result = np.full(f.shape[0], np.nan)
    ok = n_valid >= 10
    result[ok] = cov[ok]
    return result


def _numpy_pearson_by_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐行 Pearson 相关系数——纯 numpy，释放 GIL 适合多线程。

    a, b: (n_rows, n_cols), 形状相同。NaN 按 pairwise-complete 处理。
    返回 (n_rows,) 数组。当行内有效样本 < 3 时结果为 NaN。
    """
    valid = np.isfinite(a) & np.isfinite(b)
    a_c = np.where(valid, a - np.nanmean(a, axis=1, keepdims=True), 0.0)
    b_c = np.where(valid, b - np.nanmean(b, axis=1, keepdims=True), 0.0)
    num = (a_c * b_c).sum(axis=1)
    den_a = np.sqrt((a_c * a_c).sum(axis=1))
    den_b = np.sqrt((b_c * b_c).sum(axis=1))
    den = den_a * den_b
    result = np.full(a.shape[0], np.nan)
    count = valid.sum(axis=1)
    with np.errstate(invalid="ignore"):
        ok = (count >= 3) & (den > 1e-15)
        result[ok] = num[ok] / den[ok]
    return result


def _exp_weight(n: int, halflife: int) -> np.ndarray:
    """生成长度为 n 的指数衰减权重向量（BARRA 风格）。

    w_t = 0.5^(t / halflife)，归一化使 sum(w) = 1。
    t=0 对应最新观测。
    """
    t = np.arange(n - 1, -1, -1)  # t=0 = 最旧
    raw = 0.5 ** (t / halflife)
    return raw / raw.sum()


def _exp_weighted_mean(series: np.ndarray, window: int, halflife: int) -> np.ndarray:
    """指数加权滑动均值（BARRA 风格）。"""
    w = _exp_weight(window, halflife)
    n = len(series)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = float(np.dot(w, series[i - window + 1: i + 1]))
    return result


def _exp_weighted_std(series: np.ndarray, window: int, halflife: int) -> np.ndarray:
    """指数加权滑动标准差。"""
    w = _exp_weight(window, halflife)
    n = len(series)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        win = series[i - window + 1: i + 1]
        if np.any(np.isnan(win)):
            continue
        mean_w = np.dot(w, win)
        result[i] = float(np.sqrt(np.dot(w, (win - mean_w) ** 2)))
    return result


def _exp_weighted_beta(
    stock_ret: np.ndarray, bench_ret: np.ndarray, window: int, halflife: int,
) -> float | np.ndarray:
    """指数加权回归 Beta（BARRA 风格）。

    返回整个序列的滚动 Beta（与输入同长）。
    """
    w = _exp_weight(window, halflife)
    n = len(stock_ret)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        ys = stock_ret[i - window + 1: i + 1]
        xs = bench_ret[i - window + 1: i + 1]
        mask = ~(np.isnan(ys) | np.isnan(xs))
        if mask.sum() < 30:
            continue
        yv = ys[mask]
        xv = xs[mask]
        wm = w[mask]
        wm = wm / wm.sum()
        mean_y = np.dot(wm, yv)
        mean_x = np.dot(wm, xv)
        cov = np.dot(wm, (yv - mean_y) * (xv - mean_x))
        var = np.dot(wm, (xv - mean_x) ** 2)
        if var > 1e-12:
            result[i] = float(cov / var)
    return result


def _neutralize_single_date(
    y: pd.Series,
    industry: pd.Series | None,
    market_cap: pd.Series | None,
    style_factors: dict[str, pd.Series] | None,
) -> pd.Series:
    """对单日截面做多元回归，取残差作为纯Alpha。

    回归模型：
        y = β_0 + β_ind·X_ind + β_size·X_size + β_beta·X_beta
              + β_vol·X_vol + β_mom·X_mom + ε
    """
    codes = y.dropna().index
    if len(codes) < 20:
        return y

    design = []

    # 行业哑变量
    if industry is not None:
        ind_series = industry.reindex(codes).dropna()
        if len(ind_series) > 0:
            ind_dummies = pd.get_dummies(ind_series, prefix="ind", drop_first=True)
            design.append(ind_dummies)

    # 市值（取对数后 Z-score）
    if market_cap is not None:
        if isinstance(market_cap, pd.Series):
            mc = market_cap.reindex(codes).dropna()
        else:
            mc = market_cap.dropna()
        if len(mc) > 0:
            log_mc = np.log(mc.replace(0, np.nan)).dropna()
            mc_z = (log_mc - log_mc.mean()) / log_mc.std()
            design.append(mc_z.to_frame("size"))

    # 其他风格因子（Beta / Volatility / Momentum）
    if style_factors is not None:
        for name, sf in style_factors.items():
            sf_reindexed = sf.reindex(codes).dropna()
            if len(sf_reindexed) > 0:
                sf_z = (sf_reindexed - sf_reindexed.mean()) / sf_reindexed.std()
                design.append(sf_z.to_frame(name))

    if len(design) < 1:
        return y

    X = pd.concat(design, axis=1).dropna()
    y_aligned = y.reindex(X.index).dropna()
    common = y_aligned.index.intersection(X.index)
    if len(common) < 20:
        return y

    X = X.loc[common]
    y_vals = y_aligned.loc[common].values

    X_aug = np.column_stack([np.ones(len(X)), X.values])

    try:
        with np.errstate(all="ignore"):
            beta = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y_vals)
            if np.any(~np.isfinite(beta)):
                return y
            residuals = y_vals - X_aug @ beta
    except Exception:
        return y

    result = pd.Series(np.nan, index=y.index)
    result.loc[common] = residuals
    return result


def _neutralize_chunk(args: tuple) -> dict:
    """模块级函数：中性化一组日期的因子截面。"""
    factor, dates, industry, market_cap, style_factors = args
    chunk_result = {}
    for date in dates:
        row = factor.loc[date].dropna()
        if len(row) < 20:
            continue
        codes = row.index
        ind = industry.reindex(codes) if industry is not None else None
        if market_cap is not None:
            if isinstance(market_cap, pd.Series):
                mc = market_cap.reindex(codes)
            elif date in market_cap.index:
                mc = market_cap.loc[date].reindex(codes)
            else:
                mc = None
        else:
            mc = None
        style = {}
        if style_factors:
            for name, panel in style_factors.items():
                if date in panel.index:
                    style[name] = panel.loc[date].reindex(codes)
        neutralized = _neutralize_single_date(row, ind, mc, style)
        chunk_result[date] = neutralized
    return chunk_result


def _neutralize_chunk_batch(args: tuple) -> dict:
    """模块级函数：批量中性化多因子在日期块上的截面。

    每日期预计算行业哑变量、市值和风格因子设计矩阵，
    所有因子复用，避免重复 pd.get_dummies。
    """
    panel_dict, dates, industry, market_cap, style_factors = args
    results: dict[str, dict] = {name: {} for name in panel_dict}

    all_stocks = next(iter(panel_dict.values())).columns

    # 行业哑变量 —— 跨日期不变，预计算一次
    ind_dummies_all = None
    if industry is not None:
        ind_series = industry.reindex(all_stocks).dropna()
        if len(ind_series) > 0:
            ind_dummies_all = pd.get_dummies(ind_series, prefix="ind", drop_first=True)

    for date in dates:
        # ── 市值 Z-score（每日期）──
        mc_z = None
        if market_cap is not None:
            if isinstance(market_cap, pd.Series):
                mc_raw = market_cap.reindex(all_stocks).dropna()
            elif date in market_cap.index:
                mc_raw = market_cap.loc[date].reindex(all_stocks).dropna()
            else:
                mc_raw = None
            if mc_raw is not None and len(mc_raw) > 0:
                log_mc = np.log(mc_raw.replace(0, np.nan)).dropna()
                mc_z = (log_mc - log_mc.mean()) / log_mc.std()

        # ── 风格因子 Z-score（每日期）──
        style_z: dict[str, pd.Series] = {}
        if style_factors:
            for sf_name, sf_panel in style_factors.items():
                if date in sf_panel.index:
                    sf_raw = sf_panel.loc[date].reindex(all_stocks).dropna()
                    if len(sf_raw) > 0:
                        style_z[sf_name] = (sf_raw - sf_raw.mean()) / sf_raw.std()

        # ── 构建设计矩阵（全股票，不含截距）──
        design_parts = []
        if ind_dummies_all is not None:
            design_parts.append(ind_dummies_all)
        if mc_z is not None:
            design_parts.append(mc_z.to_frame("size"))
        for sf_name, sf in style_z.items():
            design_parts.append(sf.to_frame(sf_name))

        X_full: pd.DataFrame | None = None
        if design_parts:
            X_full = pd.concat(design_parts, axis=1).dropna()

        # ── 对每个因子做 OLS 残差化 ──
        for name, panel in panel_dict.items():
            if date not in panel.index:
                continue
            y = panel.loc[date].dropna()
            if len(y) < 20:
                continue

            if X_full is None:
                results[name][date] = y
                continue

            common = y.index.intersection(X_full.index)
            if len(common) < 20:
                continue

            X_mat = X_full.loc[common].values
            y_v = y.loc[common].values
            X_aug = np.column_stack([np.ones(len(X_mat)), X_mat])

            try:
                with np.errstate(all="ignore"):
                    beta = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y_v)
                    if np.any(~np.isfinite(beta)):
                        continue
                    residuals = y_v - X_aug @ beta
                    results[name][date] = pd.Series(residuals, index=common)
            except np.linalg.LinAlgError:
                continue

    return results


def _compute_vol_stock(args: tuple) -> tuple:
    """模块级函数：计算单只股票的 DASTD 波动率。"""
    code, ret_arr = args
    vol = _exp_weighted_std(ret_arr, 252, 40) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return code, vol


def _compute_mom_stock(args: tuple) -> tuple:
    """模块级函数：计算单只股票的 RSTR 动量。"""
    code, lr_arr, w_mom = args
    T, L = 504, 21
    rstr = np.full(len(lr_arr), np.nan)
    for i in range(T, len(lr_arr)):
        window_lr = lr_arr[i - T: i - L + 1]
        if np.any(np.isnan(window_lr)):
            continue
        rstr[i] = float(np.dot(w_mom, window_lr))
    return code, rstr


def _compute_beta_stock(args: tuple) -> tuple:
    """模块级函数：计算单只股票的滚动指数加权 Beta。"""
    code, stock_ret, bench_ret = args
    beta = _exp_weighted_beta(stock_ret, bench_ret, 252, 60)
    return code, beta


class FactorEvaluator:
    """因子评价器。

    Parameters
    ----------
    close_panel : DataFrame
        收盘价面板（行=日期，列=股票代码），用于计算前向收益。
    industry : Series, optional
        股票代码 → 行业分类的映射，用于风格正交化。
    market_cap : Series or DataFrame, optional
        股票代码 → 市值的映射（Series），或每日市值面板（DataFrame）。
    style_factors : dict, optional
        额外的风格因子，{name: DataFrame(行=日期, 列=股票代码)}。
        自动计算 beta, volatility, momentum（若未提供且 close_panel 可用）。
    benchmark_returns : Series, optional
        基准日收益率序列，用于计算滚动 Beta。
    """

    def __init__(
        self,
        close_panel: pd.DataFrame,
        industry: pd.Series | None = None,
        market_cap: pd.Series | pd.DataFrame | None = None,
        style_factors: dict[str, pd.DataFrame] | None = None,
        benchmark_returns: pd.Series | None = None,
    ) -> None:
        self.close = close_panel
        self.industry = industry
        self.market_cap = market_cap
        self._style_factors = style_factors or {}
        self.benchmark_returns = benchmark_returns

        # 自动计算未提供的风格因子
        self._compute_missing_style_factors()

    def _compute_missing_style_factors(self) -> None:
        """自动计算缺失的风格因子（BARRA CNE5 算法）—— 并行化。

        - Beta: 半衰60日指数加权回归，窗口252日
        - Volatility (DASTD): 半衰40日指数加权标准差，窗口252日 + CMRA + HSIGMA
        - Momentum (RSTR): 半衰120日累计对数收益，T=504, L=21
        """
        returns = self.close.pct_change()
        codes = self.close.columns.tolist()
        max_workers = min(max(os.cpu_count() - 2, 1), len(codes))
        ctx = multiprocessing.get_context("fork")

        # ── Beta (BARRA: 半衰60日，窗口252日) ──
        if "beta" not in self._style_factors and self.benchmark_returns is not None:
            beta_panel = pd.DataFrame(
                index=self.close.index, columns=self.close.columns, dtype=float,
            )
            bench = self.benchmark_returns.reindex(self.close.index)
            items = [(code, returns[code].values, bench.values) for code in codes]
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(_compute_beta_stock, item): item[0] for item in items
                }
                for future in as_completed(futures):
                    code, beta = future.result()
                    beta_panel[code] = beta
            beta_panel = beta_panel.ffill().bfill()
            self._style_factors["beta"] = beta_panel

        # ── Volatility: DASTD（半衰40日，窗口252日）──
        if "volatility" not in self._style_factors:
            vol_panel = pd.DataFrame(
                index=self.close.index, columns=self.close.columns, dtype=float,
            )
            items = [(code, returns[code].values) for code in codes]
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(_compute_vol_stock, item): item[0] for item in items
                }
                for future in as_completed(futures):
                    code, vol = future.result()
                    vol_panel[code] = vol
            vol_panel = vol_panel.ffill().bfill()
            self._style_factors["volatility"] = vol_panel

        # ── Momentum: RSTR（半衰120日，T=504, L=21）──
        if "momentum" not in self._style_factors:
            log_ret = np.log(1 + returns.clip(lower=-0.5))
            T, L = 504, 21
            w_mom = _exp_weight(T - L + 1, 120)
            mom_panel = pd.DataFrame(
                index=self.close.index, columns=self.close.columns, dtype=float,
            )
            items = [(code, log_ret[code].values, w_mom) for code in codes]
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(_compute_mom_stock, item): item[0] for item in items
                }
                for future in as_completed(futures):
                    code, rstr = future.result()
                    mom_panel[code] = rstr
            mom_panel = mom_panel.ffill().bfill()
            self._style_factors["momentum"] = mom_panel

    def _get_date_style_factors(self, date) -> dict[str, pd.Series]:
        """获取某个日期的所有风格因子截面。"""
        result = {}
        for name, panel in self._style_factors.items():
            if date in panel.index:
                result[name] = panel.loc[date]
        return result

    def _get_date_market_cap(self, date) -> pd.Series | None:
        """获取某个日期的市值截面。"""
        if self.market_cap is None:
            return None
        if isinstance(self.market_cap, pd.Series):
            return self.market_cap  # 静态映射，所有日期相同
        if date in self.market_cap.index:
            return self.market_cap.loc[date]
        return None

    @staticmethod
    def shutdown_pool() -> None:
        """关闭模块级进程池，释放 worker 进程内存。

        在完成所有中性化计算后、进入回测阶段前调用。
        """
        close_pool()

    def forward_returns(self, period: int) -> pd.DataFrame:
        return self.close.pct_change(periods=period).shift(-period)

    def neutralize_panel(self, factor: pd.DataFrame) -> pd.DataFrame:
        """对每日截面做风格中性化，取残差作为纯Alpha因子值（并行化）。

        仅在主进程的主线程中启用并行；ThreadPoolExecutor 工作线程或
        ProcessPoolExecutor 子进程调用时走串行，避免池竞争或嵌套 fork。
        """
        result = factor.copy()
        dates = factor.index.tolist()
        if not dates:
            return result

        is_main = multiprocessing.current_process().name == "MainProcess"
        is_main_thread = threading.current_thread() is threading.main_thread()
        use_parallel = is_main and is_main_thread and len(dates) >= 60

        if not use_parallel:
            for date in dates:
                row = factor.loc[date].dropna()
                if len(row) < 20:
                    continue
                codes = row.index
                ind = self.industry.reindex(codes) if self.industry is not None else None
                mc = self._get_date_market_cap(date)
                mc_s = mc.reindex(codes) if mc is not None else None
                style = self._get_date_style_factors(date)
                style_s = {name: sf.reindex(codes) for name, sf in style.items()}
                neutralized = _neutralize_single_date(row, ind, mc_s, style_s)
                result.loc[date, neutralized.index] = neutralized.values
            return result

        # 并行：复用模块级进程池，避免每次重新 fork
        n_workers = min(max(os.cpu_count() - 2, 1), 8)
        chunk_size = max(20, len(dates) // n_workers)
        chunks = [dates[i:i + chunk_size] for i in range(0, len(dates), chunk_size)]

        style_factors = self._style_factors if self._style_factors else None

        items = [
            (factor, chunk, self.industry, self.market_cap, style_factors)
            for chunk in chunks
        ]

        pool = _get_pool(min(n_workers, len(chunks)))
        futures = [pool.submit(_neutralize_chunk, item) for item in items]
        for future in as_completed(futures):
            chunk_result = future.result()
            for date, neutralized in chunk_result.items():
                result.loc[date, neutralized.index] = neutralized.values

        return result

    def neutralize_panels_batch(
        self, panels: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """批量风格中性化——一次池提交处理所有因子。

        将日期分块，每个块内处理所有因子，消除 O(F×C) 的池任务洪水。
        """
        if not panels:
            return {}

        # 收集所有因子共有的日期超集
        all_dates = sorted(set().union(*(p.index for p in panels.values())))
        n_dates = len(all_dates)
        if n_dates < 60:
            return {name: self.neutralize_panel(p) for name, p in panels.items()}

        n_workers = min(max(os.cpu_count() - 2, 1), 8)
        chunk_size = max(20, n_dates // n_workers)
        chunks = [all_dates[i:i + chunk_size] for i in range(0, n_dates, chunk_size)]

        style_factors = self._style_factors if self._style_factors else None

        # 每个 chunk 携带所有因子在该日期范围的切片
        chunk_items = []
        for chunk_dates in chunks:
            panel_slices = {}
            for name, panel in panels.items():
                idx = panel.index.intersection(chunk_dates)
                if len(idx) > 0:
                    panel_slices[name] = panel.loc[idx]
            if panel_slices:
                chunk_items.append((
                    panel_slices, chunk_dates, self.industry,
                    self.market_cap, style_factors,
                ))

        pool = _get_pool(min(n_workers, len(chunk_items)))
        futures = [pool.submit(_neutralize_chunk_batch, item) for item in chunk_items]

        # 初始化结果容器
        first_panel = next(iter(panels.values()))
        all_cols = first_panel.columns
        results: dict[str, pd.DataFrame] = {
            name: pd.DataFrame(index=all_dates, columns=all_cols, dtype=float)
            for name in panels
        }

        for future in as_completed(futures):
            chunk_result = future.result()
            for name, date_results in chunk_result.items():
                result_df = results[name]
                for date, neutralized in date_results.items():
                    result_df.loc[date, neutralized.index] = neutralized.values

        return results

    def evaluate(
        self,
        factor: pd.DataFrame,
        forward_period: int = 1,
        method: str = "spearman",
        filter_limit: bool = True,
        factor_neutral: pd.DataFrame | None = None,
        fwd_neutral: pd.DataFrame | None = None,
        fwd_ranked: pd.DataFrame | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """单因子完整评价 — IC / Factor Return / IR / T统计量。

        Parameters
        ----------
        factor : DataFrame
            因子面板（行=日期，列=股票代码）。
        forward_period : int
            预测周期（T+1 / T+2 / ...）。
        method : str
            ``spearman`` (Rank IC) 或 ``pearson``。
        filter_limit : bool
            是否剔除涨跌停股票（日涨跌幅绝对值 ≥ 9.5%）。
            仅在 factor_neutral 和 fwd_neutral 均未提供时生效。
        factor_neutral : DataFrame, optional
            预中性化的因子面板，提供时跳过中性化。
        fwd_neutral : DataFrame, optional
            预中性化的前向收益面板，提供时跳过中性化。
        fwd_ranked : DataFrame, optional
            预排名的前向收益面板（Spearman 专用）。
            与 fwd_neutral 配合：fwd_neutral 用于 FR OLS，
            fwd_ranked 用于 numpy 快速 IC 计算。

        Returns
        -------
        dict
            ic_mean, ic_std, ic_ir, ic_pos_ratio,
            fr_mean, fr_std, fr_ir, fr_annual, t_stat, n_periods
        """
        if factor_neutral is not None and fwd_neutral is not None:
            # 使用预中性化面板，跳过中性化和涨跌停过滤
            pass
        elif fwd_ranked is not None:
            # fwd 已预排名，factor 仍需中性化（除非提供了 factor_neutral）
            # fwd_neutral 仍需用于 FR OLS——如未提供则计算（不常见）
            factor_neutral = factor_neutral or self.neutralize_panel(factor)
            if fwd_neutral is None:
                fwd = self.forward_returns(forward_period)
                fwd_neutral = self.neutralize_panel(fwd)
        else:
            fwd = self.forward_returns(forward_period)

            if filter_limit:
                returns_daily = self.close.pct_change()
                limit_mask = returns_daily.abs() < 0.095
                factor = factor.where(limit_mask.reindex(factor.index))
                fwd = fwd.where(limit_mask.shift(-forward_period).reindex(fwd.index))

            factor_neutral = self.neutralize_panel(factor)
            fwd_neutral = self.neutralize_panel(fwd)

        common_idx = factor_neutral.index.intersection(fwd_neutral.index)
        common_cols = factor_neutral.columns.intersection(fwd_neutral.columns)
        f = factor_neutral.loc[common_idx, common_cols]
        r = fwd_neutral.loc[common_idx, common_cols]

        # IC = corr(预测Alpha, 实际Alpha)
        if fwd_ranked is not None and method == "spearman":
            # 快速路径：fwd 已预排名，仅 rank factor，用 numpy 算 Pearson (= Spearman)
            # numpy 运算释放 GIL，适合 ThreadPoolExecutor 并行
            f_ranked = f.rank(axis=1)
            r_aligned = fwd_ranked.loc[common_idx, common_cols]
            ic_series = pd.Series(
                _numpy_pearson_by_row(f_ranked.values, r_aligned.values),
                index=common_idx,
            ).dropna()
        else:
            ic_series = f.corrwith(r, axis=1, method=method).dropna()

        # Factor Return — 向量化计算，避免逐日 Python 循环
        fr_vals = _vectorized_factor_return(f.values, r.values)
        fr_series = pd.Series(fr_vals, index=common_idx).dropna()

        ic_mean = ic_series.mean() if len(ic_series) else np.nan
        ic_std = ic_series.std() if len(ic_series) else np.nan
        ic_ir = ic_mean / ic_std if ic_std and ic_std > 0 else np.nan
        ic_pos = (ic_series > 0).mean() if len(ic_series) else np.nan

        fr_mean = fr_series.mean() if len(fr_series) else np.nan
        fr_std = fr_series.std() if len(fr_series) else np.nan
        fr_ir = fr_mean / fr_std if fr_std and fr_std > 0 else np.nan

        d = forward_period
        n = len(fr_series)
        fr_annual = fr_mean * (TRADING_DAYS_PER_YEAR / d) if not np.isnan(fr_mean) else np.nan
        t_stat = (
            ic_mean / (ic_std / np.sqrt(n))
            if n > 0 and ic_std and ic_std > 0
            else np.nan
        )

        return {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "ic_pos_ratio": ic_pos,
            "fr_mean": fr_mean,
            "fr_std": fr_std,
            "fr_ir": fr_ir,
            "fr_annual": fr_annual,
            "t_stat": t_stat,
            "n_periods": n,
        }

    def evaluate_multi_horizon(
        self,
        factor: pd.DataFrame,
        periods: list[int] | None = None,
        **kwargs,
    ) -> dict[int, dict[str, float]]:
        """多周期因子评价 — 扫描不同预测周期的因子表现。"""
        if periods is None:
            periods = [1, 2, 3, 4, 5]

        results = {}
        for d in periods:
            results[d] = self.evaluate(factor, forward_period=d, **kwargs)
        return results

    def evaluate_system(
        self,
        factors: dict[str, pd.DataFrame],
        forward_period: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """因子体系批量评价 — 返回所有因子的四指标汇总表（含覆盖率）。

        预中性化 fwd 一次供所有因子复用，避免 O(F) 次冗余中性化。
        """
        global _evaluator
        _evaluator = self

        # 预中性化 + 预排名 fwd（主进程，复用模块级进程池）
        fwd = self.forward_returns(forward_period)
        fwd_neutral = self.neutralize_panel(fwd)
        fwd_ranked = fwd_neutral.rank(axis=1)  # Spearman 快速路径

        kwargs = {**kwargs, "fwd_neutral": fwd_neutral, "fwd_ranked": fwd_ranked}
        items = [(name, panel, forward_period, kwargs) for name, panel in factors.items()]
        rows: list[dict] = []
        max_workers = min(max(os.cpu_count() - 2, 1), len(items))

        # 预先计算覆盖率
        coverage_map: dict[str, float] = {}
        total_dates = len(self.close.index)
        for name, panel in factors.items():
            valid_dates = panel.notna().any(axis=1).sum()
            coverage_map[name] = valid_dates / total_dates if total_dates > 0 else 0.0

        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            total = len(items)
            futures = {executor.submit(_eval_one, item): item[0] for item in items}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    name, metrics = future.result()
                    metrics["factor"] = name
                    metrics["coverage"] = coverage_map.get(name, np.nan)
                    rows.append(metrics)
                except Exception as e:
                    name = futures[future]
                    logger.warning(f"[{i}/{total}] 评价失败 {name}: {e}")

        df = pd.DataFrame(rows).set_index("factor")
        return df.reindex(df["ic_mean"].abs().sort_values(ascending=False).index)

    def factor_correlation_matrix(
        self,
        factors: dict[str, pd.DataFrame],
        forward_period: int = 1,
        neutralized_factors: dict[str, pd.DataFrame] | None = None,
        fwd_neutral: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """因子收益率相关性矩阵 — 用于去冗余。

        使用向量化 numpy 替代逐日 Python 循环计算因子收益率。
        """
        if fwd_neutral is None:
            fwd_neutral = self.neutralize_panel(self.forward_returns(forward_period))

        if neutralized_factors is None:
            neutralized_factors = {}
            for name, panel in factors.items():
                neutralized_factors[name] = self.neutralize_panel(panel)

        fr_dict: dict[str, np.ndarray] = {}
        common_idx = fwd_neutral.index
        common_cols = fwd_neutral.columns
        r_vals = fwd_neutral.loc[common_idx, common_cols].values  # (n_dates, n_stocks)

        for name, panel in neutralized_factors.items():
            c_idx = panel.index.intersection(common_idx)
            c_cols = panel.columns.intersection(common_cols)
            f_vals = panel.loc[c_idx, c_cols].values
            r = fwd_neutral.loc[c_idx, c_cols].values
            fr_dict[name] = pd.Series(
                _vectorized_factor_return(f_vals, r),
                index=c_idx,
            )

        fr_df = pd.DataFrame(fr_dict)
        return fr_df.corr()

    def evaluate_model_predictive_power(
        self,
        factors: dict[str, pd.DataFrame],
        weights: dict[str, float] | None = None,
        forward_period: int = 1,
        method: str = "spearman",
        neutralized_factors: dict[str, pd.DataFrame] | None = None,
        fwd_neutral: pd.DataFrame | None = None,
        fwd_ranked: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """因子体系整体预测力评估 — 合成 Alpha 预测的 IC。

        将所有因子中性化后按权重合成为单一 Alpha 预测值 E(ε)，
        计算其与风格中性化实际 Alpha 收益的截面 IC。
        对应 Alpha191 报告的“Alpha 模型预测力”指标。

        Parameters
        ----------
        factors : dict
            因子名 → 因子面板（行=日期，列=股票代码）。
        weights : dict, optional
            因子名 → 权重。省略时等权合成。
        forward_period : int
            预测周期。
        method : str
            ``spearman`` 或 ``pearson``。
        neutralized_factors : dict, optional
            预中性化的因子面板，提供时跳过逐因子中性化。
        fwd_neutral : DataFrame, optional
            预中性化的前向收益面板，提供时跳过中性化步骤。
        fwd_ranked : DataFrame, optional
            预排名的前向收益面板，Spearman 下走 numpy 快速路径。

        Returns
        -------
        dict
            ic_mean, ic_std, ic_ir, ic_pos_ratio, t_stat, n_periods
        """
        if not factors:
            return {}

        if fwd_neutral is None:
            fwd_neutral = self.neutralize_panel(self.forward_returns(forward_period))

        if neutralized_factors is None:
            neutralized_factors = {}
            for name in factors:
                neutralized_factors[name] = self.neutralize_panel(factors[name])

        if weights is None:
            weights = {name: 1.0 / len(factors) for name in factors}

        alpha_pred: pd.DataFrame | None = None
        n_factors = len(neutralized_factors)
        report_every = max(20, n_factors // 5)
        for i, (name, panel) in enumerate(neutralized_factors.items()):
            if name not in weights or weights[name] == 0:
                continue
            z = (panel - panel.mean()) / (panel.std() + 1e-10)
            w = weights[name]
            if alpha_pred is None:
                alpha_pred = z * w
            else:
                alpha_pred = alpha_pred + z * w
            if (i + 1) % report_every == 0:
                print(f"  ... 已合成 {i + 1}/{n_factors} 个因子", flush=True)

        if alpha_pred is None:
            return {}

        print(f"  计算 {len(alpha_pred)} 期截面 IC...", flush=True)
        common_idx = alpha_pred.index.intersection(fwd_neutral.index)
        common_cols = alpha_pred.columns.intersection(fwd_neutral.columns)
        pred = alpha_pred.loc[common_idx, common_cols]
        actual = fwd_neutral.loc[common_idx, common_cols]

        if fwd_ranked is not None and method == "spearman":
            pred_ranked = pred.rank(axis=1)
            r_aligned = fwd_ranked.loc[common_idx, common_cols]
            ic_series = pd.Series(
                _numpy_pearson_by_row(pred_ranked.values, r_aligned.values),
                index=common_idx,
            ).dropna()
        else:
            ic_series = pred.corrwith(actual, axis=1, method=method).dropna()

        ic_mean = ic_series.mean() if len(ic_series) else np.nan
        ic_std = ic_series.std() if len(ic_series) else np.nan
        ic_ir = ic_mean / ic_std if ic_std and ic_std > 0 else np.nan
        ic_pos = (ic_series > 0).mean() if len(ic_series) else np.nan
        n = len(ic_series)
        t_stat = (
            ic_mean / (ic_std / np.sqrt(n))
            if n > 0 and ic_std and ic_std > 0
            else np.nan
        )

        return {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "ic_pos_ratio": ic_pos,
            "t_stat": t_stat,
            "n_periods": n,
        }

    def factor_count_analysis(
        self,
        factors: dict[str, pd.DataFrame],
        forward_period: int = 1,
        method: str = "spearman",
        neutralized_factors: dict[str, pd.DataFrame] | None = None,
        fwd_neutral: pd.DataFrame | None = None,
        fr_ir_map: dict[str, float] | None = None,
        fwd_ranked: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """因子数量边际分析 — 按 FR IR 降序逐步增加因子，输出 IC 边际提升曲线。

        对应 Alpha191 报告的“因子数量与预测力”分析。

        预中性化所有因子面板和 forward returns 各一次，避免
        反复 spawn ProcessPoolExecutor。

        Parameters
        ----------
        factors : dict
            因子名 → 因子面板。
        forward_period : int
            预测周期。
        method : str
            ``spearman`` 或 ``pearson``。
        neutralized_factors : dict, optional
            预中性化的因子面板，提供时跳过中性化。
        fwd_neutral : DataFrame, optional
            预中性化的前向收益面板，提供时跳过中性化步骤。
        fr_ir_map : dict, optional
            因子名 → FR IR 的预计算映射。提供时跳过 Phase 2
            (O(F·d·s·k)) 的逐一 IC/FR 重算，直接按此排名。

        Returns
        -------
        DataFrame
            n_factors, ic_mean, ic_std, ic_ir, t_stat, added_factor
        """
        if not factors:
            return pd.DataFrame()

        if fwd_neutral is None:
            fwd_neutral = self.neutralize_panel(self.forward_returns(forward_period))

        if neutralized_factors is None:
            neutralized: dict[str, pd.DataFrame] = {}
            for name, panel in factors.items():
                neutralized[name] = self.neutralize_panel(panel)
        else:
            neutralized = neutralized_factors

        # Phase 2: 按 |FR IR| 降序排名
        if fr_ir_map is not None:
            # 直接使用预计算 FR IR —— 跳过 O(F·d·s·k) 的重算
            rankings = [(name, abs(fr_ir_map.get(name, 0) or 0))
                        for name in neutralized]
            rankings.sort(key=lambda x: x[1], reverse=True)
            print(f"  使用预计算 FR IR 排名 ({len(rankings)} 因子), 跳过 Phase 2 重算", flush=True)
        else:
            rankings = []
            nf_total = len(neutralized)
            rank_report_every = max(10, nf_total // 10)
            for i, (name, panel) in enumerate(neutralized.items()):
                metrics = self.evaluate(
                    panel, forward_period=forward_period, method=method,
                    factor_neutral=panel, fwd_neutral=fwd_neutral,
                    fwd_ranked=fwd_ranked,
                )
                fr_ir = abs(metrics.get("fr_ir", 0) or 0)
                rankings.append((name, fr_ir))
                if (i + 1) % rank_report_every == 0:
                    print(f"  ... 已排名 {i + 1}/{nf_total} 个因子", flush=True)
            rankings.sort(key=lambda x: x[1], reverse=True)

        # Phase 3: 数量边际分析 — 维护运行和避免 O(n²) 重复计算
        rows = []
        z_sum: pd.DataFrame | None = None  # running sum of z-scored panels
        total = len(rankings)
        report_every = max(5, total // 10)

        for i, (name, _) in enumerate(rankings):
            panel = neutralized[name]
            z = (panel - panel.mean()) / (panel.std() + 1e-10)
            if z_sum is None:
                z_sum = z
            else:
                z_sum = z_sum + z

            n = i + 1
            alpha_pred = z_sum / n  # 等权合成

            common_idx = alpha_pred.index.intersection(fwd_neutral.index)
            common_cols = alpha_pred.columns.intersection(fwd_neutral.columns)
            pred = alpha_pred.loc[common_idx, common_cols]
            actual = fwd_neutral.loc[common_idx, common_cols]

            if fwd_ranked is not None and method == "spearman":
                pred_ranked = pred.rank(axis=1)
                r_aligned = fwd_ranked.loc[common_idx, common_cols]
                ic_series = pd.Series(
                    _numpy_pearson_by_row(pred_ranked.values, r_aligned.values),
                    index=common_idx,
                ).dropna()
            else:
                ic_series = pred.corrwith(actual, axis=1, method=method).dropna()
            ic_mean = ic_series.mean() if len(ic_series) else np.nan
            ic_std = ic_series.std() if len(ic_series) else np.nan
            ic_ir = ic_mean / ic_std if ic_std and ic_std > 0 else np.nan
            n_periods = len(ic_series)
            t_stat = (
                ic_mean / (ic_std / np.sqrt(n_periods))
                if n_periods > 0 and ic_std and ic_std > 0
                else np.nan
            )

            rows.append({
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_ir": ic_ir,
                "t_stat": t_stat,
                "n_factors": n,
                "added_factor": name,
            })

            if (i + 1) % report_every == 0:
                print(f"  ... {i + 1}/{total} 因子, ICIR={ic_ir:.4f}", flush=True)

        return pd.DataFrame(rows).set_index("n_factors")

    def coverage_report(
        self,
        factors: dict[str, pd.DataFrame],
    ) -> dict:
        """因子覆盖率报告 — 统计各因子有效数据占比。

        Returns
        -------
        dict
            min_coverage, median_coverage, max_coverage,
            low_coverage_factors (list of (name, coverage) for <30%),
            all_coverage (dict name → coverage)
        """
        if not factors:
            return {}

        total_dates = len(self.close.index)
        coverage: dict[str, float] = {}
        for name, panel in factors.items():
            valid_dates = panel.notna().any(axis=1).sum()
            coverage[name] = valid_dates / total_dates if total_dates > 0 else 0.0

        cov_vals = list(coverage.values())
        low = [(n, c) for n, c in coverage.items() if c < 0.30]
        low.sort(key=lambda x: x[1])

        return {
            "min_coverage": min(cov_vals) if cov_vals else np.nan,
            "median_coverage": float(np.median(cov_vals)) if cov_vals else np.nan,
            "max_coverage": max(cov_vals) if cov_vals else np.nan,
            "low_coverage_factors": low,
            "all_coverage": coverage,
        }
