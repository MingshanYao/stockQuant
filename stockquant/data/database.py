"""
DuckDB 数据库封装 — 本地高性能列式存储。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from stockquant.utils.logger import get_logger

logger = get_logger("data.database")


class Database:
    """DuckDB 数据库管理器。

    Parameters
    ----------
    db_path : str | Path
        数据库文件路径，默认使用配置中的路径。
    read_only : bool
        是否以只读模式打开。
    """

    def __init__(self, db_path: str | Path | None = None, read_only: bool = False) -> None:
        if db_path is None:
            from stockquant.utils.config import Config
            cfg = Config()
            db_path = cfg.resolve_path("database.path", "./stockquant/data/db/stockquant.duckdb")

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._read_only = read_only
        self._conn: duckdb.DuckDBPyConnection | None = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(
                str(self._db_path), read_only=self._read_only
            )
            logger.info(f"已连接 DuckDB: {self._db_path}")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # 表操作
    # ------------------------------------------------------------------

    def init_tables(self) -> None:
        """初始化核心表结构（含自动迁移旧表）。"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                code        VARCHAR NOT NULL,
                date        DATE NOT NULL,
                open        DOUBLE,
                high        DOUBLE,
                low         DOUBLE,
                close       DOUBLE,
                pre_close   DOUBLE,
                volume      BIGINT,
                amount      DOUBLE,
                vwap        DOUBLE,
                turnover    DOUBLE,
                pct_change  DOUBLE,
                adj_factor  DOUBLE,
                PRIMARY KEY (code, date)
            )
        """)
        self._migrate_daily_bars()

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_info (
                code          VARCHAR PRIMARY KEY,
                name          VARCHAR,
                industry      VARCHAR,
                sector        VARCHAR,
                market        VARCHAR,
                list_date     DATE,
                total_shares  DOUBLE,
                float_shares  DOUBLE,
                total_cap     DOUBLE,
                float_cap     DOUBLE,
                updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 兼容旧表：若缺少新字段则自动追加
        self._ensure_stock_info_columns()

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS index_daily (
                code   VARCHAR NOT NULL,
                date   DATE NOT NULL,
                open   DOUBLE,
                high   DOUBLE,
                low    DOUBLE,
                close  DOUBLE,
                volume BIGINT,
                amount DOUBLE,
                PRIMARY KEY (code, date)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS financials (
                code          VARCHAR NOT NULL,
                report_date   DATE NOT NULL,
                pub_date      DATE,
                roe           DOUBLE,
                eps           DOUBLE,
                net_profit    DOUBLE,
                revenue       DOUBLE,
                gp_margin     DOUBLE,
                np_margin     DOUBLE,
                total_shares  DOUBLE,
                float_shares  DOUBLE,
                growth_equity DOUBLE,
                growth_asset  DOUBLE,
                growth_ni     DOUBLE,
                current_ratio DOUBLE,
                debt_ratio    DOUBLE,
                PRIMARY KEY (code, report_date)
            )
        """)

        logger.info("数据库表初始化完成")

    def _ensure_stock_info_columns(self) -> None:
        """兼容旧版 stock_info 表：若缺少新字段则 ALTER TABLE 追加。"""
        expected = {
            "industry": "VARCHAR",
            "sector": "VARCHAR",
            "total_shares": "DOUBLE",
            "float_shares": "DOUBLE",
            "total_cap": "DOUBLE",
            "float_cap": "DOUBLE",
            "out_date": "DATE",
            "status": "INTEGER",
            "industry_source": "VARCHAR",
        }
        try:
            existing = {
                row[0]
                for row in self.conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'stock_info'"
                ).fetchall()
            }
            for col, dtype in expected.items():
                if col not in existing:
                    self.conn.execute(
                        f"ALTER TABLE stock_info ADD COLUMN {col} {dtype}"
                    )
                    logger.info(f"stock_info 表追加列: {col} {dtype}")
        except Exception as e:
            logger.warning(f"stock_info 列检查/追加失败: {e}")

    def _migrate_daily_bars(self) -> None:
        """兼容旧版 daily_bars 表：新增 pre_close / vwap / adj_factor，移除 change。"""
        expected_new = {
            "pre_close": "DOUBLE",
            "vwap": "DOUBLE",
            "adj_factor": "DOUBLE",
        }
        try:
            existing = {
                row[0]
                for row in self.conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'daily_bars'"
                ).fetchall()
            }
        except Exception:
            return  # 表可能尚未创建

        # 追加缺失的新列
        for col, dtype in expected_new.items():
            if col not in existing:
                try:
                    self.conn.execute(
                        f"ALTER TABLE daily_bars ADD COLUMN {col} {dtype}"
                    )
                    logger.info(f"daily_bars 表追加列: {col} {dtype}")
                except Exception as e:
                    logger.warning(f"daily_bars 追加列 {col} 失败: {e}")

        # 回填已存在数据：vwap = amount / volume, adj_factor = 1.0
        if "vwap" in expected_new and "vwap" in expected_new:
            try:
                self.conn.execute("""
                    UPDATE daily_bars
                    SET vwap = amount / NULLIF(volume, 0)
                    WHERE vwap IS NULL AND amount IS NOT NULL AND volume > 0
                """)
            except Exception as e:
                logger.debug(f"vwap 回填跳过: {e}")

        if "adj_factor" in expected_new:
            try:
                self.conn.execute("""
                    UPDATE daily_bars SET adj_factor = 1.0 WHERE adj_factor IS NULL
                """)
            except Exception as e:
                logger.debug(f"adj_factor 回填跳过: {e}")

        if "pre_close" in expected_new:
            try:
                # 如果有 pct_change 列，从 close 和 pct_change 反推
                if "pct_change" in existing:
                    self.conn.execute("""
                        UPDATE daily_bars
                        SET pre_close = close / (1 + pct_change / 100)
                        WHERE pre_close IS NULL AND pct_change IS NOT NULL
                    """)
                # 兜底：用 close.shift(1)（按 date 排序后）
                self.conn.execute("""
                    UPDATE daily_bars
                    SET pre_close = (
                        SELECT b.close FROM daily_bars b
                        WHERE b.code = daily_bars.code
                          AND b.date < daily_bars.date
                        ORDER BY b.date DESC LIMIT 1
                    )
                    WHERE pre_close IS NULL
                """)
            except Exception as e:
                logger.debug(f"pre_close 回填跳过: {e}")

    # ------------------------------------------------------------------
    # 写入：三种明确语义
    # ------------------------------------------------------------------

    def insert_or_ignore(self, df: pd.DataFrame, table: str) -> int:
        """INSERT OR IGNORE：主键冲突静默跳过，返回真实新增行数。"""
        if df is None or df.empty:
            return 0

        before = self._row_count(table)
        tmp = f"_tmp_{table}"
        self.conn.register(tmp, df)
        cols = ", ".join(df.columns)
        try:
            self.conn.execute(
                f"INSERT OR IGNORE INTO {table} ({cols}) SELECT {cols} FROM {tmp}"
            )
        finally:
            self.conn.unregister(tmp)
        inserted = self._row_count(table) - before
        logger.debug(f"insert_or_ignore {table}: 提交 {len(df)} 行，新增 {inserted} 行")
        return inserted

    def upsert(self, df: pd.DataFrame, table: str) -> int:
        """INSERT OR REPLACE：按主键合并，返回提交行数。

        要求表已声明 PRIMARY KEY。
        """
        if df is None or df.empty:
            return 0

        tmp = f"_tmp_{table}"
        self.conn.register(tmp, df)
        cols = ", ".join(df.columns)
        try:
            self.conn.execute(
                f"INSERT OR REPLACE INTO {table} ({cols}) SELECT {cols} FROM {tmp}"
            )
        finally:
            self.conn.unregister(tmp)
        logger.debug(f"upsert {table}: 提交 {len(df)} 行")
        return len(df)

    def truncate(self, table: str) -> None:
        """清空表所有行。"""
        self.conn.execute(f"DELETE FROM {table}")
        logger.info(f"truncate {table}")

    def save_dataframe(self, df: pd.DataFrame, table: str) -> int:
        """兼容旧接口 —— 等同于 :meth:`insert_or_ignore`。

        新代码请直接使用 :meth:`insert_or_ignore` / :meth:`upsert` / :meth:`truncate`，
        语义更清晰。
        """
        return self.insert_or_ignore(df, table)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        """执行 SQL 查询并返回 DataFrame。"""
        if params:
            return self.conn.execute(sql, params).fetchdf()
        return self.conn.execute(sql).fetchdf()

    def execute(self, sql: str, params: list[Any] | None = None) -> None:
        """执行 SQL（不返回结果）。"""
        if params:
            self.conn.execute(sql, params)
        else:
            self.conn.execute(sql)

    def table_exists(self, table: str) -> bool:
        """检查表是否存在。"""
        result = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()
        return result[0] > 0 if result else False

    def get_latest_date(self, table: str, code: str) -> str | None:
        """获取指定股票在表中的最新日期。"""
        try:
            result = self.conn.execute(
                f"SELECT MAX(date) FROM {table} WHERE code = ?", [code]
            ).fetchone()
            return str(result[0]) if result and result[0] else None
        except Exception:
            return None

    def _row_count(self, table: str) -> int:
        row = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
