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
        """初始化核心表结构。"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                code        VARCHAR NOT NULL,
                date        DATE NOT NULL,
                open        DOUBLE,
                high        DOUBLE,
                low         DOUBLE,
                close       DOUBLE,
                volume      BIGINT,
                amount      DOUBLE,
                turnover    DOUBLE,
                pct_change  DOUBLE,
                change      DOUBLE,
                PRIMARY KEY (code, date)
            )
        """)

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

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save_dataframe(
        self,
        df: pd.DataFrame,
        table: str,
        if_exists: str = "append",
    ) -> int:
        """将 DataFrame 写入指定表。

        Parameters
        ----------
        df : DataFrame
            待写入数据。
        table : str
            目标表名。
        if_exists : str
            冲突处理: ``append`` 追加（跳过重复）/ ``replace`` 替换。

        Returns
        -------
        int
            写入行数。
        """
        if df.empty:
            return 0

        if if_exists == "replace":
            self.conn.execute(f"DELETE FROM {table}")

        # 使用临时表做 INSERT OR IGNORE（显式列名，避免列顺序不一致）
        tmp = f"_tmp_{table}"
        self.conn.register(tmp, df)
        cols = ", ".join(df.columns)
        self.conn.execute(f"""
            INSERT OR IGNORE INTO {table} ({cols})
            SELECT {cols} FROM {tmp}
        """)
        self.conn.unregister(tmp)

        row_count = len(df)
        logger.debug(f"写入 {table}: {row_count} 行")
        return row_count

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
