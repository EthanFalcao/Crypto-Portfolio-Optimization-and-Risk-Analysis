"""Shared Turso (libsql) connection, used by every notebook in the pipeline.

Needs TURSO_DATABASE_URL and TURSO_AUTH_TOKEN, either in a local .env file
(see .env.example) or already set as environment variables. Get these by
creating a free database at https://turso.tech.
"""
import os

import libsql
from dotenv import load_dotenv

load_dotenv()

# Local on-disk replica that libsql syncs with Turso. Safe to delete anytime
# (it just gets re-synced from the cloud on next connect).
LOCAL_REPLICA_PATH = "local_replica.db"


def get_connection():
    url = os.environ.get("TURSO_DATABASE_URL")
    token = os.environ.get("TURSO_AUTH_TOKEN")
    if not url or not token:
        raise RuntimeError(
            "TURSO_DATABASE_URL and TURSO_AUTH_TOKEN are not set. "
            "Copy .env.example to .env and fill them in."
        )
    conn = libsql.connect(LOCAL_REPLICA_PATH, sync_url=url, auth_token=token)
    conn.sync()
    return conn


def save_df(df, table_name):
    """Replace a table's contents with a dataframe.

    Turso forwards every write to the remote database individually, so a
    plain row-at-a-time insert (which is what pandas' df.to_sql does) takes
    ~150ms per row - fine for a handful of rows, unusable for a table with
    thousands of them. Instead we build multi-row INSERT statements (one
    round trip per chunk of rows, not per row).
    """
    import sqlite3
    import pandas as pd

    # libsql can't bind datetime/Timestamp params directly - store as ISO text
    df = df.copy()
    for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        df[col] = df[col].astype(str)

    # Reuse pandas' own SQLite type inference for the CREATE TABLE statement
    # (built locally, in memory - no network involved). Do this before the
    # NaN->None conversion below, since that flattens every column to dtype
    # object and would otherwise make pandas infer everything as TEXT.
    tmp_conn = sqlite3.connect(":memory:")
    df.to_sql(table_name, tmp_conn, index=False)
    create_stmt = tmp_conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()[0]
    tmp_conn.close()

    # NaN isn't valid JSON, and Turso's remote protocol is JSON-based - a bare
    # NaN gets mangled into something the server rejects. Use real SQL NULL.
    df = df.astype(object).where(pd.notnull(df), None)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    cur.execute(create_stmt)

    n_cols = len(df.columns)
    chunk_size = max(1, 900 // n_cols)  # stay well under SQLite's per-statement parameter limit
    row_placeholder = "(" + ",".join(["?"] * n_cols) + ")"
    records = list(df.itertuples(index=False, name=None))

    for start in range(0, len(records), chunk_size):
        chunk = records[start:start + chunk_size]
        stmt = f"INSERT INTO {table_name} VALUES " + ",".join([row_placeholder] * len(chunk))
        params = [value for row in chunk for value in row]
        cur.execute(stmt, params)

    conn.commit()
    conn.sync()
    conn.close()


def load_df(table_name):
    """Read a full table into a dataframe."""
    import pandas as pd

    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df
