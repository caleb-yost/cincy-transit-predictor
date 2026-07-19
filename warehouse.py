"""Shared warehouse connection: MotherDuck in production, local DuckDB in dev.

Both the ML training job and the Streamlit app import this so they read the same marts.
Set MOTHERDUCK_TOKEN (env var or Streamlit secret) to use the cloud warehouse; otherwise
it falls back to the local DuckDB file dbt writes under transform/.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

_ROOT = Path(__file__).resolve().parent
LOCAL_WAREHOUSE = os.getenv("DBT_DUCKDB_PATH", str(_ROOT / "transform" / "warehouse.duckdb"))
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cincy_transit")


def _secret(name: str) -> str:
    """Read a secret from the environment, then Streamlit secrets if available."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return str(st.secrets.get(name, ""))
    except Exception:
        return ""


def using_motherduck() -> bool:
    return bool(_secret("MOTHERDUCK_TOKEN"))


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    token = _secret("MOTHERDUCK_TOKEN")
    if token:
        return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={token}")
    return duckdb.connect(LOCAL_WAREHOUSE, read_only=read_only)
