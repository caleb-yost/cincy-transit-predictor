"""Create the MotherDuck database if it doesn't exist yet.

dbt attaches an existing database via `md:cincy_transit`, which fails on a fresh MotherDuck
account. This runs once before `dbt build` to create it (idempotent).
"""

from __future__ import annotations

import os

import duckdb

DB = os.getenv("MOTHERDUCK_DB", "cincy_transit")


def main() -> None:
    con = duckdb.connect("md:?motherduck_token=" + os.environ["MOTHERDUCK_TOKEN"])
    existing = {row[0] for row in con.execute("SHOW DATABASES").fetchall()}
    if DB in existing:
        print(f"MotherDuck database '{DB}' already exists")
    else:
        con.execute(f"CREATE DATABASE {DB}")
        print(f"created MotherDuck database '{DB}'")


if __name__ == "__main__":
    main()
